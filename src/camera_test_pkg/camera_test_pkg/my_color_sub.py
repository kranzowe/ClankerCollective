import rclpy
from rclpy.node import Node
import numpy as np
import time

from vision_msgs.msg import BoundingBox2D
from sensor_msgs.msg import Image
from nav_2d_msgs.msg import Path2D
from geometry_msgs.msg import Pose2D


# ──────────────────────────────────────────────────────────────────────────────
#  Pure-NumPy helpers  (no OpenCV)
# ──────────────────────────────────────────────────────────────────────────────

def bgr_to_hls(bgr: np.ndarray) -> np.ndarray:
    """
    Convert an HxWx3 uint8 BGR image to HLS (Hue, Lightness, Saturation).
    Hue   → [0, 255]  (full circle, same convention as cv2.COLOR_BGR2HLS_FULL)
    L / S → [0, 255]
    """
    # Normalise to [0, 1]
    img = bgr.astype(np.float32) / 255.0
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    c_max = np.maximum(np.maximum(r, g), b)
    c_min = np.minimum(np.minimum(r, g), b)
    delta = c_max - c_min

    # --- Lightness ---
    l = (c_max + c_min) / 2.0

    # --- Saturation ---
    s = np.where(delta == 0, 0.0, delta / (1.0 - np.abs(2.0 * l - 1.0)))

    # --- Hue (full 0-360 → rescaled to 0-255) ---
    eps = 1e-6
    h = np.zeros_like(delta)

    mask_r = (c_max == r) & (delta > eps)
    mask_g = (c_max == g) & (delta > eps)
    mask_b = (c_max == b) & (delta > eps)

    h[mask_r] = (60.0 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360.0
    h[mask_g] = (60.0 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120.0) % 360.0
    h[mask_b] = (60.0 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240.0) % 360.0

    # Rescale H 0-360 → 0-255  (matches HLS_FULL convention)
    h = h / 360.0 * 255.0

    hls = np.stack([h, l * 255.0, s * 255.0], axis=-1).astype(np.uint8)
    return hls


def gaussian_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Separable Gaussian blur.  ksize must be odd.
    Works on single-channel or multi-channel uint8 images.
    """
    # Build 1-D kernel
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    x = np.arange(ksize) - ksize // 2
    kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    out = img.astype(np.float32)
    pad = ksize // 2

    # Horizontal pass
    out_h = np.pad(out, ((0, 0), (pad, pad), (0, 0)), mode='edge') if out.ndim == 3 \
        else np.pad(out, ((0, 0), (pad, pad)), mode='edge')
    out = np.apply_along_axis(lambda row: np.convolve(row, kernel_1d, mode='valid'), 1, out_h) \
        if out.ndim == 2 else \
        np.stack([np.apply_along_axis(lambda row: np.convolve(row, kernel_1d, mode='valid'),
                                      1, out_h[:, :, c]) for c in range(out_h.shape[2])], axis=2)

    # Vertical pass
    out_v = np.pad(out, ((pad, pad), (0, 0), (0, 0)), mode='edge') if out.ndim == 3 \
        else np.pad(out, ((pad, pad), (0, 0)), mode='edge')
    out = np.apply_along_axis(lambda col: np.convolve(col, kernel_1d, mode='valid'), 0, out_v) \
        if out.ndim == 2 else \
        np.stack([np.apply_along_axis(lambda col: np.convolve(col, kernel_1d, mode='valid'),
                                      0, out_v[:, :, c]) for c in range(out_v.shape[2])], axis=2)

    return np.clip(out, 0, 255).astype(np.uint8)


def in_range(img: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Element-wise per-channel threshold; returns a bool mask (HxW)."""
    mask = np.all((img >= lower) & (img <= upper), axis=-1)
    return mask.astype(np.uint8) * 255


def morphology_open_close(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Binary open then close using a square structuring element.
    Implemented with 2-D max/min sliding-window via stride tricks.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    def dilate(m, k):
        pad = k // 2
        padded = np.pad(m, pad, mode='constant', constant_values=0)
        windows = sliding_window_view(padded, (k, k))
        return (windows.max(axis=(-2, -1)) > 0).astype(np.uint8) * 255

    def erode(m, k):
        pad = k // 2
        padded = np.pad(m, pad, mode='constant', constant_values=255)
        windows = sliding_window_view(padded, (k, k))
        return (windows.min(axis=(-2, -1)) > 0).astype(np.uint8) * 255

    # Open  = erode then dilate  → removes small blobs
    opened = dilate(erode(mask, ksize), ksize)
    # Close = dilate then erode  → fills small holes
    closed = erode(dilate(opened, ksize), ksize)
    return closed


def label_connected_components(mask: np.ndarray):
    """
    Simple flood-fill connected-component labelling on a binary mask.
    Returns (label_map, num_labels) where label_map is HxW int32.
    Labels are 1-indexed; 0 = background.
    """
    binary = (mask > 0)
    labels = np.zeros_like(binary, dtype=np.int32)
    current_label = 0
    rows, cols = binary.shape

    for r in range(rows):
        for c in range(cols):
            if binary[r, c] and labels[r, c] == 0:
                current_label += 1
                # BFS
                queue = [(r, c)]
                labels[r, c] = current_label
                while queue:
                    cr, cc = queue.pop()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols \
                                and binary[nr, nc] and labels[nr, nc] == 0:
                            labels[nr, nc] = current_label
                            queue.append((nr, nc))

    return labels, current_label


def find_blobs(mask: np.ndarray):
    """
    Return a list of dicts, one per connected component:
      { 'area', 'cx', 'cy', 'x', 'y', 'w', 'h' }
    """
    labels, n = label_connected_components(mask)
    blobs = []
    for lbl in range(1, n + 1):
        coords = np.argwhere(labels == lbl)   # (N, 2) → rows, cols
        if len(coords) == 0:
            continue
        rows_lbl = coords[:, 0]
        cols_lbl = coords[:, 1]
        area = len(coords)
        cy = int(rows_lbl.mean())
        cx = int(cols_lbl.mean())
        y = int(rows_lbl.min())
        x = int(cols_lbl.min())
        h = int(rows_lbl.max()) - y + 1
        w = int(cols_lbl.max()) - x + 1
        blobs.append(dict(area=area, cx=cx, cy=cy, x=x, y=y, w=w, h=h))
    return blobs


# ──────────────────────────────────────────────────────────────────────────────
#  ROS2 Node
# ──────────────────────────────────────────────────────────────────────────────

class ImageListener(Node):
    def __init__(self):
        super().__init__('image_listener')

        # --- subscriptions / publications ---
        self.sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.listener_cb, 10)

        self.rectangle_pub = self.create_publisher(BoundingBox2D, '/blob_rectangle', 10)
        self.path_pub      = self.create_publisher(Path2D,        '/line_path',       10)

        self.prev_point = None

        # --- ROS Image → NumPy array bridge (manual, no CvBridge needed) ---
        # We handle decoding directly from sensor_msgs/Image raw data.

        # --- HSV / HLS parameters (focus point + tolerance) ---
        self.declare_parameter("fh", 120)
        self.declare_parameter("fs", 180)
        self.declare_parameter("fl", 120)

        self.declare_parameter("th", 45)
        self.declare_parameter("ts", 80)
        self.declare_parameter("tl", 100)

        self.create_timer(1.0, self.update_params)

        self.hls_lower = np.zeros(3, dtype=np.float32)
        self.hls_upper = np.zeros(3, dtype=np.float32)

        self.get_logger().info("ImageListener node started (no-OpenCV version).")

    # --------------------------------------------------------------------------
    def ros_image_to_numpy(self, msg: Image) -> np.ndarray:
        """
        Convert a sensor_msgs/Image to an HxWx3 uint8 BGR numpy array.
        Supports encoding: bgr8, rgb8, rgba8, bgra8.
        """
        dtype = np.uint8
        channels_map = {
            'bgr8':  ('bgr',  3),
            'rgb8':  ('rgb',  3),
            'rgba8': ('rgba', 4),
            'bgra8': ('bgra', 4),
        }
        if msg.encoding not in channels_map:
            raise ValueError(f"Unsupported encoding: {msg.encoding}")

        order, n_ch = channels_map[msg.encoding]
        img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, n_ch)

        # Ensure output is BGR
        if order == 'rgb':
            img = img[:, :, ::-1]          # RGB → BGR
        elif order == 'rgba':
            img = img[:, :, [2, 1, 0]]     # RGBA → BGR  (drop alpha)
        elif order == 'bgra':
            img = img[:, :, :3]            # drop alpha

        return img.copy()

    # --------------------------------------------------------------------------
    def image_to_robot(self, pt, width, height):
        px, py = pt
        x_norm = (px - width / 2) / (width / 2)
        y_norm = ((height - py) / height) ** 1.5

        forward_scale = 1.0
        lateral_scale = 1.4

        x_robot = y_norm * forward_scale
        y_robot = x_norm * lateral_scale
        return x_robot, y_robot

    # --------------------------------------------------------------------------
    def listener_cb(self, data: Image):
        frame  = self.ros_image_to_numpy(data)   # HxWx3 BGR uint8
        height, width = frame.shape[:2]
        prev_point = self.prev_point

        # ---- Crop top region out ----
        crop_ratio = 0.75
        start_y    = int(height * (1 - crop_ratio))
        frame      = frame[start_y:height, :]        # work on cropped copy

        # ---- Blur → HLS → threshold ----
        blur_frame = gaussian_blur(frame, ksize=5)
        hls_frame  = bgr_to_hls(blur_frame)

        lower = self.hls_lower.astype(np.uint8)
        upper = self.hls_upper.astype(np.uint8)

        mask = in_range(hls_frame, lower, upper)

        # ---- Morphological open + close ----
        mask = morphology_open_close(mask, ksize=5)

        # ---- Largest-blob bounding box (mirrors original rectangle publisher) ----
        all_blobs = find_blobs(mask)
        if all_blobs:
            biggest = max(all_blobs, key=lambda b: b['area'])
            if biggest['area'] > 500:
                rectangle_msg = BoundingBox2D()
                rectangle_msg.center.position.x = float(biggest['cx'])
                rectangle_msg.center.position.y = float(biggest['cy'])
                rectangle_msg.center.theta       = 0.0
                rectangle_msg.size_x             = float(biggest['w'])
                rectangle_msg.size_y             = float(biggest['h'])
                self.rectangle_pub.publish(rectangle_msg)

        # ---- Band-based path extraction ----
        cropped_height = frame.shape[0]
        num_bands  = 8
        band_height = cropped_height // num_bands
        chosen_points = []

        for i in range(num_bands):
            y1 = i * band_height
            y2 = (i + 1) * band_height if i < num_bands - 1 else cropped_height

            band_mask = mask[y1:y2, :]
            blobs     = find_blobs(band_mask)

            best_score = -1e18
            best_point = None

            for blob in blobs:
                area = blob['area']
                if area < 45:
                    continue

                # Convert band-local centroid to global (cropped) frame coords
                cx = blob['cx']
                cy = y1 + blob['cy']

                x_norm = (cx - width / 2) / (width / 2)
                x_bias = (x_norm + 1) / 2   # 0=left, 1=right, only for scoring

                continuity_penalty = 0.0
                if prev_point is not None:
                    continuity_penalty = abs(cx - prev_point[0]) * 2.0

                area_weight  = 1.0
                right_weight = 2000.0

                score = (area * area_weight) + (x_bias * right_weight) - continuity_penalty

                if x_bias > 0.7:
                    score += 1500.0

                if score > best_score:
                    best_score = score
                    best_point = (cx, cy)

            chosen_points.append(best_point)

            if best_point is not None:
                prev_point = best_point

        # ---- No blobs: go straight ----
        if not any(pt is not None for pt in chosen_points):
            straight_point = (width // 2, cropped_height // 2)
            chosen_points  = [straight_point]

        # ---- Build and publish Path2D ----
        path_msg = Path2D()
        path_msg.poses = []

        valid_points = [pt for pt in chosen_points if pt is not None]

        for i, pt in enumerate(valid_points):
            x_r, y_r = self.image_to_robot(pt, width, cropped_height)

            pose   = Pose2D()
            pose.x = float(x_r)
            pose.y = float(y_r)

            if i < len(valid_points) - 1:
                x_next, y_next = self.image_to_robot(valid_points[i + 1], width, cropped_height)
                pose.theta = float(np.arctan2(y_r, x_r))
            else:
                pose.theta = 0.0

            path_msg.poses.append(pose)

        if len(path_msg.poses) >= 1:
            self.path_pub.publish(path_msg)

        self.prev_point = prev_point

    # --------------------------------------------------------------------------
    def update_params(self):
        h     = self.get_parameter("fh").value
        s     = self.get_parameter("fs").value
        l     = self.get_parameter("fl").value
        tol_h = self.get_parameter("th").value
        tol_s = self.get_parameter("ts").value
        tol_l = self.get_parameter("tl").value

        # Order matches bgr_to_hls output: [H, L, S]
        self.hls_lower = np.array([h - tol_h, l - tol_l, s - tol_s], dtype=np.float32)
        self.hls_upper = np.array([h + tol_h, l + tol_l, s + tol_s], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ImageListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()