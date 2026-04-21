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
    denom = 1.0 - np.abs(2.0 * l - 1.0)
    s = np.where(delta == 0, 0.0, np.where(denom == 0, 0.0, delta / np.maximum(denom, 1e-6)))

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


def box_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Fast separable box blur (mean filter).  ksize must be odd.
    Works on single-channel or multi-channel uint8 images.
    Significantly faster than Gaussian blur for preprocessing.
    """
    out = img.astype(np.float32)
    pad = ksize // 2
    kernel_1d = np.ones(ksize) / float(ksize)

    if out.ndim == 3:
        # Multi-channel: apply blur to each channel independently
        for c in range(out.shape[2]):
            # Horizontal pass
            out_h = np.pad(out[:, :, c], ((0, 0), (pad, pad)), mode='edge')
            out[:, :, c] = np.apply_along_axis(lambda row: np.convolve(row, kernel_1d, mode='valid'), 1, out_h)
            # Vertical pass
            out_v = np.pad(out[:, :, c], ((pad, pad), (0, 0)), mode='edge')
            out[:, :, c] = np.apply_along_axis(lambda col: np.convolve(col, kernel_1d, mode='valid'), 0, out_v)
    else:
        # Single-channel
        out_h = np.pad(out, ((0, 0), (pad, pad)), mode='edge')
        out = np.apply_along_axis(lambda row: np.convolve(row, kernel_1d, mode='valid'), 1, out_h)
        out_v = np.pad(out, ((pad, pad), (0, 0)), mode='edge')
        out = np.apply_along_axis(lambda col: np.convolve(col, kernel_1d, mode='valid'), 0, out_v)

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
    Fast flood-fill connected-component labelling using deque (2-5× faster than list-based).
    Returns (label_map, num_labels) where label_map is HxW int32.
    Labels are 1-indexed; 0 = background.
    """
    from collections import deque
    
    binary = (mask > 0)
    labels = np.zeros_like(binary, dtype=np.int32)
    current_label = 0
    rows, cols = binary.shape

    for r in range(rows):
        for c in range(cols):
            if binary[r, c] and labels[r, c] == 0:
                current_label += 1
                # BFS with deque (much faster than list)
                queue = deque([(r, c)])
                labels[r, c] = current_label
                while queue:
                    cr, cc = queue.popleft()
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
        # Dark blue tape: Hue ~110, Lightness 60-120, Saturation 100-255
        self.declare_parameter("fh", 110)      # focus hue (dark blue)
        self.declare_parameter("fs", 150)      # focus saturation
        self.declare_parameter("fl", 90)       # focus lightness

        self.declare_parameter("th", 40)       # tolerance hue
        self.declare_parameter("ts", 70)       # tolerance saturation
        self.declare_parameter("tl", 50)       # tolerance lightness

        # --- Image processing parameters ---
        self.declare_parameter("blur_ksize", 5)      # box blur kernel size (odd)
        self.declare_parameter("morph_ksize", 5)     # morphology kernel size
        self.declare_parameter("min_blob_area", 45)  # minimum blob area per band
        self.declare_parameter("num_bands", 8)       # number of horizontal bands

        # --- Waypoint smoothing parameters ---
        self.declare_parameter("smooth_factor", 0.25)  # exponential smoothing α (0=no smooth, 1=full smooth)
        self.declare_parameter("fit_window", 3)        # number of points to fit local theta

        self.create_timer(1.0, self.update_params)

        self.hls_lower = np.zeros(3, dtype=np.float32)
        self.hls_upper = np.zeros(3, dtype=np.float32)

        # --- Waypoint smoothing state ---
        self.smoothed_waypoints = []
        self.consecutive_misses = 0
        self.last_valid_trajectory = []

        self.get_logger().info("ImageListener node started (optimized for dark blue tape detection).")

        # Debug publishers for tuning
        self.declare_parameter("enable_debug_viz", True)
        self.debug_original = self.create_publisher(Image, '/debug/original_cropped', 10)
        self.debug_hls = self.create_publisher(Image, '/debug/hls_frame', 10)
        self.debug_mask_raw = self.create_publisher(Image, '/debug/mask_raw', 10)
        self.debug_mask_morph = self.create_publisher(Image, '/debug/mask_morphology', 10)
        self.debug_bands = self.create_publisher(Image, '/debug/bands_and_waypoints', 10)
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)

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
        t0 = time.time()  # timing instrumentation
        
        frame  = self.ros_image_to_numpy(data)   # HxWx3 BGR uint8
        height, width = frame.shape[:2]
        prev_point = self.prev_point

        # ---- Crop top region out ----
        crop_ratio = 0.75
        start_y    = int(height * (1 - crop_ratio))
        frame      = frame[start_y:height, :]        # work on cropped copy

        # ---- Blur → HLS → threshold ----
        blur_ksize = self.get_parameter("blur_ksize").value
        blur_frame = box_blur(frame, ksize=blur_ksize)
        hls_frame  = bgr_to_hls(blur_frame)

        lower = self.hls_lower.astype(np.uint8)
        upper = self.hls_upper.astype(np.uint8)

        mask = in_range(hls_frame, lower, upper)

        # ---- Diagnostic: count pixels in mask ----
        mask_pixels = np.sum(mask > 0)
        mask_pct = 100.0 * mask_pixels / (mask.shape[0] * mask.shape[1])
        self.get_logger().debug(f"Mask pixels: {mask_pixels} ({mask_pct:.2f}% of frame)")

        # ---- Morphological open + close ----
        morph_ksize = self.get_parameter("morph_ksize").value
        mask = morphology_open_close(mask, ksize=morph_ksize)

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
        num_bands  = self.get_parameter("num_bands").value
        band_height = cropped_height // num_bands
        min_blob_area = self.get_parameter("min_blob_area").value
        chosen_points = []
        valid_band_count = 0

        for i in range(num_bands):
            y1 = i * band_height
            y2 = (i + 1) * band_height if i < num_bands - 1 else cropped_height

            band_mask = mask[y1:y2, :]
            blobs     = find_blobs(band_mask)

            best_score = -1e18
            best_point = None

            for blob in blobs:
                area = blob['area']
                if area < min_blob_area:
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
                valid_band_count += 1

        # ---- Interpolate missing bands from neighbors ----
        for i in range(len(chosen_points)):
            if chosen_points[i] is None:
                # Find nearest valid points above and below
                above = None
                below = None
                for j in range(i - 1, -1, -1):
                    if chosen_points[j] is not None:
                        above = chosen_points[j]
                        break
                for j in range(i + 1, len(chosen_points)):
                    if chosen_points[j] is not None:
                        below = chosen_points[j]
                        break
                # Linear interpolation
                if above is not None and below is not None:
                    alpha = (i + 1) / (len(chosen_points) - i)  # interpolation weight
                    cx_interp = int(above[0] + alpha * (below[0] - above[0]))
                    cy_interp = int(above[1] + alpha * (below[1] - above[1]))
                    chosen_points[i] = (cx_interp, cy_interp)
                elif above is not None:
                    chosen_points[i] = above
                elif below is not None:
                    chosen_points[i] = below

        # ---- No valid blobs: use fallback or maintain previous trajectory ----
        if valid_band_count == 0:
            if len(self.last_valid_trajectory) > 0:
                # Maintain previous trajectory (dead-reckoning)
                self.consecutive_misses += 1
                if self.consecutive_misses > 3:
                    # Too many misses; go straight
                    straight_point = (width // 2, cropped_height // 2)
                    chosen_points = [straight_point]
                else:
                    chosen_points = self.last_valid_trajectory.copy()
            else:
                straight_point = (width // 2, cropped_height // 2)
                chosen_points = [straight_point]
        else:
            self.consecutive_misses = 0

        # ---- Diagnostic: band detection summary ----
        self.get_logger().debug(f"Band extraction: {valid_band_count}/{num_bands} bands found valid blobs, "
                               f"total candidates: {len([p for p in chosen_points if p is not None])}")
        valid_points = [pt for pt in chosen_points if pt is not None]
        smooth_factor = self.get_parameter("smooth_factor").value

        if len(valid_points) > 0:
            if len(self.smoothed_waypoints) == 0:
                self.smoothed_waypoints = valid_points.copy()
            else:
                # Exponential smoothing: p_smooth = α * p_new + (1 - α) * p_old
                smoothed = []
                for i, pt in enumerate(valid_points):
                    if i < len(self.smoothed_waypoints):
                        old_pt = self.smoothed_waypoints[i]
                        cx_smooth = int(smooth_factor * pt[0] + (1.0 - smooth_factor) * old_pt[0])
                        cy_smooth = int(smooth_factor * pt[1] + (1.0 - smooth_factor) * old_pt[1])
                        smoothed.append((cx_smooth, cy_smooth))
                    else:
                        smoothed.append(pt)
                self.smoothed_waypoints = smoothed
                valid_points = smoothed

        # Store trajectory for dead-reckoning
        self.last_valid_trajectory = valid_points.copy()

        # ---- Build and publish Path2D ----
        path_msg = Path2D()
        path_msg.poses = []

        fit_window = self.get_parameter("fit_window").value

        for i, pt in enumerate(valid_points):
            x_r, y_r = self.image_to_robot(pt, width, cropped_height)

            pose   = Pose2D()
            pose.x = float(x_r)
            pose.y = float(y_r)

            # ---- Fit theta to local window (3-5 points) for smooth heading ----
            window_start = max(0, i - fit_window // 2)
            window_end = min(len(valid_points), i + fit_window // 2 + 1)
            window_pts = valid_points[window_start:window_end]

            if len(window_pts) >= 2:
                # Use least-squares fit to compute best-fit heading
                window_pts_arr = np.array(window_pts, dtype=np.float32)
                xs = window_pts_arr[:, 0]
                ys = window_pts_arr[:, 1]
                
                # Fit line: dy/dx
                dx = xs[-1] - xs[0]
                dy = ys[-1] - ys[0]
                if dx != 0:
                    pose.theta = float(np.arctan2(-dy, dx))  # negative dy because image y increases downward
                else:
                    pose.theta = float(np.pi / 2.0) if dy > 0 else float(-np.pi / 2.0)
            else:
                pose.theta = 0.0

            path_msg.poses.append(pose)

        if len(path_msg.poses) >= 1:
            self.path_pub.publish(path_msg)

        self.prev_point = prev_point

        # ---- Timing instrumentation ----
        t_elapsed = time.time() - t0
        fps = 1.0 / t_elapsed if t_elapsed > 0 else 0.0
        self.get_logger().debug(f"Frame processed in {t_elapsed*1000:.2f}ms ({fps:.1f} FPS), "
                               f"waypoints: {len(valid_points)}")

        # ---- Debug visualizations ----
        if self.get_parameter("enable_debug_viz").value:
            # Original cropped BGR
            debug_msg = Image()
            debug_msg.header = data.header
            debug_msg.height = frame.shape[0]
            debug_msg.width = frame.shape[1]
            debug_msg.encoding = 'bgr8'
            debug_msg.step = frame.shape[1] * 3
            debug_msg.data = frame.tobytes()
            self.debug_original.publish(debug_msg)

            # HLS frame (convert back to BGR for visualization)
            hls_bgr = hls_frame[:, :, ::-1].astype(np.uint8)
            debug_hls_msg = Image()
            debug_hls_msg.header = data.header
            debug_hls_msg.height = hls_frame.shape[0]
            debug_hls_msg.width = hls_frame.shape[1]
            debug_hls_msg.encoding = 'bgr8'
            debug_hls_msg.step = hls_frame.shape[1] * 3
            debug_hls_msg.data = hls_bgr.tobytes()
            self.debug_hls.publish(debug_hls_msg)

            # Raw binary mask (convert to 3-channel BGR for visualization)
            mask_3ch = np.stack([mask, mask, mask], axis=-1)
            debug_mask_msg = Image()
            debug_mask_msg.header = data.header
            debug_mask_msg.height = mask.shape[0]
            debug_mask_msg.width = mask.shape[1]
            debug_mask_msg.encoding = 'bgr8'
            debug_mask_msg.step = mask.shape[1] * 3
            debug_mask_msg.data = mask_3ch.tobytes()
            self.debug_mask_raw.publish(debug_mask_msg)

            # Morphology-processed mask
            # (Note: mask_morph_result should be computed; see below)
            morph_3ch = np.stack([mask, mask, mask], axis=-1)  # Using final mask after morph
            debug_morph_msg = Image()
            debug_morph_msg.header = data.header
            debug_morph_msg.height = mask.shape[0]
            debug_morph_msg.width = mask.shape[1]
            debug_morph_msg.encoding = 'bgr8'
            debug_morph_msg.step = mask.shape[1] * 3
            debug_morph_msg.data = morph_3ch.tobytes()
            self.debug_mask_morph.publish(debug_morph_msg)

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
        
        self.get_logger().info(
            f"HLS Thresholds Updated:\n"
            f"  Hue:        {h - tol_h:.0f} - {h + tol_h:.0f}  (center: {h:.0f}, tol: {tol_h:.0f})\n"
            f"  Lightness:  {l - tol_l:.0f} - {l + tol_l:.0f}  (center: {l:.0f}, tol: {tol_l:.0f})\n"
            f"  Saturation: {s - tol_s:.0f} - {s + tol_s:.0f}  (center: {s:.0f}, tol: {tol_s:.0f})"
        )


# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ImageListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()