import rclpy
from rclpy.node import Node
import numpy as np
import time

#from vision_msgs.msg import BoundingBox2D
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
    Fast separable box blur using cumsum (10-20x faster than apply_along_axis).
    ksize must be odd.  Works on single-channel or multi-channel uint8 images.
    """
    try:
        from scipy.ndimage import uniform_filter
        return uniform_filter(img, size=(ksize, ksize, 1) if img.ndim == 3 else (ksize, ksize)).astype(np.uint8)
    except ImportError:
        # Fallback: cumsum-based separable blur (much faster than apply_along_axis)
        out = img.astype(np.float32)
        pad = ksize // 2
        
        if out.ndim == 3:
            for c in range(out.shape[2]):
                # Horizontal pass via cumsum
                padded = np.pad(out[:, :, c], ((0, 0), (pad, pad)), mode='edge')
                cumsum = np.cumsum(padded, axis=1)
                out[:, :, c] = (cumsum[:, ksize:] - cumsum[:, :-ksize]) / ksize
        else:
            padded = np.pad(out, ((0, 0), (pad, pad)), mode='edge')
            cumsum = np.cumsum(padded, axis=1)
            out = (cumsum[:, ksize:] - cumsum[:, :-ksize]) / ksize
        
        return np.clip(out, 0, 255).astype(np.uint8)


def in_range(img: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Element-wise per-channel threshold; returns a bool mask (HxW)."""
    mask = np.all((img >= lower) & (img <= upper), axis=-1)
    return mask.astype(np.uint8) * 255


def morphology_open_close(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Fast morphology: just erode to remove noise (no full open+close).
    For Pi, full open+close is too slow. Single erode is often sufficient.
    """
    try:
        from scipy.ndimage import binary_erosion, binary_dilation
        # Simple erode (faster than open+close)
        struct = np.ones((ksize, ksize), dtype=bool)
        eroded = binary_erosion(mask > 0, structure=struct)
        return (eroded * 255).astype(np.uint8)
    except ImportError:
        # Fallback: simple numpy-based erode using min filter
        pad = ksize // 2
        padded = np.pad(mask, pad, mode='constant', constant_values=255)
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(padded, (ksize, ksize))
        result = (windows.min(axis=(-2, -1)) > 0) * 255
        return result.astype(np.uint8)


def find_blobs_fast(mask: np.ndarray, min_area: int = 45):
    """
    Ultra-fast blob detection: just compute centroid + area of entire mask.
    Skips connected-component labeling entirely (expensive on Pi).
    Returns a list with single dict or empty list.
    """
    coords = np.argwhere(mask > 0)
    if len(coords) < min_area:
        return []
    
    rows_lbl = coords[:, 0]
    cols_lbl = coords[:, 1]
    area = len(coords)
    cy = int(rows_lbl.mean())
    cx = int(cols_lbl.mean())
    y = int(rows_lbl.min())
    x = int(cols_lbl.min())
    h = int(rows_lbl.max()) - y + 1
    w = int(cols_lbl.max()) - x + 1
    return [dict(area=area, cx=cx, cy=cy, x=x, y=y, w=w, h=h)]


# ──────────────────────────────────────────────────────────────────────────────
#  ROS2 Node
# ──────────────────────────────────────────────────────────────────────────────

class ImageListener(Node):
    def __init__(self):
        super().__init__('image_listener')

        # --- subscriptions / publications ---
        self.sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.listener_cb, 10)

        #self.rectangle_pub = self.create_publisher(BoundingBox2D, '/blob_rectangle', 10)
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
        self.declare_parameter("blur_ksize", 3)      # box blur kernel size (odd) - smaller for Pi
        self.declare_parameter("morph_ksize", 3)     # morphology kernel size - smaller
        self.declare_parameter("min_blob_area", 45)  # minimum blob area
        self.declare_parameter("num_bands", 6)       # reduced from 8 for Pi speed
        self.declare_parameter("num_vert_bands", 3)  # reduced from 4 for Pi speed
        self.declare_parameter("downsample_factor", 2)  # CRITICAL: downsample 2-4x for Pi (1=none, 2=2x, 4=4x)

        # --- Scoring parameters (same as OpenCV version) ---
        self.declare_parameter("area_weight", 1.0)    # weight for blob area in scoring
        self.declare_parameter("right_weight", 2000)  # weight for right-turn bias
        self.declare_parameter("continuity_scale", 2.0) # distance penalty scale
        self.declare_parameter("skip_morph", False)   # skip morphology entirely (saves ~20ms)

        self.create_timer(1.0, self.update_params)

        self.hls_lower = np.zeros(3, dtype=np.float32)
        self.hls_upper = np.zeros(3, dtype=np.float32)

        self.get_logger().info("ImageListener node started (8 horizontal + 4 vertical bands).")

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
        y_ratio = max(0.0, (height - py) / height)
        y_norm = y_ratio ** 1.5

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

        # ---- EARLY DOWNSAMPLE (CRITICAL for Pi performance) ----
        downsample_factor = self.get_parameter("downsample_factor").value
        if downsample_factor > 1:
            frame = frame[::downsample_factor, ::downsample_factor, :]

        cropped_height, cropped_width = frame.shape[:2]

        # ---- Blur → HLS → threshold ----
        blur_ksize = self.get_parameter("blur_ksize").value
        if blur_ksize < 3:
            blur_ksize = 3
        blur_frame = box_blur(frame, ksize=blur_ksize)
        hls_frame  = bgr_to_hls(blur_frame)

        lower = self.hls_lower.astype(np.uint8)
        upper = self.hls_upper.astype(np.uint8)

        mask = in_range(hls_frame, lower, upper)

        # ---- Optional morphology (skip on Pi for speed) ----
        skip_morph = self.get_parameter("skip_morph").value
        if not skip_morph:
            morph_ksize = self.get_parameter("morph_ksize").value
            if morph_ksize < 3:
                morph_ksize = 3
            mask = morphology_open_close(mask, ksize=morph_ksize)

        # ---- HORIZONTAL BANDS: 6 bands for forward path planning ----
        num_bands  = self.get_parameter("num_bands").value
        band_height = cropped_height // num_bands
        min_blob_area = self.get_parameter("min_blob_area").value
        area_weight = self.get_parameter("area_weight").value
        right_weight = self.get_parameter("right_weight").value
        continuity_scale = self.get_parameter("continuity_scale").value
        
        chosen_points = []

        for i in range(num_bands):
            y1 = i * band_height
            y2 = (i + 1) * band_height if i < num_bands - 1 else cropped_height

            band_mask = mask[y1:y2, :]
            
            # Find blob in this band (fast: just centroid, no connected components)
            blobs = find_blobs_fast(band_mask, min_area=min_blob_area)
            
            best_score = -1
            best_point = None

            for blob in blobs:
                area = blob['area']
                if area < min_blob_area:
                    continue
                
                # Convert to global coordinates
                cx = blob['cx']
                cy = y1 + blob['cy']
                
                # Normalize x position (-1 = left, 0 = center, +1 = right)
                x_norm = (cx - cropped_width / 2) / (cropped_width / 2)
                
                # Convert to 0 → 1 for scoring
                x_bias = (x_norm + 1) / 2
                
                # Continuity penalty
                if prev_point is not None:
                    dist = abs(cx - prev_point[0])
                    continuity_penalty = dist * continuity_scale
                else:
                    continuity_penalty = 0
                
                # Compute score (matches OpenCV version)
                score = (area * area_weight) + (x_bias * right_weight) - continuity_penalty
                
                # Strong right-turn bias
                if x_bias > 0.7:
                    score += 1500
                
                if score > best_score:
                    best_score = score
                    best_point = (cx, cy)
            
            chosen_points.append(best_point)
            if best_point is not None:
                prev_point = best_point

        # ---- No blobs: go straight for one frame ----
        if not any(pt is not None for pt in chosen_points):
            straight_point = (cropped_width // 2, cropped_height // 2)
            chosen_points = [straight_point]

        # ---- VERTICAL BANDS: 3 bands in right third for right-side detection ----
        num_vert_bands = self.get_parameter("num_vert_bands").value
        vert_band_width = (cropped_width * 1.0 / 3.0) // num_vert_bands
        vert_chosen_points = []
        vert_prev_point = None

        for i in range(num_vert_bands):
            x1 = int(cropped_width * 2.0 / 3.0) + int(i * vert_band_width)
            x2 = int(cropped_width * 2.0 / 3.0) + int((i + 1) * vert_band_width) if i < num_vert_bands - 1 else cropped_width

            vert_band_mask = mask[:, x1:x2]
            
            # Find blob in this vertical band (fast)
            blobs = find_blobs_fast(vert_band_mask, min_area=min_blob_area)
            
            best_score = -1
            best_point = None

            for blob in blobs:
                area = blob['area']
                if area < min_blob_area:
                    continue
                
                # Convert to global coordinates
                cx = x1 + blob['cx']
                cy = blob['cy']
                
                # Normalize y position (-1 = top, 0 = center, +1 = bottom)
                y_norm = (cy - cropped_height / 2) / (cropped_height / 2)
                
                # Convert to 0 → 1 for scoring
                y_bias = (y_norm + 1) / 2
                
                # Continuity penalty
                if vert_prev_point is not None:
                    dist = abs(cy - vert_prev_point[1])
                    continuity_penalty = dist * continuity_scale
                else:
                    continuity_penalty = 0
                
                # Compute score
                score = (area * area_weight) + (y_bias * right_weight) - continuity_penalty
                
                # Strong bottom-bias (prefer obstacles/paths lower in image)
                if y_bias > 0.7:
                    score += 1500
                
                if score > best_score:
                    best_score = score
                    best_point = (cx, cy)
            
            vert_chosen_points.append(best_point)
            if best_point is not None:
                vert_prev_point = best_point

        # ---- Build and publish Path2D ----
        path_msg = Path2D()
        path_msg.poses = []

        # Add horizontal band waypoints
        valid_points = [pt for pt in chosen_points if pt is not None]

        for i in range(len(valid_points)):
            pt = valid_points[i]

            x_r, y_r = self.image_to_robot(pt, cropped_width, cropped_height)

            pose   = Pose2D()
            pose.x = float(x_r)
            pose.y = float(y_r)

            # Compute heading to next waypoint
            if i < len(valid_points) - 1:
                next_pt = valid_points[i + 1]
                x_next, y_next = self.image_to_robot(next_pt, cropped_width, cropped_height)
                dx = x_next - x_r
                dy = y_next - y_r
                pose.theta = float(np.arctan2(dy, dx))
            else:
                pose.theta = 0.0

            path_msg.poses.append(pose)

        # Add vertical band waypoints
        for i in range(num_vert_bands):
            try:
                pnt = vert_chosen_points[i]
                if pnt is not None:
                    x_r, y_r = self.image_to_robot(pnt, cropped_width, cropped_height)
                    theta = np.pi / 2.0  # 90 degrees for right-side waypoints
                else:
                    x_r, y_r = -99.9, -99.9
                    theta = np.inf
            except:
                x_r, y_r = -99.9, -99.9
                theta = np.inf
            
            pose = Pose2D()
            pose.x = float(x_r)
            pose.y = float(y_r)
            pose.theta = float(theta)
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