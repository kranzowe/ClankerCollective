import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import time

from vision_msgs.msg import BoundingBox2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Bool

from nav_2d_msgs.msg import Path2D
from geometry_msgs.msg import Pose2D


class ImageListener(Node):
    def __init__(self):
        super().__init__('image_listener')

        # --- subscription ---
        self.sub = self.create_subscription(Image,'/camera/camera/color/image_raw',self.listener_cb,10)

        # --- publishing ---
        self.rectangle_pub = self.create_publisher(BoundingBox2D,'/blob_rectangle',10)
        self.path_pub = self.create_publisher(Path2D,'/line_path',10)
        self.stop_pub = self.create_publisher(Bool, '/stop_sign_stop', 10)

        self.prev_point = None

        # --- CV Bridge init ---
        cv_up = False
        while not cv_up:
            try:
                self.br = CvBridge()
                cv_up = True
            except:
                time.sleep(0.2)

        # --- Parameters  (HSV point +tolerance) ---
        self.declare_parameter("fh", 120)
        self.declare_parameter("fs", 180)
        self.declare_parameter("fl", 120)

        self.declare_parameter("th", 45)
        self.declare_parameter("ts", 80)
        self.declare_parameter("tl", 100)

        # --- Stop Sign Parameters ---
        self.declare_parameter("stop_sign_detection_enabled", True)
        self.declare_parameter("stop_sign_min_area", 1000)

        self.create_timer(1.0, self.update_params)

        self.bgr_lower = np.zeros(3)
        self.bgr_upper = np.zeros(3)
        self.last_save_time = 0.0

        # --- Stop sign state ---
        self.stop_sign_detection_enabled = True
        self.stop_sign_min_area = 1000


    # ------------------ Image to Robot Relation, creating "points" for pure pursuit (unused for pid follower) ------------------
    def image_to_robot(self, pt, width, height):
        px, py = pt

        x_norm = (px - width / 2) / (width / 2)
        y_norm = ((height - py) / height) ** 1.5

        forward_scale = 1.0
        lateral_scale = 1.4

        x_robot = y_norm * forward_scale
        y_robot = x_norm * lateral_scale

        return x_robot, y_robot

    # ------------------ Red Blob Detection (Stop Sign) ------------------
    def detect_stop_sign(self, frame):
        """
        Detects red blobs in the frame using two HSV ranges (red wraps around 0/180).
        Returns True if a sufficiently large red blob is found, False otherwise.
        Also draws debug visuals on the frame in-place. 

        This will run over the normal image overlay, won't take up too much computing power, since this is on OpenCV, 
        we just return a bool if we see it or not, planning on having a cooldown where after we initially see a sign,
        we continue as normal until we dont see it then the cooldown stops then normal
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red spans two ranges in HSV: ~0-10 and ~160-180
        lower_red1 = np.array([0,   120,  70])
        upper_red1 = np.array([10,  255, 255])
        lower_red2 = np.array([160, 120,  70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.stop_sign_min_area:
                detected = True
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(frame, "STOP SIGN", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return detected

    # ------------------ Main Callback function ------------------
    def listener_cb(self, data):

        frame = self.br.imgmsg_to_cv2(data, 'bgr8')

        height, width = frame.shape[:2]

        prev_point = self.prev_point

        # ---- STOP SIGN CHECK (runs the full frame, before cropping) ----
        if self.stop_sign_detection_enabled:
            if self.detect_stop_sign(frame):
                stop_msg = Bool()
                stop_msg.data = True
                self.stop_pub.publish(stop_msg)
                self.get_logger().info("Stop sign detected :)")
            else:
                stop_msg = Bool()
                stop_msg.data = False
                self.stop_pub.publish(stop_msg)
                self.get_logger().info("No stop sign :(")
        # ---- CROP TOP REGION OUT, ignore noise at top ----

        crop_ratio = 0.75
        start_y = int(height * (1 - crop_ratio))
        frame = frame[start_y:height, :]

        # ---- OpenCV, getting just the blue, Image Processing ----
        blur_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        hls_frame = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HLS_FULL)

        lower_bound = np.array(self.bgr_lower)
        upper_bound = np.array(self.bgr_upper)

        mask = cv2.inRange(hls_frame, lower_bound, upper_bound)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # ---- origional code, for the paper follower, unused for final: largest contour code, unused in this model (CAN IGNORE) ----
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                x_pos = int(M["m10"] / M["m00"])
                y_pos = int(M["m01"] / M["m00"])

                cv2.circle(frame, (x_pos, y_pos), 7, (0, 255, 0), -1)

                area = cv2.contourArea(cnt)

                if area > 500:
                    rectangle_msg = BoundingBox2D()
                    rectangle_msg.center.position.x = float(x_pos)
                    rectangle_msg.center.position.y = float(y_pos)
                    rectangle_msg.center.theta = 0.0
                    rectangle_msg.size_x = float(w)
                    rectangle_msg.size_y = float(h)

                    self.rectangle_pub.publish(rectangle_msg)

        # ----New, line follower openCV: Band-based path extraction ----
        num_bands = 8
        band_height = height // num_bands
        chosen_points = []

        for i in range(num_bands):
            y1 = i * band_height
            y2 = (i + 1) * band_height if i < num_bands - 1 else height

            band_mask = mask[y1:y2, :]

            contours, _ = cv2.findContours(band_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            best_area = 0
            best_point = None

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 45:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)

                x_global = x
                y_global = y1 + y

                cx = x_global + w // 2
                cy = y_global + h // 2

                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

                x_norm = (cx - width / 2) / (width / 2)
                x_bias = (x_norm + 1) / 2

                if prev_point is not None:
                    dist = abs(cx - prev_point[0])
                    continuity_penalty = dist * 2.0
                else:
                    continuity_penalty = 0

                area_weight = 1.0
                right_weight = 2000

                score = (area * area_weight) + (x_bias * right_weight) - continuity_penalty

                if x_bias > 0.7:
                    score += 1500

                if score > best_area:
                    best_area = score
                    best_point = (cx, cy)

            chosen_points.append(best_point)

            if best_point is not None:
                cv2.circle(frame, best_point, 6, (0, 255, 255), -1)
                prev_point = best_point


        ### SOME VERTICAL POINTS TO THE RIGHT NEEDED, RIGHT TURN SECQUENCE :D
        num_vert_bands = 4
        vert_band_width = (width*1.0/3.0) // num_vert_bands
        vert_chosen_points = []
        prev_point = None

        for i in range(num_vert_bands):
            x1 =int(width*2.0/3.0) + i * vert_band_width
            x2 = int(width*2.0/3.0) + (i + 1) * vert_band_width if i < num_vert_bands - 1 else width

            vert_band_mask = mask[:, int(x1):int(x2)]

            contours, _ = cv2.findContours(vert_band_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            best_area = 0
            best_point = None

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 45:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)

                x_global = x1 + x
                y_global = y

                cx = x_global + w // 2
                cy = y_global + h // 2

                cv2.circle(frame, (int(cx), int(cy)), 3, (255, 0, 255), -1)

                y_norm = (cy - height / 2) / (height / 2)
                y_bias = (y_norm + 1) / 2

                if prev_point is not None:
                    dist = abs(cy - prev_point[1])
                    continuity_penalty = dist * 2.0
                else:
                    continuity_penalty = 0

                area_weight = 1.0
                right_weight = 2000

                score = (area * area_weight) + (y_bias * right_weight) - continuity_penalty

                if y_bias > 0.7:
                    score += 1500

                if score > best_area:
                    best_area = score
                    best_point = (cx, cy)

            vert_chosen_points.append(best_point)

        # ---- NO BLOBS: go straight for one frame ----
        if not any(pt is not None for pt in chosen_points):
            cropped_height = frame.shape[0]
            straight_point = (width // 2, cropped_height // 2)
            chosen_points = [straight_point]

        # ---- Draw path for robo car ----
        prev = None
        for pt in chosen_points:
            if pt is not None:
                if prev is not None:
                    cv2.line(frame, prev, pt, (0, 255, 255), 2)
                prev = pt

        # ---- Build Path2D, for publishing ----
        path_msg = Path2D()
        path_msg.poses = []

        valid_points = [pt for pt in chosen_points if pt is not None]

        for i in range(len(valid_points)):
            pt = valid_points[i]

            x_r, y_r = self.image_to_robot(pt, width, height)

            pose = Pose2D()
            pose.x = float(x_r)
            pose.y = float(y_r)

            if i < len(valid_points) - 1:
                next_pt = valid_points[i + 1]

                x_next, y_next = self.image_to_robot(next_pt, width, height)

                dx = x_r
                dy = y_r

                pose.theta = float(np.arctan2(dy, dx))
            else:
                pose.theta = 0.0

            path_msg.poses.append(pose)

        ## always add vert points
        for i in range(num_vert_bands):
            try:
                pnt = vert_chosen_points[i]
                x_r, y_r = pnt[0], pnt[1]
                theta = 90.0
            except:
                x_r, y_r = -99.9, -99.9
                theta = np.inf

            pose = Pose2D()
            pose.x = float()
            pose.y = float(y_r)
            pose.theta = theta
            path_msg.poses.append(pose)

        if len(path_msg.poses) >= 1:
            self.path_pub.publish(path_msg)

        self.prev_point = prev_point


    # ------------------ Parameter Update ------------------
    def update_params(self):

        h = self.get_parameter("fh").value
        s = self.get_parameter("fs").value
        l = self.get_parameter("fl").value

        tol_h = self.get_parameter("th").value
        tol_s = self.get_parameter("ts").value
        tol_l = self.get_parameter("tl").value

        self.bgr_lower = np.array([h - tol_h, l - tol_l, s - tol_s])
        self.bgr_upper = np.array([h + tol_h, l + tol_l, s + tol_s])

        self.stop_sign_detection_enabled = self.get_parameter("stop_sign_detection_enabled").value
        self.stop_sign_min_area          = self.get_parameter("stop_sign_min_area").value


def main(args=None):
    rclpy.init(args=args)
    node = ImageListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()