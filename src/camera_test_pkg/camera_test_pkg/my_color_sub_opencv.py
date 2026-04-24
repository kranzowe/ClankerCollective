import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import time

from vision_msgs.msg import BoundingBox2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

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

        self.prev_point = None   #define in listener_cb hepler

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

        self.create_timer(1.0, self.update_params)

        self.bgr_lower = np.zeros(3)
        self.bgr_upper = np.zeros(3)

    # ------------------ Image to Robot Relation ------------------
    def image_to_robot(self, pt, width, height):
        px, py = pt

        x_norm = (px - width / 2) / (width / 2)
        y_norm = ((height - py) / height) ** 1.5  #points further away are exponentially farther away, just sqaure it and it should work

        forward_scale = 1.0
        lateral_scale = 1.4

        x_robot = y_norm * forward_scale
        y_robot = x_norm * lateral_scale

        return x_robot, y_robot

    # ------------------ Main Callback function ------------------
    def listener_cb(self, data):

        frame = self.br.imgmsg_to_cv2(data, 'bgr8')

        height, width = frame.shape[:2] #height and width variables

        prev_point = self.prev_point

        # ---- CROP TOP REGION OUT ----
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

        

        # ---- ORIGINAL: largest contour code, unused in this model (CAN IGNORE) ----
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- see old code for explainations, removed them to make this easier to read ---
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

        # ---- NEW: Band-based path extraction ----

        num_bands = 8   #6 horizontal bands, plans out the path for the robot, max of 6 poitns
        band_height = height // num_bands
        chosen_points = []

        for i in range(num_bands):
            y1 = i * band_height
            y2 = (i + 1) * band_height if i < num_bands - 1 else height

            band_mask = mask[y1:y2, :]

            contours, _ = cv2.findContours(band_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #finding contours within each individual band

            best_area = 0   #temp vars
            best_point = None

            for cnt in contours:    #go thru all blobs in this band
                area = cv2.contourArea(cnt)
                if area < 45 :       #same as old code, filter out the noise, if too small it skips
                    continue

                x, y, w, h = cv2.boundingRect(cnt) #bounding box, top left & right, along with width and height

                x_global = x    #convert to global, each band is a slice of a bigger picture, so we are adding all of the y's on top of each other
                y_global = y1 + y

                cx = x_global + w // 2  #getting central x&y points
                cy = y_global + h // 2

                # for the debugging  (draw it on the OG frame)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

                #-----------------scoring phase, favoring right turns------------------------
                # if area > best_area:    #replace smaller area with bigger area within band
                #     best_area = area
                #     best_point = (cx, cy)

                # normalize x position (0 = left, 1 = right ONLY FOR SCORING tho)
                #-----------------scoring phase, favoring right turns------------------------

                # normalize x position (-1 = left, 0 = center, +1 = right. ONLY FOR ROBOT)
                x_norm = (cx - width / 2) / (width / 2)

                # convert to 0 → 1 ONLY FOR SCORING tho
                x_bias = (x_norm + 1) / 2

                # continuity: prefer points close to previous band
                if prev_point is not None:
                    dist = abs(cx - prev_point[0])
                    continuity_penalty = dist * 2.0
                else:
                    continuity_penalty = 0

                # scoring weights (tunable during our testing)
                area_weight = 1.0
                right_weight = 2000   # what gets increased to favor right turns more based on scoring

                score = (area * area_weight) + (x_bias * right_weight) - continuity_penalty

                # strong right-turn bias (helps choose the 90-degree turn over the trap)
                if x_bias > 0.7:
                    score += 1500

                if score > best_area:
                    best_area = score
                    best_point = (cx, cy)

            chosen_points.append(best_point)    #make a list of our path, what we are gonna send out to controls

            if best_point is not None:  #draw on OG frame, bigger than debug points
                cv2.circle(frame, best_point, 6, (0, 255, 255), -1)

                prev_point = best_point   # <-- keep your continuity system intact


        ### SOME VERTICAL POINTS TO THE RIGHT NEEDED
        num_vert_bands = 3   #6 horizontal bands, plans out the path for the robot, max of 6 poitns
        vert_band_width = (width/2.0) // num_vert_bands
        vert_chosen_points = []
        prev_point = None

        for i in range(num_vert_bands):
            x1 =int(width/2.0) + i * vert_band_width
            x2 = int(width/2.0) + (i + 1) * vert_band_width if i < num_vert_bands - 1 else width

            vert_band_mask = mask[:, int(x1):int(x2)]

            contours, _ = cv2.findContours(vert_band_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #finding contours within each individual band

            best_area = 0   #temp vars
            best_point = None

            for cnt in contours:    #go thru all blobs in this band
                area = cv2.contourArea(cnt)
                if area < 45 :       #same as old code, filter out the noise, if too small it skips
                    continue

                x, y, w, h = cv2.boundingRect(cnt) #bounding box, top left & right, along with width and height

                x_global = x1 + x    #convert to global, each band is a slice of a bigger picture, so we are adding all of the y's on top of each other
                y_global = y

                cx = x_global + w // 2  #getting central x&y points
                cy = y_global + h // 2

                # for the debugging  (draw it on the OG frame)
                cv2.circle(frame, (cx, cy), 3, (255, 0, 255), -1)

                # normalize x position (-1 = left, 0 = center, +1 = right. ONLY FOR ROBOT)
                y_norm = (cy - height / 2) / (height / 2)

                # convert to 0 → 1 ONLY FOR SCORING tho
                y_bias = (y_norm + 1) / 2

                # continuity: prefer points close to previous band
                if prev_point is not None:
                    dist = abs(cy - prev_point[1])
                    continuity_penalty = dist * 2.0
                else:
                    continuity_penalty = 0

                # scoring weights (tunable during our testing)
                area_weight = 1.0
                right_weight = 2000   # what gets increased to favor right turns more based on scoring

                score = (area * area_weight) + (y_bias * right_weight) - continuity_penalty

                # strong right-turn bias (helps choose the 90-degree turn over the trap)
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
        prev = None #temp
        for pt in chosen_points:    #simple loop to connect the lines, this we may comment out if we need faster speed, although it shouldnt affect speed since it would only look ~6 time MAX
            if pt is not None:
                if prev is not None:
                    cv2.line(frame, prev, pt, (0, 255, 255), 2)
                prev = pt

        # ---- Build Path2D, for pubishing ----
        path_msg = Path2D()
        path_msg.poses = []

        valid_points = [pt for pt in chosen_points if pt is not None] #condensed loop to get points in valid

        for i in range(len(valid_points)):
            pt = valid_points[i]

            x_r, y_r = self.image_to_robot(pt, width, height) #our image points get converted to actual points the roboto can use becuase of our helper function SEE ABOVE

            pose = Pose2D() #pose publish data, this is creating a waypoint tht goes to the path, each of the ~6 points are an x, y, and a theta (to the next point)
            pose.x = float(x_r) #forward position in robot frame, big x means far away
            pose.y = float(y_r) #lateral offset, y>0 is left, y<0 is right

            if i < len(valid_points) - 1: #cannot compute direction for the LAST point because there is no “next point” to compare to
                next_pt = valid_points[i + 1] #go to next point on our path we made

                x_next, y_next = self.image_to_robot(next_pt, width, height) #next point converted to robot coords

                dx = x_r
                dy = y_r    #relative to inital frame

                pose.theta = float(np.arctan2(dy, dx))  #finally compute and append theta
            else:
                pose.theta = 0.0
                                            #pose theta is direction of path at that segment
            path_msg.poses.append(pose)
        
        ## always add vert points
        for i in range(num_vert_bands):
            try:
                pnt = vert_chosen_points[i]
                x_r, y_r = self.image_to_robot(pnt, width, height) #our image points get converted to actual points the roboto can use becuase of our helper function SEE ABOVE
                theta = -90.0 
            except:
                x_r, y_r = -99.9, -99.9
                theta = np.inf
            
            pose = Pose2D() #pose publish data, this is creating a waypoint tht goes to the path, each of the ~6 points are an x, y, and a theta (to the next point)
            pose.x = float(x_r) #forward position in robot frame, big x means far away
            pose.y = float(y_r)
            pose.theta = theta
            path_msg.poses.append(pose)

        if len(path_msg.poses) >= 1:
            self.path_pub.publish(path_msg)

        self.prev_point = prev_point

        cv2.imshow("Processed Image", frame)
        cv2.imshow("Mask", mask)
        cv2.waitKey(1)

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


def main(args=None):
    rclpy.init(args=args)
    node = ImageListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


#entry point

if __name__ == '__main__':
    main()