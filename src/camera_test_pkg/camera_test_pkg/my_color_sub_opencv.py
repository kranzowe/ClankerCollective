import rclpy
from rclpy.node import Node
import cv2
import numpy as np


from nav_2d_msgs.msg import Path2D   #publishing this
from geometry_msgs.msg import Pose2D    #publishing this
from sensor_msgs.msg import Image       #subscribing this
from cv_bridge import CvBridge

class ImageListener(Node):
    def __init__(self):
        super().__init__('image_listener')
        self.sub = self.create_subscription(Image,'/camera/camera/color/image_raw',self.listener_cb, 10)   #sub to frames that are being transmitted, need to f
        self.path_pub = self.create_publisher(Path2D, '/line_path', 10)

        #for vinny if he NEEDS it in a different format, if we do it this way I think we will need to publish to 2 topics.
        #self.pose_pub = self.create_publisher(Pose2D, '/blob_center') 

        self.sub            #prevent unused variable warning
        self.br = CvBridge()    #covert between ROS and OpenCV images

    def image_to_robot(self, pt, width, height):
        px, py = pt

        # Normalize
        x_norm = (px - width / 2) / (width / 2)   # left/right
        y_norm = (height - py) / height           # forward

        # Tune these if needed
        forward_scale = 1.0
        lateral_scale = 1.0

        x_robot = y_norm * forward_scale
        y_robot = x_norm * lateral_scale

        return x_robot, y_robot

    def listener_cb(self, data):

        #--------------------------------------OpenCV Setup--------------------------------------------------

        #self.get_logger().info('Receiving video frame', throttle_duration_sec=2.0)
        frame = self.br.imgmsg_to_cv2(data, 'bgr8')  #channels are flipped, need bgr8 to get channels correct
        
        height, width = frame.shape[:2] #define shape of the frame

        #blur to reduce noise on image with a kernel
        blur_frame = blurred = cv2.GaussianBlur(frame, (5, 5), 0)   
        
        #convert to HSV, help single out the largest contour
        hsv_frame = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

        #this is the upper and lower bound of colors that we are able to change
        
        #blue range
        lower_bound = np.array([100, 100, 50])
        upper_bound = np.array([140, 255, 255])  

        
        '''
        ----------------HSV explaination------------------------
        want to account for shadows in our view, Hue is color 
        (0-179 for openCV, green is around 60), Saturation = color 
        intensitiy, how PURE the color is (assuming bright green 
        paper 0-255), Value = brightness (0-255 black, dim, 
        bright, assuming dim - bright)
        '''

        #color thresholding, keep pixels in this range of the HSV spectrum. in range = white, out of range = black
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        #create the kernel and remove noise and fill in the blob to make centroid calculation more accurate.
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   #open = remove noise   
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  #close = fill in the blob detected

        # ---- Divide into bands ----
        num_bands = 6
        band_height = height // num_bands

        chosen_points = []

        # ---- Process each band ----
        for i in range(num_bands):
            y1 = i * band_height
            y2 = (i + 1) * band_height if i < num_bands - 1 else height

            band_mask = mask[y1:y2, :]

            contours, _ = cv2.findContours(
                band_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            best_area = 0
            best_point = None

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 50:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)

                # Convert to global coords
                x_global = x
                y_global = y1 + y

                cx = x_global + w // 2
                cy = y_global + h // 2

                # Draw all blobs (blue)
                cv2.rectangle(frame,
                              (x_global, y_global),
                              (x_global + w, y_global + h),
                              (255, 0, 0), 1)

                # Draw all centers (red)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

                if area > best_area:
                    best_area = area
                    best_point = (cx, cy)

            chosen_points.append(best_point)

            # Highlight chosen point (yellow)
            if best_point is not None:
                cv2.circle(frame, best_point, 6, (0, 255, 255), -1)

        # ---- Draw path ----
        prev = None
        for pt in chosen_points:
            if pt is not None:
                if prev is not None:
                    cv2.line(frame, prev, pt, (0, 255, 255), 2)
                prev = pt

        # ---- Draw band lines ----
        for i in range(1, num_bands):
            y = i * band_height
            cv2.line(frame, (0, y), (width, y), (0, 255, 0), 1)

        # ---- Build Path2D ----
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

                dx = x_next - x_r
                dy = y_next - y_r

                pose.theta = float(np.arctan2(dy, dx))
            else:
                pose.theta = 0.0

            path_msg.poses.append(pose)

        if len(path_msg.poses) > 1:
            self.path_pub.publish(path_msg)
            
#--------------------testing display------------------------
        cv2.imshow("Intel RealSense Camera", frame) 
        cv2.imshow("mask", mask)
        cv2.waitKey(1) 
        
       

def main(args = None):
    rclpy.init(args = args)
    node = ImageListener()           #making the listener node
    rclpy.spin(node)            #spin until it is told to stop
    node.destroy_node()
    rclpy.shutdown()

