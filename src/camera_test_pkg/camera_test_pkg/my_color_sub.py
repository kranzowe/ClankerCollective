import rclpy
from rclpy.node import Node
import cv2
import numpy as np
# import time
# import serial

from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImageListener(Node):

    def __init__(self):
        super().__init__('image_listener')

        self.sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.listener_cb,
            10
        )

        self.centroid_pub = self.create_publisher(Point, '/blob_centroid', 10)

        self.sub  # prevent unused variable warning
        self.br = CvBridge()  # convert between ROS and OpenCV images

    def listener_cb(self, data):

        self.get_logger().info('Receiving video frame')

        frame = self.br.imgmsg_to_cv2(data, 'bgr8')

        # blur to reduce noise
        blur_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # convert to HSV
        hsv_frame = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

        # HSV bounds
        lower_bound = np.array([40, 80, 80])
        upper_bound = np.array([80, 255, 255])

        '''
        ----------------HSV explanation------------------------
        Hue = color (0-179 in OpenCV)
        Saturation = color intensity (0-255)
        Value = brightness (0-255)
        '''

        # threshold
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        # morphology cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # find contours
        contours, hierarchy = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) > 0:

            cnt = max(contours, key=cv2.contourArea)

            # centroid calculation
            M = cv2.moments(cnt)

            if M["m00"] != 0:

                x_pos = int(M["m10"] / M["m00"])
                y_pos = int(M["m01"] / M["m00"])
                area = M["m00"]

                self.get_logger().info(f"Centroid -> x: {x_pos}  y: {y_pos}  area: {area}")

                # publish data
                msg = Point()
                msg.x = float(x_pos)
                msg.y = float(y_pos)
                msg.z = float(area)

                self.centroid_pub.publish(msg)

                
                # draw centroid
                cv2.circle(frame, (x_pos, y_pos), 7, (0, 255, 0), -1)

        
       
        cv2.imshow("Intel RealSense Camera", frame)
        cv2.waitKey(1)


def main(args=None):

    rclpy.init(args=args)

    node = ImageListener()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()