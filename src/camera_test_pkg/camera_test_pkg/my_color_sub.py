import rclpy
from rclpy.node import Node
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageListener(Node):
    def __init__(self):
        super().__init__('image_listener')
        self.sub = self.create_subscription(Image,'/camera/camera/color/image_raw',self.listener_cb, 10)   #sub to frames that are being transmitted, need to f
        self.sub            #prevent unused variable warning
        self.br = CvBridge()    #covert between ROS and OpenCV images

    # def img_read(self,msg: Image):
    #     im_cap = cv2.VideoCapture(1,cv2.CAP_V4L2)

    def listener_cb(self, data):
        self.get_logger().info('Receiving video frame') 
        frame = self.br.imgmsg_to_cv2(data, 'bgr8')     #need bgr8 becuase the channels are swapped from ros2 to opencv. 
        cv2.imshow("Intel RealSense Camera", frame) 

        cv2.waitKey(1)


def main(args = None):
    rclpy.init(args = args)
    node = ImageListener()           #making the listener node
    rclpy.spin(node)            #spin until it is told to stop
    node.destroy_node()
    rclpy.shutdown()
    