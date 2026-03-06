import rclpy
from rclpy.node import Node
import cv2
import numpy as np
#import time 
#import serial

from geometry_msgs.msg import Point     #publishing this
from sensor_msgs.msg import Image       #subscribing this
from cv_bridge import CvBridge


class ImageListener(Node):
    def __init__(self):        super().__init__('image_listener')
        self.sub = self.create_subscription(Image,'/camera/camera/color/image_raw',self.listener_cb, 10)   #sub to frames that are being transmitted, need to f
        self.centroid_pub = self.create_publisher(Point, '/blob_centroid', 10)
	self.sub            #prevent unused variable warning
        self.br = CvBridge()    #covert between ROS and OpenCV images



    def listener_cb(self, data):
        self.get_logger().info('Receiving video frame')
        frame = self.br.imgmsg_to_cv2(data, 'bgr8')  #channels are flipped, need bgr8 to get channels correct
        
        #blur to reduce noise on image with a kernel
        blur_frame = blurred = cv2.GaussianBlur(frame, (5, 5), 0)   
        
        #convert to HSV, help single out the largest contour
        hsv_frame = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

        #this is the upper and lower bound of colors that we are able to change
        lower_bound = np.array([50, 180, 150])
        upper_bound = np.array([70, 255, 255])  

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

        #this block returns contours: which is a nested list of x,y coords for each contour, and hierarchy which relays info about their relationships.
        contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  #image, mode, method (image,The contour retrieval mode determines the hierarchy of the retrieved contours, The contour approximation method This determines how the contour points are stored. )

        #iterate over all contours, key being the comparison method. Ideally, the largest will be the green paper held in front of it
        if len(contours) > 0: #checking if there are any contours to begin with
            cnt = max( contours, key = cv2.contourArea)  
        
        #we are going to calculate the CENTROID of the paper, this will be helpful for directional steering of the car, knowing if we need to go left or right, want to make the centroid in the center 
        M = cv2.moments(cnt)  #The result is a dictionary M containing keys like "m00", "m10", "m01", and others. m00 represents the 0th order moment which is the area of the contour, m10 and m01 are first order moments which are the x and the y positions.
        if M["m00"] != 0:   #if the largest contoured area is zero then we skip this frame
            x_pos = int(M["m10"] / M["m00"])
            y_pos = int(M["m01"] / M["m00"])       #centroid calculation (calc 2 refresher)
            area = M["m00"]

            #Data to be published
            msg = Point()
            msg.x = float(x_pos)
            msg.y = float(y_pos)
            msg.z = float(area)

            #put a circle on the centroid (testing, can remove when we need to)
            cv2.circle(frame, (x_pos, y_pos), 7, (0,255,0), -1)


        print(x_pos,y_pos) #for testing
        # cv2.imshow("Intel RealSense Camera", frame) #these go last (optional to show the frame)
        # cv2.waitKey(1)  #DO NOT REMOVE


def main(args = None):
    rclpy.init(args = args)
    node = ImageListener()           #making the listener node
    rclpy.spin(node)            #spin until it is told to stop
    node.destroy_node()
    rclpy.shutdown()
