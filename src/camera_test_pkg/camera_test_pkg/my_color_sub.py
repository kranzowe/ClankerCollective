import rclpy
from rclpy.node import Node
import cv2
import numpy as np


from vision_msgs.msg import BoundingBox2D   #publishing this
#from geometry_msgs.msg import Pose2D    #publishing this
from sensor_msgs.msg import Image       #subscribing this
from cv_bridge import CvBridge

class ImageListener(Node):
    def __init__(self):
        super().__init__('image_listener')
        self.sub = self.create_subscription(Image,'/camera/camera/color/image_raw',self.listener_cb, 10)   #sub to frames that are being transmitted, need to f
        self.rectangle_pub = self.create_publisher(BoundingBox2D, '/blob_rectangle', 10)

        #for vinny if he NEEDS it in a different format, if we do it this way I think we will need to publish to 2 topics.
        #self.pose_pub = self.create_publisher(Pose2D, '/blob_center') 

        self.sub            #prevent unused variable warning
        self.br = CvBridge()    #covert between ROS and OpenCV images


    def listener_cb(self, data):
        self.get_logger().info('Receiving video frame', throttle_duration_sec=2.0)
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

            x, y, w, h = cv2.boundingRect(cnt)  # (x,y) is the top left of the rectangle, w is the width and h is the height, assuming the shape is a rectangle
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)   #for testing, we can see the bounding rectangle on the camera frame.
            

        # #we are going to calculate the CENTROID of the paper, this will be helpful for directional steering of the car, knowing if we need to go left or right, want to make the centroid in the center 
        M = cv2.moments(cnt)  #The result is a dictionary M containing keys like "m00", "m10", "m01", and others. m00 represents the 0th order moment which is the area of the contour, m10 and m01 are first order moments which are the x and the y positions.
        if M["m00"] != 0:   #if the largest contoured area is zero then we skip this frame
            x_pos = int(M["m10"] / M["m00"])
            y_pos = int(M["m01"] / M["m00"])       #centroid calculation (calc 2 refresher)
            cv2.circle(frame, (x_pos, y_pos), 7, (0,255,0), -1)#put a circle on the centroid (testing, can remove when we need to)   
           
            area = M["m00"] #need this incase area is too small, wont publish data if blob area is tiny.

            #---------------Data to be published-----------------

            #Box dimensions 
            rectangle_msg = BoundingBox2D()

            rectangle_msg.center.x = float(x_pos)  #center of the box
            rectangle_msg.center.y = float(y_pos)
            rectangle_msg.center.theta = 0.0

            rectangle_msg.size_x = float(w) #width  
            rectangle_msg.size_y = float(h) #height
            
            if area > 200:  #avoid small irrevelvent blobs
                self.rectangle_pub.publish(rectangle_msg)
            

            '''
            what a node that is subscribed to this node will see
            geometery_msgs/Pose2D center
            center:
            x: float64 
            y: float64 
            theta: float64  (rotation)
            size_x: float64 (width)
            size_y: float64 (height)

            '''
            

        # cv2.imshow("Intel RealSense Camera", frame) 
        # cv2.waitKey(1) 
        
       

def main(args = None):
    rclpy.init(args = args)
    node = ImageListener()           #making the listener node
    rclpy.spin(node)            #spin until it is told to stop
    node.destroy_node()
    rclpy.shutdown()


