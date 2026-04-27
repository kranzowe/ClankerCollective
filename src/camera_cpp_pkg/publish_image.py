import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(
            Image,
            '/camera/camera/color/image_raw',
            10
        )

        self.bridge = CvBridge()

        # ⚠️ 改成你的圖片路徑
        img = cv2.imread('frame_20260424_143828_0.jpg')

        if img is None:
            self.get_logger().error("Image not found!")
            return

        msg = self.bridge.cv2_to_imgmsg(img, 'bgr8')

        # 發送幾次（避免 subscriber 還沒連上）
        for _ in range(5):
            self.publisher_.publish(msg)

        self.get_logger().info("Image published!")


def main():
    rclpy.init()
    node = ImagePublisher()
    rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()