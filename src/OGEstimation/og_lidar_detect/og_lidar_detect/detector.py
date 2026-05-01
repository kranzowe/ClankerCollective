#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np

class WallDetector(Node):
    def __init__(self):
        super().__init__('wall_detector_node')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        self.get_logger().info('front wall distance detector initialized...')

    def scan_callback(self, msg):
        # 1. 找到 0 弧度 (正前方) 的索引
        # 公式: (目標角度 - 最小角度) / 角度增量
        center_idx = int((0.0 - msg.angle_min) / msg.angle_increment)
        
        # 2. define the angle range to consider (e.g., ±2 degrees)
        angle_range_deg = 2
        angle_range_rad = np.deg2rad(angle_range_deg)
        index_offset = int(angle_range_rad / msg.angle_increment)
        
        # 3. get the ranges in the front area
        front_ranges = msg.ranges[center_idx - index_offset : center_idx + index_offset]
        
        # 4. filtering out usless data (too far, too close or inf)
        valid_ranges = [r for r in front_ranges if msg.range_min < r < msg.range_max]
        
        if valid_ranges:
            # apply Median filter to reduce noise
            avg_distance = np.median(valid_ranges)
            
            # distance threshold for warning
            if avg_distance < 1.0:
                self.get_logger().warn(f'⚠️ warning! too close：{avg_distance:.2f} m')
            else:
                self.get_logger().info(f'front wall dist：{avg_distance:.2f} m')
        else:
            self.get_logger().info('detecting, currently no valid data in front...')

def main(args=None):
    rclpy.init(args=args)
    node = WallDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()