#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import numpy as np


class WallDetector(Node):
    def __init__(self):
        super().__init__('wall_detector_node')

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.distance_pub = self.create_publisher(Float32, 'ld_distance', 10)

        self.get_logger().info('front wall distance detector initialized...')

    def scan_callback(self, msg):
        # Find index corresponding to 0 rad (front of car)
        center_idx = int((0.0 - msg.angle_min) / msg.angle_increment)

        # Define front angle window (±2 degrees)
        angle_range_deg = 2.0
        angle_range_rad = np.deg2rad(angle_range_deg)
        index_offset = int(angle_range_rad / msg.angle_increment)

        # Clamp slice bounds for safety
        start_idx = max(0, center_idx - index_offset)
        end_idx = min(len(msg.ranges), center_idx + index_offset + 1)

        # Extract front ranges
        front_ranges = msg.ranges[start_idx:end_idx]

        # Filter valid values
        valid_ranges = [r for r in front_ranges if msg.range_min < r < msg.range_max and np.isfinite(r)]

        distance_msg = Float32()

        if valid_ranges:
            # Median is robust to noise
            front_distance = float(np.median(valid_ranges))
            distance_msg.data = front_distance
            self.distance_pub.publish(distance_msg)

            if front_distance < 1.0:
                self.get_logger().warn(f'⚠️ warning! too close: {front_distance:.2f} m')
            else:
                self.get_logger().info(f'front wall dist: {front_distance:.2f} m')
        else:
            # Option 1: publish NaN when no valid detection
            distance_msg.data = float('nan')
            self.distance_pub.publish(distance_msg)

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


if __name__ == '__main__':
    main()