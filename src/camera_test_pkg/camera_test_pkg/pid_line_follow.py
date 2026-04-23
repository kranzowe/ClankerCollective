#!/usr/bin/env python3
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_2d_msgs.msg import Path2D

DEFAULT_SPEED = 0.38

DEFAULT_TURN_RATE = 7.0
CENTER = 0.7

# Path-follow control
PATH_TIMEOUT_SEC = 2
PATH_ANGLE_KP = 3.0
PATH_ANGLE_KD = 0.6
MAX_PATH_TURN = 7
TURN_BIAS = 0.0

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower_node')

        self.path_sub = self.create_subscription(
            Path2D,
            '/line_path',
            self.path_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.get_logger().info('Line follower node initialized with line-path steering...')

        # Latest visual path steering state
        self.path_angle = None
        self.path_angle_stamp = None
        self.prev_path_error = 0.0
        self.prev_path_error_stamp = None

    def path_callback(self, msg: Path2D):
        thetas = []

        for pose in msg.poses:
            if np.isfinite(pose.theta):
                thetas.append(pose.theta)

        if not thetas:
            self.path_angle = None
            self.path_angle_stamp = None
            return

        thetas = np.array(thetas, dtype=np.float32)

        # Circular mean of waypoint headings
        self.path_angle = float(np.arctan2(np.mean(np.sin(thetas)),
                                           np.mean(np.cos(thetas))))
        self.path_angle_stamp = time.monotonic()

    def has_fresh_path(self):
        return (
            self.path_angle is not None and
            self.path_angle_stamp is not None and
            (time.monotonic() - self.path_angle_stamp) < PATH_TIMEOUT_SEC
        )

    def compute_path_turn(self):
        now = time.monotonic()
        error = self.path_angle

        if self.prev_path_error_stamp is None:
            derivative = 0.0
        else:
            dt = now - self.prev_path_error_stamp
            derivative = (error - self.prev_path_error) / dt if dt > 1e-3 else 0.0

        self.prev_path_error = error
        self.prev_path_error_stamp = now

        turn = (PATH_ANGLE_KP * error + PATH_ANGLE_KD * derivative)
        turn = float(np.clip(turn, -MAX_PATH_TURN, MAX_PATH_TURN))
        return turn, error, derivative

    def scan_callback(self, msg):
        # Kept for potential future use / front obstacle detection
        pass

    def control_loop(self):
        twist = Twist()

        if self.has_fresh_path():
            turn_cmd, angle_error, angle_derivative = self.compute_path_turn()

            speed_scale = max(0.35, 1.0 - min(abs(angle_error) / 1.2, 0.65))
            twist.linear.x = DEFAULT_SPEED #* speed_scale
            twist.angular.z = turn_cmd + TURN_BIAS

            self.get_logger().info(
                f'path_err: {np.rad2deg(angle_error):.1f}deg | '
                f'path_d: {np.rad2deg(angle_derivative):.1f}deg/s | '
                f'linear.x: {twist.linear.x:.3f} | '
                f'angular.z: {twist.angular.z:.3f}'
            )
        else:
            # No fresh path: stop and wait
            self.prev_path_error_stamp = None
            twist.linear.x = 0.0
            twist.angular.z = 0.0 + TURN_BIAS
            self.get_logger().info('No fresh path received, stopping.')

        self.cmd_pub.publish(twist)


def main(args=None):
    print('Waiting 13 seconds before starting...')
    time.sleep(13)
    print('Starting line follower node!')
    rclpy.init(args=args)
    node = LineFollower()
    node.create_timer(0.1, node.control_loop)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
