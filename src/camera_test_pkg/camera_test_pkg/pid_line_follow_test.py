#!/usr/bin/env python3
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_2d_msgs.msg import Path2D

DEFAULT_SPEED = 1415.0

DEFAULT_TURN_RATE = 7.0
CENTER = 0.7

# Path-follow control
PATH_TIMEOUT_SEC = 2
PATH_ANGLE_KP = -170.0
PATH_ANGLE_KD = -25.0
MAX_LEFT_TURN = 800.0
MAX_RIGHT_TURN = -1000.0
TURN_BIAS = 1450.0

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

        self.right_turn_detected = False
       # self.steps_right_turn = 0
       # self.fresh_path_flag = False
       # self.fresh_path_count = 0 

    def angle_diff(self, a, b):
        d = a - b
        return (d + np.pi) % (2 * np.pi) - np.pi
    
    def path_callback(self, msg: Path2D):
        thetas = []

        for pose in msg.poses[0:-4]:

            if np.isfinite(pose.theta):
                thetas.append(pose.theta)
        
        if all(np.isfinite([pose.theta for pose in msg.poses[-4:]])):
            # Check if all final poses are horizontally aligned (within tolerance)
            final_poses = msg.poses[-3:]
            y_tolerance = 10  # adjust based on your image scale
            first_y = final_poses[0].y
            y_aligned = all(abs(pose.y - first_y) < y_tolerance for pose in final_poses)
            
            if y_aligned:
                if not self.right_turn_detected:
                    self.get_logger().info('RIGHT TURN RIGHT TURN RIGHT TURN')
                    self.right_turn_detected = True
                 #   self.steps_right_turn = 0
                thetas.append(np.pi / 2)  # bias it rightwards
            else:
                self.get_logger().info(f'Not horizontal: y spread = {max(p.y for p in final_poses) - min(p.y for p in final_poses):.3f}')

        if not thetas:
            self.path_angle = None
            self.path_angle_stamp = None
            return

        thetas = np.array(thetas, dtype=np.float32)

        # Circular mean of waypoint headings
        angle = float(np.arctan2(np.mean(np.sin(thetas)),
                         np.mean(np.cos(thetas))))

        # normalize to [-pi, pi]
        self.path_angle = (angle + np.pi) % (2 * np.pi) - np.pi
        self.path_angle_stamp = time.monotonic()




    def has_fresh_path(self):
        return (
            self.path_angle is not None and
            self.path_angle_stamp is not None and
            (time.monotonic() - self.path_angle_stamp) < PATH_TIMEOUT_SEC
        )

    def compute_path_turn(self):
        now = time.monotonic()
        error = (self.path_angle + np.pi) % (2 * np.pi) - np.pi

        if self.prev_path_error_stamp is None:
            derivative = 0.0
        else:
            dt = now - self.prev_path_error_stamp
            if dt > 1e-3:
                derivative = self.angle_diff(error, self.prev_path_error) / dt
            else:
                derivative = 0.0

        self.prev_path_error = error
        self.prev_path_error_stamp = now

        turn = (PATH_ANGLE_KP * error + PATH_ANGLE_KD * derivative)
        turn = float(np.clip(turn, MAX_RIGHT_TURN, MAX_LEFT_TURN))
        return turn, error, derivative

    def scan_callback(self, msg):
        # Kept for potential future use / front obstacle detection
        pass

    def control_loop(self):
        twist = Twist()

        # ------------------ RIGHT TURN MODE ------------------
        if self.right_turn_detected:

            if self.has_fresh_path():
                turn_cmd, angle_error, angle_derivative = self.compute_path_turn()

                RIGHT_BIAS = -180.0   # tune this bias to adjust how aggressively it turns right during the turn maneuver

                twist.linear.x = DEFAULT_SPEED
                twist.angular.z = turn_cmd + TURN_BIAS + RIGHT_BIAS

                # if the path angle error is small enough, consider the right turn complete
                if abs(angle_error) < 0.18:
                    self.get_logger().info('Right turn complete')
                    self.right_turn_detected = False

            else:
                # if no fresh path, just keep turning right at a fixed rate (with bias)
                twist.linear.x = DEFAULT_SPEED
                twist.angular.z = TURN_BIAS - 300.0

        # ------------------ NORMAL PATH FOLLOW ------------------
        elif self.has_fresh_path():

            turn_cmd, angle_error, angle_derivative = self.compute_path_turn()

            speed_scale = max(0.35, 1.0 - min(abs(angle_error) / 1.2, 0.65))
            twist.linear.x = DEFAULT_SPEED
            twist.angular.z = turn_cmd + TURN_BIAS

            self.get_logger().info(
                f'path_err: {np.rad2deg(angle_error):.1f}deg | '
                f'path_d: {np.rad2deg(angle_derivative):.1f}deg/s | '
                f'linear.x: {twist.linear.x:.3f} | '
                f'angular.z: {twist.angular.z:.3f}'
            )

        # ------------------ NO PATH ------------------
        else:
            self.prev_path_error_stamp = None
            twist.linear.x = 1500.0
            twist.angular.z = TURN_BIAS
            self.get_logger().info('No fresh path received, stopping.')

        self.cmd_pub.publish(twist)


def main(args=None):
    print('Waiting 5 seconds before starting...')
    time.sleep(5)
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
    #nick
