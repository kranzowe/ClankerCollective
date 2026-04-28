#!/usr/bin/env python3
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_2d_msgs.msg import Path2D

DEFAULT_SPEED = 1400.0

DEFAULT_TURN_RATE = 7.0
CENTER = 0.7

# Path-follow control
PATH_TIMEOUT_SEC = 10
PATH_ANGLE_KP = -100.0 # 
PATH_ANGLE_KD = 0.0
MAX_LEFT_TURN = 500.0
MAX_RIGHT_TURN = -500.0
MIN_TURN = 60.0
TURN_BIAS = 1520.0

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
        self.steps_right_turn = 0
        self.fresh_path_flag = False
        self.fresh_path_count = 0 

        # Debug logging throttle
        self._last_fresh_status_log = 0.0

    def path_callback(self, msg: Path2D):
        thetas = []

        for pose in msg.poses[0:-4]:

            if np.isfinite(pose.theta):
                thetas.append(pose.theta)
        
        # if all(np.isfinite([pose.theta for pose in msg.poses[-4:]])):
        #     # Check if all final poses are horizontally aligned (within tolerance)
        #     final_poses = msg.poses[-3:]
        #     y_tolerance = 0.0  # adjust based on your image scale
        #     first_y = final_poses[0].y
        #     y_aligned = all(abs(pose.y - first_y) < y_tolerance for pose in final_poses)
            
        #     if y_aligned:
        #         if not self.right_turn_detected:
        #             self.get_logger().info('RIGHT TURN RIGHT TURN RIGHT TURN')
        #             self.right_turn_detected = True
        #             self.steps_right_turn = 0
        #         thetas.append(np.pi / 2)  # bias it rightwards
        #     else:
        #         self.get_logger().info(f'Not horizontal: y spread = {max(p.y for p in final_poses) - min(p.y for p in final_poses):.3f}')

        if not thetas:
            now = time.monotonic()
            prev_age = (now - self.path_angle_stamp) if self.path_angle_stamp is not None else None
            self.get_logger().warn(
                f'[path_callback] No finite theta values. poses={len(msg.poses)} | '
                f'prev_path_age={prev_age:.3f}s' if prev_age is not None
                else f'[path_callback] No finite theta values. poses={len(msg.poses)} | prev_path_age=None'
            )
            self.path_angle = None
            self.path_angle_stamp = None
            return

        thetas = np.array(thetas, dtype=np.float32)

        # Circular mean of waypoint headings
        self.path_angle = float(np.arctan2(np.mean(np.sin(thetas)),
                                           np.mean(np.cos(thetas))))
        self.path_angle_stamp = time.monotonic()
        self.get_logger().info(
            f'[path_callback] finite_thetas={len(thetas)} | '
            f'path_angle={np.rad2deg(self.path_angle):.2f} deg'
        )

    def has_fresh_path(self):
        now = time.monotonic()
        age = (now - self.path_angle_stamp) if self.path_angle_stamp is not None else float('inf')

        fresh = (
            self.path_angle is not None and
            self.path_angle_stamp is not None and
            age < PATH_TIMEOUT_SEC
        )

        # Print periodically so logs are readable
        if (now - self._last_fresh_status_log) > 0.5:
            self.get_logger().info(
                f'[has_fresh_path] fresh={fresh} | '
                f'path_angle_is_none={self.path_angle is None} | '
                f'path_stamp_is_none={self.path_angle_stamp is None} | '
                f'age={age:.3f}s | timeout={PATH_TIMEOUT_SEC}s'
            )
            self._last_fresh_status_log = now

        return fresh

    def compute_path_turn(self):
        now = time.monotonic()
        error = self.path_angle

        if self.prev_path_error_stamp is None:
            derivative = 0.0 #hi
        else:
            dt = now - self.prev_path_error_stamp
            derivative = (error - self.prev_path_error) / dt if dt > 1e-3 else 0.0

        self.prev_path_error = error
        self.prev_path_error_stamp = now

        turn = (PATH_ANGLE_KP * error + PATH_ANGLE_KD * derivative)
        if turn > 0.0:
            turn += MIN_TURN
        elif turn < 0.0:
            turn -= MIN_TURN
        else:
            turn = 0.0

        turn = float(np.clip(turn, MAX_RIGHT_TURN, MAX_LEFT_TURN))
        
        return turn, error, derivative

    def scan_callback(self, msg):
        # Kept for potential future use / front obstacle detection
        pass

    def control_loop(self):
        twist = Twist()
        
        if self.right_turn_detected:
            self.get_logger().info('TURN TURN TURN TURN TURN')
            twist.linear.x = DEFAULT_SPEED #* speed_scale
            twist.angular.z =  -250.0 + TURN_BIAS                   #do until we see a line instead of hardcoded, also change turn bias
            if self.steps_right_turn < 25 and not self.fresh_path_flag:
                self.steps_right_turn += 1
                if self.steps_right_turn >15 and self.has_fresh_path():  #want to get into the turn before looking for fresh path
                    self.fresh_path_count+=1   
                    if self.fresh_path_count > 6:
                        self.fresh_path_flag = True
            else:
                self.right_turn_detected = False
                self.steps_right_turn = 0
                self.fresh_path_flag = False
                self.fresh_path_count = 0

        elif self.has_fresh_path():
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
            now = time.monotonic()
            age = (now - self.path_angle_stamp) if self.path_angle_stamp is not None else float('inf')
            self.prev_path_error_stamp = None
            twist.linear.x = 1500.0
            twist.angular.z = TURN_BIAS
            self.get_logger().warn(
                f'[control_loop] No fresh path -> fallback command. '
                f'path_angle_is_none={self.path_angle is None} | '
                f'path_stamp_is_none={self.path_angle_stamp is None} | '
                f'age={age:.3f}s'
            )

        self.cmd_pub.publish(twist)


def main(args=None):
    # print('Waiting 13 seconds before starting...')
    # time.sleep(13)
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