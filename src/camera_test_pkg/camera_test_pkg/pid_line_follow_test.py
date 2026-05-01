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
PATH_TIMEOUT_SEC = 2
PATH_ANGLE_KP = -100.0
PATH_ANGLE_KD = -25.0
MAX_LEFT_TURN = 800.0
MAX_RIGHT_TURN = -1000.0
TURN_BIAS = 1450.0

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower_node')

        self.declare_parameter('default_speed', DEFAULT_SPEED)
        self.declare_parameter('path_angle_kp', PATH_ANGLE_KP)
        self.declare_parameter('path_angle_kd', PATH_ANGLE_KD)
        self.declare_parameter('right_bias', -220.0)
        self.declare_parameter('angle_error_threshold', 0.25)
        self.default_speed = self.get_parameter('default_speed').get_parameter_value().double_value
        self.path_angle_kp = self.get_parameter('path_angle_kp').get_parameter_value().double_value
        self.path_angle_kd = self.get_parameter('path_angle_kd').get_parameter_value().double_value
        self.right_bias = self.get_parameter('right_bias').get_parameter_value().double_value
        self.angle_error_threshold = self.get_parameter('angle_error_threshold').get_parameter_value().double_value
        self.add_on_set_parameters_callback(self._on_params_changed)
        
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
        self.turn_stable_count = 0
        self.right_turn_detected = False
        self.steps_right_turn = 0
       # self.fresh_path_flag = False
       # self.fresh_path_count = 0 

    def _on_params_changed(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == 'default_speed':
                self.default_speed = p.value
            elif p.name == 'path_angle_kp':
                self.path_angle_kp = p.value
            elif p.name == 'path_angle_kd':
                self.path_angle_kd = p.value
            elif p.name == 'right_bias':
                self.right_bias = p.value
            elif p.name == 'angle_error_threshold':
                self.angle_error_threshold = p.value
            self.get_logger().info(f'{p.name} updated to {p.value}')
        return SetParametersResult(successful=True)

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
                        
            if y_aligned and not self.right_turn_detected:
                self.get_logger().info('RIGHT TURN DETECTED')
                self.right_turn_detected = True

            # as long as right turn detected 
            if self.right_turn_detected:
                thetas.append(np.pi / 2)
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

        turn = (self.path_angle_kp * error + self.path_angle_kd * derivative)
        turn = float(np.clip(turn, MAX_RIGHT_TURN, MAX_LEFT_TURN))
        return turn, error, derivative

    def scan_callback(self, msg):
        # Kept for potential future use / front obstacle detection
        pass

    def control_loop(self):
        twist = Twist()

        # ------------------ RIGHT TURN MODE ------------------
        if self.right_turn_detected:

            twist.linear.x = self.default_speed * 1.01

            if self.has_fresh_path():
                turn_cmd, angle_error, _ = self.compute_path_turn()
                twist.angular.z = turn_cmd + TURN_BIAS + self.right_bias
            else:
                twist.angular.z = TURN_BIAS - 300.0

            
            if self.steps_right_turn < 8:
                self.steps_right_turn += 1
            else:
                self.right_turn_detected = False
                self.steps_right_turn = 0

        # ------------------ NORMAL PATH FOLLOW ------------------
        elif self.has_fresh_path():

            turn_cmd, angle_error, angle_derivative = self.compute_path_turn()

            # 🔥90 degree turn override
            if abs(angle_error) > 0.8:   # about 45°
                twist.linear.x = DEFAULT_SPEED
                twist.angular.z = TURN_BIAS + np.sign(angle_error) * 500

            else:
                #regular PID
                speed_scale = max(0.35, 1.0 - min(abs(angle_error) / 1.2, 0.65))
                twist.linear.x = DEFAULT_SPEED
                twist.angular.z = turn_cmd + TURN_BIAS

            self.get_logger().info(
                f'path_err: {np.rad2deg(angle_error):.1f}deg | '
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

