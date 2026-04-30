#!/usr/bin/env python3
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_2d_msgs.msg import Path2D
from std_msgs.msg import Bool

DEFAULT_SPEED = 1418.0

DEFAULT_TURN_RATE = 7.0
CENTER = 0.7

# Path-follow control
PATH_TIMEOUT_SEC = 2
PATH_ANGLE_KP = -170.0
PATH_ANGLE_KD = -25.0
MAX_LEFT_TURN = 800.0
MAX_RIGHT_TURN = -1000.0
TURN_BIAS = 1450.0

STOP_DURATION_SEC = 2.0   # how long to hold at the stop sign (seconds)

# Stop sign states
STOP_STATE_READY    = 'READY'      # watching for a stop sign
STOP_STATE_STOPPING = 'STOPPING'   # saw one, stopping for x amount of seconds
STOP_STATE_COOLDOWN = 'COOLDOWN'   # done, ignoring signs until it clears, then ready to stop again


class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower_node')

        self.path_sub = self.create_subscription(
            Path2D,
            '/line_path',
            self.path_callback,
            10
        )

        self.stop_sub = self.create_subscription(       #do we see a stop sign??? True = yes False = Nooooo
            Bool,
            '/stop_sign_stop',
            self.stop_sign_callback,
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

        # Stop sign state machine
        self.stop_state = STOP_STATE_READY
        self.stop_start_time = None    # when we entered STOPPING
        self.stop_sign_active = False  # latest value from topic /stop_sign_stop

    # ------------------ Stop Sign Callback ------------------
    def stop_sign_callback(self, msg: Bool):
        self.stop_sign_active = msg.data

    # ------------------ Path Callback ------------------
    def path_callback(self, msg: Path2D):
        thetas = []

        for pose in msg.poses[0:-4]:
            if np.isfinite(pose.theta):
                thetas.append(pose.theta)

        if all(np.isfinite([pose.theta for pose in msg.poses[-4:]])):
            final_poses = msg.poses[-3:]
            y_tolerance = 20
            first_y = final_poses[0].y
            y_aligned = all(abs(pose.y - first_y) < y_tolerance for pose in final_poses)

            if y_aligned:
                if not self.right_turn_detected:
                    self.get_logger().info('RIGHT TURN RIGHT TURN RIGHT TURN')
                    self.right_turn_detected = True
                    self.steps_right_turn = 0
                thetas.append(np.pi / 2)
            else:
                self.get_logger().info(
                    f'Not horizontal: y spread = '
                    f'{max(p.y for p in final_poses) - min(p.y for p in final_poses):.3f}'
                )

        if not thetas:
            self.path_angle = None
            self.path_angle_stamp = None
            return

        thetas = np.array(thetas, dtype=np.float32)
        self.path_angle = float(np.arctan2(np.mean(np.sin(thetas)),
                                           np.mean(np.cos(thetas))))
        self.path_angle_stamp = time.monotonic()

    # ------------------ Helpers ------------------
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
        turn = float(np.clip(turn, MAX_RIGHT_TURN, MAX_LEFT_TURN))
        return turn, error, derivative

    def hold_position(self, twist):
        twist.linear.x = 1500.0
        twist.angular.z = TURN_BIAS
        self.cmd_pub.publish(twist)

    def scan_callback(self, msg):
        pass

    # ------------------ Control Loop ------------------
    def control_loop(self):
        twist = Twist()
        now = time.monotonic()

        # ---- Stop sign state machine (runs first, highest priority (PEMDAS of the events)) ----
        if self.stop_state == STOP_STATE_READY:
            if self.stop_sign_active:
                # Just saw a stop sign (bool is true, we see red), STOP
                self.get_logger().info('Stop sign! Stopping for 2 seconds.')
                self.stop_state = STOP_STATE_STOPPING
                self.stop_start_time = now
                self.hold_position(twist)
                return

        elif self.stop_state == STOP_STATE_STOPPING:
            if now - self.stop_start_time < STOP_DURATION_SEC:
                # Still within the 2-second holding period, we stay stopped
                self.hold_position(twist)
                return
            else:
                # 2 seconds are up, we enter cooldown phase, 
                self.get_logger().info('Stop complete. Entering cooldown.')
                self.stop_state = STOP_STATE_COOLDOWN

        elif self.stop_state == STOP_STATE_COOLDOWN:
            if self.stop_sign_active:
                # Sign still visible during cooldown: keep ignoring it, fall through to drive normally
                pass
            else:
                # Sign has cleared, ready to respond to the next one, return to origional STATE!!!
                self.get_logger().info('Stop sign cleared. Ready for next stop.')
                self.stop_state = STOP_STATE_READY

        # ---- Driving logic from other PID follower code, remains the same, except for P and D terms----
        if self.right_turn_detected:
            self.get_logger().info('TURN TURN TURN TURN TURN')
            twist.linear.x = DEFAULT_SPEED
            twist.angular.z = -300.0 + TURN_BIAS
            if self.steps_right_turn < 25 and not self.fresh_path_flag:
                self.steps_right_turn += 1
                if self.steps_right_turn > 15 and self.has_fresh_path():
                    self.fresh_path_count += 1
                    if self.fresh_path_count > 10:
                        self.fresh_path_flag = True
            else:
                self.right_turn_detected = False
                self.steps_right_turn = 0
                self.fresh_path_flag = False
                self.fresh_path_count = 0

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
        else:
            self.prev_path_error_stamp = None
            twist.linear.x = 1500.0
            twist.angular.z = TURN_BIAS
            self.get_logger().info('No fresh path received, stopping.')

        self.cmd_pub.publish(twist)


def main(args=None):
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


#entry point
if __name__ == '__main__':
    main()