#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, TransformStamped, Vector3
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import numpy as np
from scipy.spatial.transform import Rotation


class OdomTFPublisher(Node):
    def __init__(self):
        super().__init__('odom_tf_publisher')

        self.declare_parameter('calibrate_at_startup', True)
        self.declare_parameter('calibration_duration', 2.0)
        self.declare_parameter('vx_sign', -1.0)
        self.declare_parameter('gyro_sign', 1.0)
        self.calibrate_at_startup = bool(self.get_parameter('calibrate_at_startup').value)
        self.calibration_duration = float(self.get_parameter('calibration_duration').value)
        self.vx_sign = float(self.get_parameter('vx_sign').value)
        self.gyro_sign = float(self.get_parameter('gyro_sign').value)

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_time = self.get_clock().now()

        self.vx = 0.0
        self.gyro_yaw_rate = 0.0
        self.have_gyro = False

        # Stationary-bias calibration. While calibrating, vx/gyro stay at zero so
        # a small standing offset can't push integrated odom to infinity.
        self.vx_bias = 0.0
        self.gyro_bias = 0.0
        self._calib_vx_samples = []
        self._calib_gyro_samples = []
        self._calib_done = not self.calibrate_at_startup
        self._calib_start_time = self.get_clock().now()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.create_subscription(Twist, '/ol_rates', self.ol_rates_cb, sensor_qos)
        self.create_subscription(Vector3, '/imu/gyro', self.gyro_cb, sensor_qos)

        # Integrate at a fixed rate so gyro updates between /ol_rates messages still tick yaw forward.
        self.timer = self.create_timer(0.02, self.step)  # 50 Hz

    def ol_rates_cb(self, msg):
        raw = float(msg.linear.x)
        if not self._calib_done:
            self._calib_vx_samples.append(raw)
            self.vx = 0.0
        else:
            self.vx = self.vx_sign * (raw - self.vx_bias)

    def gyro_cb(self, msg):
        raw = float(msg.z)
        self.have_gyro = True
        if not self._calib_done:
            self._calib_gyro_samples.append(raw)
            self.gyro_yaw_rate = 0.0
        else:
            self.gyro_yaw_rate = self.gyro_sign * (raw - self.gyro_bias)

    def _maybe_finalize_calibration(self, now):
        if self._calib_done:
            return
        elapsed = (now.nanoseconds - self._calib_start_time.nanoseconds) * 1e-9
        if elapsed < self.calibration_duration:
            return
        n_vx = len(self._calib_vx_samples)
        n_gyro = len(self._calib_gyro_samples)
        if n_vx > 0:
            self.vx_bias = float(np.mean(self._calib_vx_samples))
        if n_gyro > 0:
            self.gyro_bias = float(np.mean(self._calib_gyro_samples))
        self._calib_done = True
        self._calib_vx_samples = []
        self._calib_gyro_samples = []
        self.get_logger().info(
            f"Bias calibration done in {elapsed:.2f}s "
            f"({n_vx} vx, {n_gyro} gyro samples). "
            f"vx_bias={self.vx_bias:+.4f} m/s, "
            f"gyro_bias={np.degrees(self.gyro_bias):+.3f} deg/s"
        )

    def step(self):
        now = self.get_clock().now()
        last_s = self.last_time.seconds_nanoseconds()
        now_s = now.seconds_nanoseconds()
        dt = (now_s[0] + now_s[1] * 1e-9) - (last_s[0] + last_s[1] * 1e-9)
        self.last_time = now

        if dt <= 0.0 or dt > 1.0:
            return

        self._maybe_finalize_calibration(now)

        # Prefer measured gyro yaw rate over commanded; fall back if gyro missing.
        vyaw = self.gyro_yaw_rate if self.have_gyro else 0.0
        vx = self.vx

        self.yaw += vyaw * dt
        self.x += vx * np.cos(self.yaw) * dt
        self.y += vx * np.sin(self.yaw) * dt

        quat = Rotation.from_euler('z', self.yaw).as_quat()

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]
        odom.twist.twist.linear.x = vx
        odom.twist.twist.angular.z = vyaw
        self.odom_pub.publish(odom)

        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomTFPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
