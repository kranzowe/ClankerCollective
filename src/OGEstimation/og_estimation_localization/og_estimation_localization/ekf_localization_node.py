#!/usr/bin/env python3
"""
LIDAR-EKF Localization (Challenge 1).

State: x = [x, y, theta] in the map frame.
Predict: unicycle kinematics driven by /ol_rates.linear.x and /imu/gyro.z.
Update: point-to-point ICP scan-match against a pre-loaded occupancy-grid point
cloud — the corrected pose is the measurement (H = I).

The math lives in core.py (no ROS imports → unit-testable in isolation).
This module is the ROS wrapper + RViz visualization.
"""
import math
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import (Point, PoseStamped, PoseWithCovarianceStamped,
                               TransformStamped, Twist, Vector3)
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       HistoryPolicy)
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Float32MultiArray, MultiArrayDimension
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from og_estimation_localization.core import (ekf_predict, ekf_update, icp_se2,
                                          load_occupancy_pointcloud,
                                          se2_from_xyt, wrap_angle,
                                          xyt_from_se2)
from rclpy.parameter import Parameter
from rclpy.node import ParameterNotDeclaredException

# =============================================================================
# RViz visualization helpers (depend on ROS messages but not on Node state).
# =============================================================================

def make_pointcloud2(points_xy, frame_id, stamp):
    """Build a PointCloud2 from an (N, 2) numpy array. z = 0."""
    pts = np.zeros((points_xy.shape[0], 3), dtype=np.float32)
    pts[:, 0] = points_xy[:, 0]
    pts[:, 1] = points_xy[:, 1]
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = pts.shape[0]
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = pts.tobytes()
    return msg


def make_correspondence_markers(src_pts, dst_pts, frame_id, stamp,
                                ns='icp_correspondences', max_lines=400):
    """LINE_LIST marker connecting src -> dst pairs (src and dst already in `frame_id`)."""
    n = min(len(src_pts), len(dst_pts), max_lines)
    if n == 0:
        m = Marker()
        m.action = Marker.DELETE
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = ns
        return MarkerArray(markers=[m])

    if len(src_pts) > max_lines:
        idx = np.linspace(0, len(src_pts) - 1, max_lines).astype(int)
        src_pts = src_pts[idx]
        dst_pts = dst_pts[idx]

    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = 0
    m.type = Marker.LINE_LIST
    m.action = Marker.ADD
    m.scale.x = 0.01
    m.color = ColorRGBA(r=1.0, g=0.6, b=0.0, a=0.8)
    m.pose.orientation.w = 1.0
    pts = []
    for s, d in zip(src_pts, dst_pts):
        pts.append(Point(x=float(s[0]), y=float(s[1]), z=0.0))
        pts.append(Point(x=float(d[0]), y=float(d[1]), z=0.0))
    m.points = pts
    return MarkerArray(markers=[m])


def make_covariance_ellipse_marker(x, y, theta, P_xy, frame_id, stamp,
                                   ns='ekf_cov', sigma=2.0, segments=48):
    """
    LINE_STRIP ellipse for 2D position covariance (sigma sigma confidence).
    P_xy is the 2x2 (x, y) sub-covariance.
    """
    eigvals, eigvecs = np.linalg.eigh(P_xy)
    eigvals = np.maximum(eigvals, 1e-9)
    a = sigma * math.sqrt(eigvals[0])
    b = sigma * math.sqrt(eigvals[1])
    R = eigvecs

    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = 0
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.scale.x = 0.02
    m.color = ColorRGBA(r=0.2, g=0.8, b=1.0, a=0.9)
    m.pose.orientation.w = 1.0
    pts = []
    for k in range(segments + 1):
        phi = 2.0 * math.pi * k / segments
        px = a * math.cos(phi)
        py = b * math.sin(phi)
        wx, wy = R @ np.array([px, py])
        pts.append(Point(x=float(x + wx), y=float(y + wy), z=0.0))
    m.points = pts
    return m


# =============================================================================
# ROS Node
# =============================================================================

class EKFLocalization(Node):
    def __init__(self):
        super().__init__('ekf_localization')

        # ---- parameters -----------------------------------------------------
        self.declare_parameter('map_yaml', '')
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)
        self.declare_parameter('q_xy', 0.05)
        self.declare_parameter('q_yaw', 0.03)
        self.declare_parameter('r_xy', 0.05)
        self.declare_parameter('r_yaw', 0.03)
        self.declare_parameter('icp_max_iters', 15)
        self.declare_parameter('icp_max_corr_dist', 0.5)
        self.declare_parameter('icp_residual_reject', 0.30)
        self.declare_parameter('scan_stride', 2)
        self.declare_parameter('scan_min_range', 0.20)
        self.declare_parameter('scan_max_range', 12.0)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('laser_frame', 'laser')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('predict_rate', 50.0)
        self.declare_parameter('publish_debug_viz', True)
        self.declare_parameter('trajectory_max_len', 2000)
        self.declare_parameter('calibrate_at_startup', True)
        self.declare_parameter('calibration_duration', 2.0)
        self.declare_parameter('vx_sign', -1.0)
        self.declare_parameter('gyro_sign', 1.0)
        # Lost-localization recovery.
        self.declare_parameter('recovery_enable', True)
        self.declare_parameter('recovery_reject_count', 5)
        self.declare_parameter('recovery_success_count', 5)
        self.declare_parameter('recovery_stop_throttle', 1500.0)
        self.declare_parameter('recovery_inflate_xy', 1.0)
        self.declare_parameter('recovery_inflate_yaw_deg', 30.0)
        self.declare_parameter('recovery_cmd_topic', '/cmd_vel')

        map_yaml = self.get_parameter('map_yaml').get_parameter_value().string_value
        if not map_yaml:
            raise RuntimeError("ekf_localization requires the 'map_yaml' parameter.")

        # ---- state ----------------------------------------------------------
        self.state = np.array([
            float(self.get_parameter('initial_x').value),
            float(self.get_parameter('initial_y').value),
            float(self.get_parameter('initial_yaw').value),
        ])
        self.P = np.diag([0.5**2, 0.5**2, math.radians(15.0)**2])

        self.q_xy = float(self.get_parameter('q_xy').value)
        self.q_yaw = float(self.get_parameter('q_yaw').value)
        self.r_xy = float(self.get_parameter('r_xy').value)
        self.r_yaw = float(self.get_parameter('r_yaw').value)
        self.icp_max_iters = int(self.get_parameter('icp_max_iters').value)
        self.icp_max_corr_dist = float(self.get_parameter('icp_max_corr_dist').value)
        self.icp_residual_reject = float(self.get_parameter('icp_residual_reject').value)
        self.scan_stride = max(1, int(self.get_parameter('scan_stride').value))
        self.scan_min_range = float(self.get_parameter('scan_min_range').value)
        self.scan_max_range = float(self.get_parameter('scan_max_range').value)
        self.odom_frame = self.get_parameter('odom_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.laser_frame = self.get_parameter('laser_frame').get_parameter_value().string_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        predict_rate = float(self.get_parameter('predict_rate').value)
        self.publish_debug_viz = bool(self.get_parameter('publish_debug_viz').value)
        self.trajectory_max_len = int(self.get_parameter('trajectory_max_len').value)

        self.vx = 0.0
        self.gyro_yaw_rate = 0.0
        self.have_gyro = False

        # Stationary-bias calibration. While calibrating, vx/gyro are forced to
        # zero so the predict step doesn't integrate the raw biased reading.
        self.calibrate_at_startup = bool(self.get_parameter('calibrate_at_startup').value)
        self.calibration_duration = float(self.get_parameter('calibration_duration').value)
        self.vx_sign = float(self.get_parameter('vx_sign').value)
        self.gyro_sign = float(self.get_parameter('gyro_sign').value)
        self.vx_bias = 0.0
        self.gyro_bias = 0.0
        self._calib_vx_samples = []
        self._calib_gyro_samples = []
        self._calib_done = not self.calibrate_at_startup
        self._calib_start_time = self.get_clock().now()

        # Lost-localization recovery state.
        self.recovery_enable = bool(self.get_parameter('recovery_enable').value)
        self.recovery_reject_count = int(self.get_parameter('recovery_reject_count').value)
        self.recovery_success_count = int(self.get_parameter('recovery_success_count').value)
        self.recovery_stop_throttle = float(self.get_parameter('recovery_stop_throttle').value)
        self.recovery_inflate_xy = float(self.get_parameter('recovery_inflate_xy').value)
        self.recovery_inflate_yaw_deg = float(self.get_parameter('recovery_inflate_yaw_deg').value)
        recovery_cmd_topic = self.get_parameter('recovery_cmd_topic').get_parameter_value().string_value
        self._consecutive_rejects = 0
        self._consecutive_successes = 0
        self._in_recovery = False

        self.lock = threading.Lock()
        self.last_predict_time = self.get_clock().now()
        self.trajectory = []

        # ---- map cloud + KD-tree -------------------------------------------
        self.get_logger().info(f"Loading map cloud from {map_yaml} ...")
        self.map_cloud = load_occupancy_pointcloud(map_yaml)
        self.get_logger().info(f"Map cloud: {self.map_cloud.shape[0]} occupied points.")
        self.map_tree = cKDTree(self.map_cloud)

        # ---- subscribers / publishers --------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(Twist, '/ol_rates', self.ol_rates_cb, sensor_qos)
        self.create_subscription(Vector3, '/imu/gyro', self.gyro_cb, sensor_qos)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, sensor_qos)

        # RViz "2D Pose Estimate" tool publishes here. Use a reliable QoS so
        # we don't drop the single message that comes from each click.
        initialpose_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose',
            self.initialpose_cb, initialpose_qos)

        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/ekf_pose', 10)
        self.cmd_pub = self.create_publisher(Twist, recovery_cmd_topic, 10)

        # Debug / visualization publishers.
        self.map_cloud_pub = self.create_publisher(
            PointCloud2, '/ekf/map_cloud', latched_qos)
        self.scan_in_map_pub = self.create_publisher(
            PointCloud2, '/ekf/scan_in_map', 10)
        self.correspondences_pub = self.create_publisher(
            MarkerArray, '/ekf/icp_correspondences', 10)
        self.cov_marker_pub = self.create_publisher(
            Marker, '/ekf/covariance_ellipse', 10)
        self.trajectory_pub = self.create_publisher(
            Path, '/ekf/trajectory', 10)
        self.icp_diag_pub = self.create_publisher(
            Float32MultiArray, '/ekf/icp_diagnostics', 10)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.predict_timer = self.create_timer(1.0 / predict_rate, self.predict_step)
        self.recovery_timer = self.create_timer(0.1, self._publish_stop_if_recovering)
        self._warned_no_laser_tf = False

        # Latch the map cloud so RViz picks it up whenever it (re)connects.
        if self.publish_debug_viz:
            self.map_cloud_pub.publish(make_pointcloud2(
                self.map_cloud, self.map_frame, self.get_clock().now().to_msg()))

    # ---- callbacks ---------------------------------------------------------

    def ol_rates_cb(self, msg):
        raw = float(msg.linear.x)
        with self.lock:
            if not self._calib_done:
                self._calib_vx_samples.append(raw)
                self.vx = 0.0
            else:
                self.vx = self.vx_sign * (raw - self.vx_bias)

    def gyro_cb(self, msg):
        raw = float(msg.z)
        with self.lock:
            self.have_gyro = True
            if not self._calib_done:
                self._calib_gyro_samples.append(raw)
                self.gyro_yaw_rate = 0.0
            else:
                self.gyro_yaw_rate = self.gyro_sign * (raw - self.gyro_bias)

    def initialpose_cb(self, msg):
        """Re-seed EKF state + covariance from RViz "2D Pose Estimate"."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]

        cov6 = np.asarray(msg.pose.covariance, dtype=np.float64).reshape(6, 6)
        idx = np.array([0, 1, 5])
        P_new = cov6[np.ix_(idx, idx)]
        if not np.any(np.diag(P_new) > 1e-9):
            P_new = np.diag([0.5**2, 0.5**2, math.radians(15.0)**2])

        with self.lock:
            self.state = np.array([float(p.x), float(p.y), wrap_angle(float(yaw))])
            self.P = P_new
            self.trajectory = []
            self.last_predict_time = self.get_clock().now()

        self.get_logger().info(
            f"Re-seeded EKF from /initialpose: "
            f"x={p.x:.3f}, y={p.y:.3f}, yaw_deg={math.degrees(yaw):.2f}"
        )

    def scan_cb(self, msg):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        angles = msg.angle_min + np.arange(len(ranges), dtype=np.float32) * msg.angle_increment

        if self.scan_stride > 1:
            ranges = ranges[::self.scan_stride]
            angles = angles[::self.scan_stride]

        finite = np.isfinite(ranges) & (ranges > self.scan_min_range) & (ranges < self.scan_max_range)
        if finite.sum() < 30:
            return
        r = ranges[finite]
        a = angles[finite]
        scan_xy_laser = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)

        scan_xy_base = self._scan_to_base_frame(scan_xy_laser)
        if scan_xy_base is None:
            return

        with self.lock:
            init_xyt = (float(self.state[0]), float(self.state[1]), float(self.state[2]))

        result = icp_se2(
            scan_xy_base, self.map_tree, init_xyt,
            max_iters=self.icp_max_iters, max_corr_dist=self.icp_max_corr_dist,
            return_correspondences=self.publish_debug_viz,
        )

        if self.publish_debug_viz:
            x_icp, y_icp, t_icp, residual, src_pairs, dst_pairs = result
        else:
            x_icp, y_icp, t_icp, residual = result
            src_pairs = dst_pairs = None

        # Diagnostics: published whether we accept or reject.
        self._publish_icp_diag(residual, init_xyt, (x_icp, y_icp, t_icp))

        if not math.isfinite(residual) or residual > self.icp_residual_reject:
            self.get_logger().warn(
                f"ICP residual {residual:.3f} > {self.icp_residual_reject:.3f}; rejecting update."
            )
            self._on_icp_rejected()
            return

        with self.lock:
            self.state, self.P = ekf_update(
                self.state, self.P, (x_icp, y_icp, t_icp), self.r_xy, self.r_yaw)
            self.publish_pose(msg.header.stamp)
            self.publish_map_to_odom_tf(self.get_clock().now().to_msg())
            if self.publish_debug_viz:
                self._publish_debug_after_update(msg.header.stamp, src_pairs, dst_pairs)

        self._on_icp_accepted()

    # ---- EKF steps (thin wrappers around pure functions) -------------------

    def predict_step(self):
        now = self.get_clock().now()
        last_s = self.last_predict_time.seconds_nanoseconds()
        now_s = now.seconds_nanoseconds()
        dt = (now_s[0] + now_s[1] * 1e-9) - (last_s[0] + last_s[1] * 1e-9)
        self.last_predict_time = now
        if dt <= 0.0 or dt > 1.0:
            return

        self._maybe_finalize_calibration(now)

        with self.lock:
            v = self.vx
            w = self.gyro_yaw_rate if self.have_gyro else 0.0
            self.state, self.P = ekf_predict(
                self.state, self.P, v, w, dt, self.q_xy, self.q_yaw)

            self.publish_pose(now.to_msg())
            self.publish_map_to_odom_tf(now.to_msg())
            if self.publish_debug_viz:
                self._publish_covariance_ellipse(now.to_msg())

    # ---- lost-localization recovery ----------------------------------------

    def _on_icp_rejected(self):
        if not self.recovery_enable:
            return
        self._consecutive_successes = 0
        self._consecutive_rejects += 1
        if not self._in_recovery and self._consecutive_rejects >= self.recovery_reject_count:
            self._enter_recovery()

    def _on_icp_accepted(self):
        if not self.recovery_enable:
            return
        self._consecutive_rejects = 0
        self._consecutive_successes += 1
        if self._in_recovery and self._consecutive_successes >= self.recovery_success_count:
            self._exit_recovery()

    def _enter_recovery(self):
        """Stop the car, re-arm bias calibration, inflate covariance."""
        self.get_logger().warn(
            f"Localization lost ({self._consecutive_rejects} consecutive ICP rejections). "
            f"Stopping car, re-calibrating bias, inflating covariance."
        )
        # Store previous speed and set to 0
        if self._ol_speed_prev is None:
            # Try to get the current value from the parameter server
            # (You may need to implement a parameter client to fetch from lidar_bug)
            self._ol_speed_prev = 100  # Default/fallback value
        self._set_ol_speed(0)
        self._in_recovery = True
        with self.lock:
            # Re-arm calibration: vx/gyro will be held at 0 for the duration,
            # then biases re-estimated from the new stationary samples.
            self._calib_done = False
            self._calib_vx_samples = []
            self._calib_gyro_samples = []
            self._calib_start_time = self.get_clock().now()
            self.P = np.diag([
                self.recovery_inflate_xy ** 2,
                self.recovery_inflate_xy ** 2,
                math.radians(self.recovery_inflate_yaw_deg) ** 2,
            ])

    def _exit_recovery(self):
        self.get_logger().info(
            f"Localization recovered ({self._consecutive_successes} consecutive good ICP updates). "
            f"Releasing stop command."
        )
        # Restore previous speed
        if self._ol_speed_prev is not None:
            self._set_ol_speed(self._ol_speed_prev)
            self._ol_speed_prev = None
        self._in_recovery = False

    def _publish_stop_if_recovering(self):
        # No longer needed to publish Twist, since we're controlling via param
        pass

    def _maybe_finalize_calibration(self, now):
        """If the calibration window has elapsed, lock in the bias means."""
        if self._calib_done:
            return
        elapsed = (now.nanoseconds - self._calib_start_time.nanoseconds) * 1e-9
        if elapsed < self.calibration_duration:
            return
        with self.lock:
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
            f"gyro_bias={math.degrees(self.gyro_bias):+.3f} deg/s"
        )

    # ---- TF / publishing helpers -------------------------------------------

    def _scan_to_base_frame(self, scan_xy_laser):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.laser_frame, rclpy.time.Time())
        except Exception as e:
            if not self._warned_no_laser_tf:
                self.get_logger().warn(
                    f"TF {self.base_frame}<-{self.laser_frame} unavailable ({e}); "
                    f"assuming identity. (suppressing further warnings)"
                )
                self._warned_no_laser_tf = True
            return scan_xy_laser

        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
        c, s = math.cos(yaw), math.sin(yaw)
        x = c * scan_xy_laser[:, 0] - s * scan_xy_laser[:, 1] + tx
        y = s * scan_xy_laser[:, 0] + c * scan_xy_laser[:, 1] + ty
        return np.stack([x, y], axis=1)

    def publish_pose(self, stamp):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x = float(self.state[0])
        msg.pose.pose.position.y = float(self.state[1])
        q = Rotation.from_euler('z', float(self.state[2])).as_quat()
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]
        cov = np.zeros((6, 6))
        cov[0, 0] = self.P[0, 0]; cov[0, 1] = self.P[0, 1]
        cov[1, 0] = self.P[1, 0]; cov[1, 1] = self.P[1, 1]
        cov[5, 5] = self.P[2, 2]
        cov[0, 5] = self.P[0, 2]; cov[5, 0] = self.P[2, 0]
        cov[1, 5] = self.P[1, 2]; cov[5, 1] = self.P[2, 1]
        msg.pose.covariance = cov.flatten().tolist()
        self.pose_pub.publish(msg)

        if self.publish_debug_viz:
            self._append_trajectory(stamp)

    def publish_map_to_odom_tf(self, stamp):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.odom_frame, self.base_frame, rclpy.time.Time())
        except Exception:
            return

        ox = tf.transform.translation.x
        oy = tf.transform.translation.y
        oq = tf.transform.rotation
        oyaw = Rotation.from_quat([oq.x, oq.y, oq.z, oq.w]).as_euler('xyz')[2]

        T_map_base = se2_from_xyt(float(self.state[0]), float(self.state[1]), float(self.state[2]))
        T_odom_base = se2_from_xyt(ox, oy, oyaw)
        T_map_odom = T_map_base @ np.linalg.inv(T_odom_base)
        mx, my, myaw = xyt_from_se2(T_map_odom)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame
        t.transform.translation.x = mx
        t.transform.translation.y = my
        t.transform.translation.z = 0.0
        q = Rotation.from_euler('z', myaw).as_quat()
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

    # ---- visualization ------------------------------------------------------

    def _publish_covariance_ellipse(self, stamp):
        m = make_covariance_ellipse_marker(
            float(self.state[0]), float(self.state[1]), float(self.state[2]),
            self.P[:2, :2], self.map_frame, stamp)
        self.cov_marker_pub.publish(m)

    def _publish_debug_after_update(self, stamp, src_pairs, dst_pairs):
        if src_pairs is not None and len(src_pairs) > 0:
            self.scan_in_map_pub.publish(make_pointcloud2(src_pairs, self.map_frame, stamp))
            self.correspondences_pub.publish(
                make_correspondence_markers(src_pairs, dst_pairs, self.map_frame, stamp))
        self._publish_covariance_ellipse(stamp)

    def _publish_icp_diag(self, residual, init_xyt, final_xyt):
        m = Float32MultiArray()
        m.layout.dim = [MultiArrayDimension(label='[residual,dx,dy,dtheta]', size=4, stride=4)]
        dx = float(final_xyt[0] - init_xyt[0])
        dy = float(final_xyt[1] - init_xyt[1])
        dtheta = float(wrap_angle(final_xyt[2] - init_xyt[2]))
        m.data = [float(residual), dx, dy, dtheta]
        self.icp_diag_pub.publish(m)

    def _append_trajectory(self, stamp):
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.map_frame
        ps.pose.position.x = float(self.state[0])
        ps.pose.position.y = float(self.state[1])
        q = Rotation.from_euler('z', float(self.state[2])).as_quat()
        ps.pose.orientation.x = q[0]; ps.pose.orientation.y = q[1]
        ps.pose.orientation.z = q[2]; ps.pose.orientation.w = q[3]
        self.trajectory.append(ps)
        if len(self.trajectory) > self.trajectory_max_len:
            self.trajectory = self.trajectory[-self.trajectory_max_len:]
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = self.map_frame
        path.poses = self.trajectory
        self.trajectory_pub.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = EKFLocalization()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
