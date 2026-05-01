"""
Pure-math core for the LIDAR-EKF localization node.

This module deliberately has NO ROS imports so it can be unit-tested in
isolation (see ../test/). The ROS node imports from here.
"""
import math
import os

import numpy as np
import yaml
from PIL import Image


def wrap_angle(a):
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def se2_from_xyt(x, y, t):
    """Build a 3x3 SE(2) homogeneous transform from (x, y, theta)."""
    c, s = math.cos(t), math.sin(t)
    T = np.eye(3)
    T[0, 0] = c; T[0, 1] = -s; T[0, 2] = x
    T[1, 0] = s; T[1, 1] = c;  T[1, 2] = y
    return T


def xyt_from_se2(T):
    """Decompose a 3x3 SE(2) transform back into (x, y, theta)."""
    return T[0, 2], T[1, 2], math.atan2(T[1, 0], T[0, 0])


def transform_points(pts, x, y, theta):
    """Apply SE(2) (x, y, theta) to an (N, 2) point array."""
    c, s = math.cos(theta), math.sin(theta)
    out = np.empty_like(pts, dtype=np.float64)
    out[:, 0] = c * pts[:, 0] - s * pts[:, 1] + x
    out[:, 1] = s * pts[:, 0] + c * pts[:, 1] + y
    return out


def icp_se2(scan_pts, map_kdtree, init_xyt, max_iters=15,
            max_corr_dist=0.5, tol_trans=1e-3, tol_rot=1e-3,
            min_correspondences=20, return_correspondences=False):
    """
    Point-to-point ICP in SE(2).

    Returns (x, y, theta, mean_residual) by default, or
            (x, y, theta, mean_residual, src_pts_in_map, dst_pts_in_map)
    if return_correspondences=True.

    The reported `mean_residual` is the mean nearest-neighbor distance across
    *all* scan points at the final pose (no upper bound). This is always finite
    and reflects how well the scan actually aligns with the map — a useful
    diagnostic even when ICP fails to converge. Internal iteration uses the
    bounded query (max_corr_dist) for convergence; if fewer than
    `min_correspondences` scan points fall within `max_corr_dist`, ICP aborts
    early without updating the pose, and the returned residual will be large
    (telling you the initial pose is too far from the map).

    scan_pts: (N, 2) in robot frame.
    map_kdtree: scipy.spatial.cKDTree built over (M, 2) occupied map points.
    init_xyt: (x, y, theta) initial guess, robot-in-map.
    """
    x, y, t = float(init_xyt[0]), float(init_xyt[1]), float(init_xyt[2])
    src_v_last = np.zeros((0, 2))
    dst_v_last = np.zeros((0, 2))

    for _ in range(max_iters):
        c, s = math.cos(t), math.sin(t)
        sx = c * scan_pts[:, 0] - s * scan_pts[:, 1] + x
        sy = s * scan_pts[:, 0] + c * scan_pts[:, 1] + y
        src = np.stack([sx, sy], axis=1)

        dists, idx = map_kdtree.query(src, k=1, distance_upper_bound=max_corr_dist)
        valid = np.isfinite(dists)
        if valid.sum() < min_correspondences:
            break

        src_v = src[valid]
        dst_v = map_kdtree.data[idx[valid]]
        src_v_last = src_v
        dst_v_last = dst_v

        # Closed-form 2D Procrustes: best R, t aligning src_v onto dst_v.
        src_c = src_v.mean(axis=0)
        dst_c = dst_v.mean(axis=0)
        a = src_v - src_c
        b = dst_v - dst_c
        sxy = float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
        cxy = float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
        dtheta = math.atan2(sxy, cxy)
        cd, sd = math.cos(dtheta), math.sin(dtheta)
        dx = dst_c[0] - (cd * src_c[0] - sd * src_c[1])
        dy = dst_c[1] - (sd * src_c[0] + cd * src_c[1])

        T_inc = np.array([[cd, -sd, dx], [sd, cd, dy], [0.0, 0.0, 1.0]])
        T_cur = se2_from_xyt(x, y, t)
        T_new = T_inc @ T_cur
        x, y, t_new = xyt_from_se2(T_new)
        if abs(dx) < tol_trans and abs(dy) < tol_trans and abs(dtheta) < tol_rot:
            t = wrap_angle(t_new)
            break
        t = wrap_angle(t_new)

    # Final residual: unbounded mean nearest-neighbor distance over ALL scan
    # points at the converged pose. Always finite, always informative.
    c, s = math.cos(t), math.sin(t)
    sx = c * scan_pts[:, 0] - s * scan_pts[:, 1] + x
    sy = s * scan_pts[:, 0] + c * scan_pts[:, 1] + y
    src_final = np.stack([sx, sy], axis=1)
    all_dists, _ = map_kdtree.query(src_final, k=1)
    finite = np.isfinite(all_dists)
    mean_residual = float(np.mean(all_dists[finite])) if finite.any() else float('inf')

    if return_correspondences:
        return x, y, t, mean_residual, src_v_last, dst_v_last
    return x, y, t, mean_residual


def ekf_predict(state, P, v, w, dt, q_xy, q_yaw):
    """
    Pure-math EKF predict for unicycle [x, y, theta].
    Q is scaled by dt so q_xy / q_yaw are stddev per sqrt-second.
    Returns (new_state, new_P).
    """
    x, y, theta = float(state[0]), float(state[1]), float(state[2])
    new_x = x + v * math.cos(theta) * dt
    new_y = y + v * math.sin(theta) * dt
    new_theta = wrap_angle(theta + w * dt)

    F = np.array([
        [1.0, 0.0, -v * math.sin(theta) * dt],
        [0.0, 1.0,  v * math.cos(theta) * dt],
        [0.0, 0.0,  1.0],
    ])
    Q = np.diag([
        (q_xy ** 2) * dt,
        (q_xy ** 2) * dt,
        (q_yaw ** 2) * dt,
    ])
    new_P = F @ P @ F.T + Q
    return np.array([new_x, new_y, new_theta]), new_P


def ekf_update(state, P, z, r_xy, r_yaw):
    """
    Pure-math EKF update with measurement z = [x, y, theta], H = I.
    Yaw innovation is wrapped to (-pi, pi].
    Returns (new_state, new_P).
    """
    x_vec = np.array([float(state[0]), float(state[1]), float(state[2])])
    z_vec = np.array([float(z[0]), float(z[1]), wrap_angle(float(z[2]))])
    innov = z_vec - x_vec
    innov[2] = wrap_angle(innov[2])

    R = np.diag([r_xy ** 2, r_xy ** 2, r_yaw ** 2])
    S = P + R
    K = P @ np.linalg.inv(S)
    x_new = x_vec + K @ innov
    x_new[2] = wrap_angle(x_new[2])
    new_P = (np.eye(3) - K) @ P
    return x_new, new_P


def load_occupancy_pointcloud(yaml_path):
    """Parse <map>.yaml + <map>.pgm and return (N, 2) occupied points in map frame."""
    with open(yaml_path, 'r') as f:
        meta = yaml.safe_load(f)
    pgm_path = meta['image']
    if not os.path.isabs(pgm_path):
        pgm_path = os.path.join(os.path.dirname(yaml_path), pgm_path)
    res = float(meta['resolution'])
    ox, oy, _ = meta['origin']
    occ_thresh = float(meta.get('occupied_thresh', 0.65))
    negate = bool(meta.get('negate', 0))

    img = np.asarray(Image.open(pgm_path)).astype(np.float32) / 255.0
    if negate:
        prob = img
    else:
        prob = 1.0 - img
    occupied = prob > occ_thresh

    h, w = occupied.shape
    js, is_ = np.meshgrid(np.arange(w), np.arange(h))
    px = ox + js[occupied].astype(np.float64) * res
    # PGM rows are top-down; ROS map origin is bottom-left, so flip vertically.
    py = oy + (h - 1 - is_[occupied].astype(np.float64)) * res
    return np.stack([px, py], axis=1)
