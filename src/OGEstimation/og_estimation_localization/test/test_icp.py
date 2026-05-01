"""Unit tests for the ICP scan-matcher in core.py.

Strategy: build a synthetic "map" of points (a 4-wall room), simulate a scan by
transforming a subset of those points into the robot frame given a known true
pose, then perturb the initial guess and assert ICP recovers the true pose.
"""
import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

from og_estimation_localization.core import icp_se2, transform_points


# ---------------------------------------------------------------------------
# Fixtures: synthetic environments
# ---------------------------------------------------------------------------

def make_room(side=4.0, density=2000):
    """
    Four walls of a square room, centered at origin.

    High point density (~2 mm spacing at default settings) keeps the discrete
    nearest-neighbor matching from biasing ICP convergence — much finer than
    the 5 cm cells of the real lab map. Real-world ICP-on-grid will be less
    accurate than these tests, so the unit tests only assert that the algorithm
    converges to the right neighborhood.
    """
    half = side / 2.0
    s = np.linspace(-half, half, density)
    walls = np.concatenate([
        np.stack([s, np.full_like(s, -half)], axis=1),
        np.stack([s, np.full_like(s, +half)], axis=1),
        np.stack([np.full_like(s, -half), s], axis=1),
        np.stack([np.full_like(s, +half), s], axis=1),
    ])
    return walls


def simulate_scan(map_pts, true_xyt, max_range=10.0):
    """
    Convert a set of map points into a scan from the robot's frame.

    For test purposes we don't ray-cast — every map point in range becomes a
    "scan return". That's enough to test alignment.
    """
    x, y, t = true_xyt
    # World -> robot: inverse of forward SE(2).
    cosm, sinm = math.cos(-t), math.sin(-t)
    dx = map_pts[:, 0] - x
    dy = map_pts[:, 1] - y
    rx = cosm * dx - sinm * dy
    ry = sinm * dx + cosm * dy
    r = np.hypot(rx, ry)
    keep = r < max_range
    return np.stack([rx[keep], ry[keep]], axis=1)


@pytest.fixture
def room_kdtree():
    return cKDTree(make_room())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestICPIdentity:
    def test_perfect_match_zero_residual(self, room_kdtree):
        true_xyt = (0.0, 0.0, 0.0)
        scan = simulate_scan(room_kdtree.data, true_xyt)
        x, y, t, residual = icp_se2(scan, room_kdtree, true_xyt)
        assert residual < 1e-6
        assert (x, y, t) == pytest.approx(true_xyt, abs=1e-6)


class TestICPPureTranslation:
    @pytest.mark.parametrize("offset_xy", [
        (0.05, 0.0),
        (-0.10, 0.0),
        (0.0, 0.15),
        (0.10, -0.10),
    ])
    def test_recovers_translation(self, room_kdtree, offset_xy):
        true_xyt = (0.3, -0.2, 0.0)
        scan = simulate_scan(room_kdtree.data, true_xyt)
        init = (true_xyt[0] + offset_xy[0], true_xyt[1] + offset_xy[1], true_xyt[2])
        x, y, t, residual = icp_se2(scan, room_kdtree, init, max_iters=30)
        assert (x, y, t) == pytest.approx(true_xyt, abs=3e-2)
        assert residual < 0.05


class TestICPPureRotation:
    @pytest.mark.parametrize("offset_deg", [-10.0, -5.0, 5.0, 10.0])
    def test_recovers_rotation(self, room_kdtree, offset_deg):
        true_xyt = (0.0, 0.0, math.radians(20.0))
        scan = simulate_scan(room_kdtree.data, true_xyt)
        init = (true_xyt[0], true_xyt[1], true_xyt[2] + math.radians(offset_deg))
        x, y, t, residual = icp_se2(scan, room_kdtree, init, max_iters=30)
        # Point-to-point ICP on a discrete-sample wall map has ~1 degree
        # discretization bias in the recovered rotation; that's fine for our use.
        assert math.degrees(abs(t - true_xyt[2])) < 1.5
        assert (x, y) == pytest.approx(true_xyt[:2], abs=3e-2)


class TestICPCombined:
    def test_recovers_combined_offset(self, room_kdtree):
        true_xyt = (0.5, -0.3, math.radians(15.0))
        scan = simulate_scan(room_kdtree.data, true_xyt)
        init = (true_xyt[0] + 0.08, true_xyt[1] - 0.05, true_xyt[2] + math.radians(4.0))
        x, y, t, residual = icp_se2(scan, room_kdtree, init, max_iters=30)
        assert (x, y) == pytest.approx(true_xyt[:2], abs=4e-2)
        assert math.degrees(abs(t - true_xyt[2])) < 1.5
        assert residual < 0.05


class TestICPConvergesFromLargerOffset:
    def test_quarter_meter_offset(self, room_kdtree):
        true_xyt = (0.0, 0.0, 0.0)
        scan = simulate_scan(room_kdtree.data, true_xyt)
        init = (0.25, 0.0, 0.0)
        # Need a generous correspondence cap and more iters from this far out.
        x, y, t, residual = icp_se2(
            scan, room_kdtree, init, max_iters=40, max_corr_dist=1.0)
        assert (x, y, t) == pytest.approx(true_xyt, abs=4e-2)


class TestICPCorrespondences:
    def test_returns_pairs(self, room_kdtree):
        true_xyt = (0.0, 0.0, 0.0)
        scan = simulate_scan(room_kdtree.data, true_xyt)
        x, y, t, residual, src, dst = icp_se2(
            scan, room_kdtree, true_xyt, return_correspondences=True)
        assert src.shape[0] > 0
        assert src.shape == dst.shape
        assert src.shape[1] == 2


class TestICPRobustness:
    def test_sparse_scan_does_not_crash(self, room_kdtree):
        # Fewer than 20 valid points -> ICP should bail out cleanly.
        scan = np.array([[0.5, 0.0], [0.6, 0.1]])
        x, y, t, residual = icp_se2(scan, room_kdtree, (0.0, 0.0, 0.0))
        assert math.isfinite(x) and math.isfinite(y) and math.isfinite(t)

    def test_residual_high_on_garbage_scan(self, room_kdtree):
        # All points far outside the room. ICP refuses to update (no
        # correspondences within max_corr_dist), but the final residual is
        # computed unbounded so it's always finite — and large, telling the
        # caller the scan is far from the map.
        garbage = np.random.RandomState(0).uniform(50.0, 60.0, size=(200, 2))
        x, y, t, residual = icp_se2(
            garbage, room_kdtree, (0.0, 0.0, 0.0), max_corr_dist=0.5)
        assert math.isfinite(residual)
        assert residual > 0.5

    def test_idempotent_when_already_aligned(self, room_kdtree):
        # Run ICP twice from the same starting point — second run should be a no-op.
        true_xyt = (0.0, 0.0, 0.0)
        scan = simulate_scan(room_kdtree.data, true_xyt)
        x1, y1, t1, _ = icp_se2(scan, room_kdtree, true_xyt)
        x2, y2, t2, _ = icp_se2(scan, room_kdtree, (x1, y1, t1))
        assert (x2, y2, t2) == pytest.approx((x1, y1, t1), abs=1e-6)
