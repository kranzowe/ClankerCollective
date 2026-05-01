"""Unit tests for the SE(2) helpers and angle wrapping in core.py."""
import math

import numpy as np
import pytest

from og_estimation_localization.core import (se2_from_xyt, transform_points,
                                          wrap_angle, xyt_from_se2)


class TestWrapAngle:
    def test_zero_unchanged(self):
        assert wrap_angle(0.0) == 0.0

    def test_already_in_range(self):
        assert wrap_angle(1.5) == pytest.approx(1.5)
        assert wrap_angle(-1.5) == pytest.approx(-1.5)

    def test_wraps_above_pi(self):
        assert wrap_angle(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)

    def test_wraps_below_neg_pi(self):
        assert wrap_angle(-math.pi - 0.1) == pytest.approx(math.pi - 0.1)

    def test_two_pi_wraps_to_zero(self):
        assert wrap_angle(2.0 * math.pi) == pytest.approx(0.0, abs=1e-9)

    def test_three_pi_wraps_correctly(self):
        # 3*pi == pi + 2*pi == pi
        assert abs(wrap_angle(3.0 * math.pi)) == pytest.approx(math.pi, abs=1e-9)


class TestSE2RoundTrip:
    @pytest.mark.parametrize("xyt", [
        (0.0, 0.0, 0.0),
        (1.0, 2.0, 0.5),
        (-3.0, 4.5, -1.2),
        (10.0, -7.0, math.pi / 3),
    ])
    def test_xyt_to_T_and_back(self, xyt):
        T = se2_from_xyt(*xyt)
        x, y, t = xyt_from_se2(T)
        assert (x, y, t) == pytest.approx(xyt, abs=1e-9)

    def test_se2_is_3x3_homogeneous(self):
        T = se2_from_xyt(1.0, 2.0, 0.5)
        assert T.shape == (3, 3)
        np.testing.assert_array_almost_equal(T[2], [0.0, 0.0, 1.0])

    def test_rotation_block_is_orthogonal(self):
        T = se2_from_xyt(0.0, 0.0, 0.7)
        R = T[:2, :2]
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(2))


class TestSE2Compose:
    def test_compose_translations(self):
        # Translate by (1, 0), then by (0, 1). Composed: (1, 1).
        T1 = se2_from_xyt(1.0, 0.0, 0.0)
        T2 = se2_from_xyt(0.0, 1.0, 0.0)
        T = T2 @ T1
        x, y, t = xyt_from_se2(T)
        assert (x, y, t) == pytest.approx((1.0, 1.0, 0.0), abs=1e-9)

    def test_compose_rotations(self):
        T1 = se2_from_xyt(0.0, 0.0, math.pi / 4)
        T2 = se2_from_xyt(0.0, 0.0, math.pi / 4)
        T = T2 @ T1
        _, _, t = xyt_from_se2(T)
        assert t == pytest.approx(math.pi / 2, abs=1e-9)

    def test_inverse_is_identity(self):
        T = se2_from_xyt(2.0, -1.0, 0.7)
        I = T @ np.linalg.inv(T)
        np.testing.assert_array_almost_equal(I, np.eye(3))


class TestTransformPoints:
    def test_identity_unchanged(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = transform_points(pts, 0.0, 0.0, 0.0)
        np.testing.assert_array_almost_equal(out, pts)

    def test_pure_translation(self):
        pts = np.array([[1.0, 2.0], [-1.0, 0.0]])
        out = transform_points(pts, 5.0, -3.0, 0.0)
        np.testing.assert_array_almost_equal(out, [[6.0, -1.0], [4.0, -3.0]])

    def test_pure_rotation_90deg(self):
        pts = np.array([[1.0, 0.0]])
        out = transform_points(pts, 0.0, 0.0, math.pi / 2)
        np.testing.assert_array_almost_equal(out, [[0.0, 1.0]])

    def test_translation_then_rotation_is_consistent(self):
        # Verify transform_points matches the equivalent SE(2) matrix multiply.
        pts = np.array([[2.0, 3.0], [-1.0, 1.0], [0.0, -2.0]])
        x, y, t = 0.5, -0.7, 1.1
        T = se2_from_xyt(x, y, t)
        homog = np.hstack([pts, np.ones((pts.shape[0], 1))])
        expected = (T @ homog.T).T[:, :2]
        out = transform_points(pts, x, y, t)
        np.testing.assert_array_almost_equal(out, expected)
