"""Unit tests for the EKF predict/update functions in core.py."""
import math

import numpy as np
import pytest

from og_estimation_localization.core import ekf_predict, ekf_update, wrap_angle


# ---------------------------------------------------------------------------
# Predict step
# ---------------------------------------------------------------------------

class TestEKFPredict:
    def test_zero_motion_state_unchanged(self):
        state = np.array([1.0, 2.0, 0.5])
        P = np.eye(3) * 0.1
        new_state, new_P = ekf_predict(state, P, v=0.0, w=0.0, dt=0.1,
                                       q_xy=0.05, q_yaw=0.03)
        np.testing.assert_array_almost_equal(new_state, state)

    def test_zero_motion_covariance_grows(self):
        state = np.zeros(3)
        P = np.eye(3) * 0.01
        _, new_P = ekf_predict(state, P, v=0.0, w=0.0, dt=0.1,
                               q_xy=0.05, q_yaw=0.03)
        # F = I in this case, so new_P = P + Q. Diagonal should grow.
        assert new_P[0, 0] > P[0, 0]
        assert new_P[1, 1] > P[1, 1]
        assert new_P[2, 2] > P[2, 2]

    def test_straight_line_along_x(self):
        state = np.zeros(3)
        P = np.eye(3) * 0.01
        new_state, _ = ekf_predict(state, P, v=1.0, w=0.0, dt=1.0,
                                   q_xy=0.05, q_yaw=0.03)
        assert new_state[0] == pytest.approx(1.0)
        assert new_state[1] == pytest.approx(0.0)
        assert new_state[2] == pytest.approx(0.0)

    def test_straight_line_at_45deg(self):
        state = np.array([0.0, 0.0, math.pi / 4])
        P = np.eye(3) * 0.01
        new_state, _ = ekf_predict(state, P, v=math.sqrt(2.0), w=0.0, dt=1.0,
                                   q_xy=0.05, q_yaw=0.03)
        # Should move (1, 1).
        assert new_state[0] == pytest.approx(1.0, abs=1e-9)
        assert new_state[1] == pytest.approx(1.0, abs=1e-9)
        assert new_state[2] == pytest.approx(math.pi / 4)

    def test_pure_rotation(self):
        state = np.zeros(3)
        P = np.eye(3) * 0.01
        new_state, _ = ekf_predict(state, P, v=0.0, w=1.0, dt=0.5,
                                   q_xy=0.05, q_yaw=0.03)
        assert new_state[0] == pytest.approx(0.0)
        assert new_state[1] == pytest.approx(0.0)
        assert new_state[2] == pytest.approx(0.5)

    def test_yaw_wraps_to_pi_range(self):
        state = np.array([0.0, 0.0, math.pi - 0.05])
        new_state, _ = ekf_predict(state, np.eye(3), v=0.0, w=1.0, dt=0.2,
                                   q_xy=0.05, q_yaw=0.03)
        # Without wrapping this would be ~ pi + 0.15.
        assert -math.pi <= new_state[2] <= math.pi
        assert new_state[2] == pytest.approx(-math.pi + 0.15, abs=1e-9)

    def test_covariance_remains_symmetric(self):
        rng = np.random.RandomState(0)
        L = rng.randn(3, 3) * 0.1
        P = L @ L.T + np.eye(3) * 0.01
        state = np.array([0.5, -0.3, 0.7])
        _, new_P = ekf_predict(state, P, v=1.5, w=0.3, dt=0.1,
                               q_xy=0.05, q_yaw=0.03)
        np.testing.assert_array_almost_equal(new_P, new_P.T)

    def test_q_scales_with_dt(self):
        state = np.zeros(3)
        P = np.zeros((3, 3))  # ignore propagation, isolate Q.
        _, P_short = ekf_predict(state, P, 0.0, 0.0, dt=0.1, q_xy=1.0, q_yaw=1.0)
        _, P_long = ekf_predict(state, P, 0.0, 0.0, dt=1.0, q_xy=1.0, q_yaw=1.0)
        assert P_long[0, 0] == pytest.approx(P_short[0, 0] * 10.0)


# ---------------------------------------------------------------------------
# Update step
# ---------------------------------------------------------------------------

class TestEKFUpdate:
    def test_measurement_equals_state_no_change(self):
        state = np.array([1.0, 2.0, 0.5])
        P = np.eye(3) * 0.1
        new_state, new_P = ekf_update(state, P, z=state.copy(), r_xy=0.05, r_yaw=0.03)
        np.testing.assert_array_almost_equal(new_state, state)
        # Even with zero innovation, covariance should shrink.
        assert new_P[0, 0] < P[0, 0]
        assert new_P[1, 1] < P[1, 1]
        assert new_P[2, 2] < P[2, 2]

    def test_pulls_toward_measurement(self):
        state = np.array([0.0, 0.0, 0.0])
        P = np.eye(3) * 1.0          # high prior uncertainty
        z = np.array([1.0, -1.0, 0.5])
        new_state, _ = ekf_update(state, P, z, r_xy=0.05, r_yaw=0.05)
        # Update should move state most of the way toward z (since R << P).
        assert abs(new_state[0] - z[0]) < abs(state[0] - z[0])
        assert abs(new_state[1] - z[1]) < abs(state[1] - z[1])
        assert abs(new_state[2] - z[2]) < abs(state[2] - z[2])

    def test_low_r_trusts_measurement(self):
        state = np.array([0.0, 0.0, 0.0])
        P = np.eye(3) * 1.0
        z = np.array([1.0, 0.0, 0.0])
        # Very small R -> Kalman gain ~= 1 -> state should snap to z.
        new_state, _ = ekf_update(state, P, z, r_xy=1e-4, r_yaw=1e-4)
        assert new_state[0] == pytest.approx(1.0, abs=1e-3)

    def test_high_r_ignores_measurement(self):
        state = np.array([0.0, 0.0, 0.0])
        P = np.eye(3) * 0.01
        z = np.array([5.0, 5.0, 1.0])
        # Very large R -> Kalman gain ~= 0 -> state barely moves.
        new_state, _ = ekf_update(state, P, z, r_xy=10.0, r_yaw=10.0)
        assert abs(new_state[0]) < 0.05
        assert abs(new_state[1]) < 0.05

    def test_yaw_innovation_wraps(self):
        # State near -pi, measurement near +pi. Innovation should be tiny, not 2*pi.
        state = np.array([0.0, 0.0, -math.pi + 0.05])
        z = np.array([0.0, 0.0, math.pi - 0.05])
        P = np.eye(3) * 0.1
        new_state, _ = ekf_update(state, P, z, r_xy=0.05, r_yaw=0.05)
        # New yaw should remain near +/- pi (i.e., a tiny correction), not jump to 0.
        assert abs(abs(new_state[2]) - math.pi) < 0.5

    def test_covariance_remains_symmetric(self):
        rng = np.random.RandomState(1)
        L = rng.randn(3, 3) * 0.1
        P = L @ L.T + np.eye(3) * 0.01
        state = np.array([0.0, 0.0, 0.0])
        z = np.array([0.2, -0.1, 0.05])
        _, new_P = ekf_update(state, P, z, r_xy=0.05, r_yaw=0.03)
        np.testing.assert_array_almost_equal(new_P, new_P.T)

    def test_covariance_strictly_decreases_diagonal(self):
        state = np.array([0.5, -0.5, 0.1])
        P = np.eye(3) * 0.5
        z = np.array([0.5, -0.5, 0.1])  # zero innovation
        _, new_P = ekf_update(state, P, z, r_xy=0.05, r_yaw=0.05)
        for i in range(3):
            assert new_P[i, i] < P[i, i]


# ---------------------------------------------------------------------------
# Integrated predict-then-update
# ---------------------------------------------------------------------------

class TestPredictUpdateIntegration:
    def test_prediction_drift_corrected_by_measurement(self):
        # Simulate a robot that thinks it's at (0,0,0) but actually moved 1m forward.
        # Provide a measurement at (1,0,0) and verify EKF converges there.
        state = np.array([0.0, 0.0, 0.0])
        P = np.eye(3) * 0.05
        # Predict 10 small steps (no command, so prediction stays at origin).
        for _ in range(10):
            state, P = ekf_predict(state, P, v=0.0, w=0.0, dt=0.05,
                                   q_xy=0.10, q_yaw=0.05)
        # Now correct with a scan-match measurement at (1,0,0).
        state, P = ekf_update(state, P, z=np.array([1.0, 0.0, 0.0]),
                              r_xy=0.02, r_yaw=0.02)
        # After enough updates, state should converge.
        for _ in range(5):
            state, P = ekf_update(state, P, z=np.array([1.0, 0.0, 0.0]),
                                  r_xy=0.02, r_yaw=0.02)
        assert state[0] == pytest.approx(1.0, abs=1e-2)

    def test_filter_tracks_constant_velocity(self):
        # Robot drives in a straight line; EKF predicts at v=1 m/s and gets
        # noisy measurements at the same rate. After 20 steps, error should be small.
        rng = np.random.RandomState(42)
        state = np.array([0.0, 0.0, 0.0])
        P = np.eye(3) * 0.1
        true_x = 0.0
        for _ in range(20):
            true_x += 1.0 * 0.1
            state, P = ekf_predict(state, P, v=1.0, w=0.0, dt=0.1,
                                   q_xy=0.05, q_yaw=0.03)
            z = np.array([true_x + rng.normal(0, 0.02), rng.normal(0, 0.02),
                          rng.normal(0, 0.01)])
            state, P = ekf_update(state, P, z, r_xy=0.02, r_yaw=0.01)
        assert state[0] == pytest.approx(true_x, abs=0.05)
        assert abs(state[1]) < 0.05
        assert abs(state[2]) < 0.05
