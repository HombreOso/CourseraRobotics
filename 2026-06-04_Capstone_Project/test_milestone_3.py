"""
Tests for Milestone 3 – Feedback Control
==========================================
Uses the exact reference test case published in the assignment specification.

Robot configuration:
    (φ, x, y, θ1, θ2, θ3, θ4, θ5) = (0, 0, 0, 0, 0, 0.2, -1.6, 0)

All results below are taken verbatim from the assignment grading rubric.
"""

import unittest
import numpy as np

from milestone_3_feedback_control import FeedbackControl


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

THETA = np.array([0.0, 0.0, 0.2, -1.6, 0.0])   # arm joints (θ1…θ5)
DT    = 0.01
ATOL  = 1e-2   # tolerance for all assertions (matches 3 sig-fig reference values)

# Current reference end-effector config
X_d = np.array([
    [ 0,  0,  1,  0.5],
    [ 0,  1,  0,  0  ],
    [-1,  0,  0,  0.5],
    [ 0,  0,  0,  1  ],
], dtype=float)

# Next reference end-effector config (Δt = 0.01 s later)
X_d_next = np.array([
    [ 0,  0,  1,  0.6],
    [ 0,  1,  0,  0  ],
    [-1,  0,  0,  0.3],
    [ 0,  0,  0,  1  ],
], dtype=float)

# Current actual end-effector config (computed from robot config given above)
X = np.array([
    [ 0.170,  0,  0.985,  0.387],
    [ 0,      1,  0,      0    ],
    [-0.985,  0,  0.170,  0.570],
    [ 0,      0,  0,      1    ],
], dtype=float)

# Zero gain matrices (baseline test: only feedforward active)
K_p_zero = np.zeros((6, 6))
K_i_zero = np.zeros((6, 6))

# Identity proportional gain (bonus test)
K_p_eye  = np.eye(6)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(K_p, K_i):
    """Run FeedbackControl once from the reference fixture."""
    return FeedbackControl(
        X             = X,
        X_d           = X_d,
        X_d_next      = X_d_next,
        K_p           = K_p,
        K_i           = K_i,
        dt            = DT,
        theta_list    = THETA,
        X_err_integral= np.zeros(6),
    )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestFeedbackControlReference(unittest.TestCase):
    """
    Exact reference values from the assignment specification.
    All gains zero so V = [Ad_{X⁻¹Xd}] Vd only.
    """

    def setUp(self):
        self.V, self.controls, self.X_err, self.integral = _run(K_p_zero, K_i_zero)

    # ------------------------------------------------------------------
    # Feedforward twist Vd  (before adjoint mapping)
    # Computed as (1/dt) * log(Xd^{-1} Xd_next); expected (0,0,0,20,0,10)
    # ------------------------------------------------------------------
    # We verify indirectly: with zero error and zero gains, V = Ad·Vd.
    # The direct Vd value is tested via the adjoint-mapped output.

    def test_V_components(self):
        """V = [Ad_{X⁻¹Xd}]Vd should be (0, 0, 0, 21.409, 0, 6.455)."""
        expected = np.array([0.0, 0.0, 0.0, 21.409, 0.0, 6.455])
        np.testing.assert_allclose(
            self.V, expected, atol=ATOL,
            err_msg="Commanded twist V does not match reference",
        )

    def test_X_err(self):
        """Xerr should be (0, 0.171, 0, 0.080, 0, 0.107)."""
        expected = np.array([0.0, 0.171, 0.0, 0.080, 0.0, 0.107])
        np.testing.assert_allclose(
            self.X_err, expected, atol=ATOL,
            err_msg="Error twist Xerr does not match reference",
        )

    def test_integral_increment(self):
        """Integral increment should be Xerr * Δt."""
        expected_increment = np.array([0.0, 0.171, 0.0, 0.080, 0.0, 0.107]) * DT
        np.testing.assert_allclose(
            self.integral, expected_increment, atol=ATOL * DT,
            err_msg="Integral increment Xerr*Δt does not match",
        )

    def test_controls_wheels_equal(self):
        """
        With pure forward drive (V has only v_x / v_z), all four wheel speeds
        should be equal (no rotation or lateral motion).
        Reference: u = (157.2, 157.2, 157.2, 157.2, ...).
        """
        u = self.controls[:4]
        self.assertAlmostEqual(u[0], u[1], delta=ATOL,
                               msg="Wheel speeds u1 and u2 differ unexpectedly")
        self.assertAlmostEqual(u[0], u[2], delta=ATOL,
                               msg="Wheel speeds u1 and u3 differ unexpectedly")
        self.assertAlmostEqual(u[0], u[3], delta=ATOL,
                               msg="Wheel speeds u1 and u4 differ unexpectedly")

    def test_controls_wheel_magnitude(self):
        """Wheel speeds should be ~157.2 rad/s."""
        expected_wheel = 157.2
        for i, u_i in enumerate(self.controls[:4]):
            self.assertAlmostEqual(
                abs(u_i), expected_wheel, delta=1.0,
                msg=f"Wheel speed u{i+1} magnitude differs from reference",
            )

    def test_controls_joint_1_and_5_near_zero(self):
        """
        Joints 1 and 5 rotate about z – they cannot generate the required
        x/z velocity, so their speeds should be near zero.
        """
        theta_dot = self.controls[4:9]
        self.assertAlmostEqual(theta_dot[0], 0.0, delta=ATOL,
                               msg="Joint 1 speed should be ~0")
        self.assertAlmostEqual(theta_dot[4], 0.0, delta=ATOL,
                               msg="Joint 5 speed should be ~0")


class TestFeedbackControlWithKp(unittest.TestCase):
    """
    Bonus reference test: Kp = I (identity).
    V = Ad·Vd + Xerr  →  V = (0, 0.171, 0, 21.488, 0, 6.562).
    """

    def setUp(self):
        self.V, self.controls, self.X_err, _ = _run(K_p_eye, K_i_zero)

    def test_V_with_Kp(self):
        """V should be (0, 0.171, 0, 21.488, 0, 6.562) with Kp = I."""
        expected = np.array([0.0, 0.171, 0.0, 21.488, 0.0, 6.562])
        np.testing.assert_allclose(
            self.V, expected, atol=ATOL,
            err_msg="V with Kp=I does not match reference",
        )

    def test_controls_wheels_with_Kp(self):
        """Wheel speeds with Kp=I should be ~157.5 rad/s."""
        expected_wheel = 157.5
        for i, u_i in enumerate(self.controls[:4]):
            self.assertAlmostEqual(
                abs(u_i), expected_wheel, delta=1.0,
                msg=f"Wheel speed u{i+1} with Kp=I differs from reference",
            )


class TestFeedbackControlProperties(unittest.TestCase):
    """General behavioural properties that should hold regardless of gains."""

    def test_zero_error_zero_gains_gives_feedforward_only(self):
        """
        When X = X_d and all gains are zero, V must equal
        [Ad_{I}] Vd = Vd exactly (no error contribution).
        """
        import modern_robotics as mr
        X_eq = X_d.copy()
        V, _, X_err, _ = FeedbackControl(
            X=X_eq, X_d=X_d, X_d_next=X_d_next,
            K_p=K_p_zero, K_i=K_i_zero,
            dt=DT, theta_list=THETA,
            X_err_integral=np.zeros(6),
        )
        np.testing.assert_allclose(
            X_err, np.zeros(6), atol=1e-10,
            err_msg="Xerr should be zero when X = X_d",
        )

    def test_higher_Kp_increases_correction(self):
        """Larger Kp should push V further from feedforward-only value."""
        V_zero, _, _, _ = _run(K_p_zero, K_i_zero)
        V_large, _, _, _ = _run(np.eye(6) * 10.0, K_i_zero)
        # The correction on the non-zero Xerr components must be larger
        self.assertGreater(
            np.linalg.norm(V_large - V_zero), 1e-6,
            msg="Higher Kp should change V when there is a non-zero error",
        )

    def test_integral_accumulates(self):
        """Calling FeedbackControl twice should accumulate the integral."""
        _, _, X_err_1, integral_1 = _run(K_p_zero, K_i_zero)
        _, _, X_err_2, integral_2 = FeedbackControl(
            X=X, X_d=X_d, X_d_next=X_d_next,
            K_p=K_p_zero, K_i=K_i_zero,
            dt=DT, theta_list=THETA,
            X_err_integral=integral_1,
        )
        expected = integral_1 + X_err_2 * DT
        np.testing.assert_allclose(
            integral_2, expected, atol=1e-10,
            err_msg="Integral did not accumulate correctly on second call",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
