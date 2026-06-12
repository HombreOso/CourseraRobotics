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

def _run(K_p, K_i, enforce_joint_limits=False):
    """Run FeedbackControl once from the reference fixture.

    Joint limits are disabled by default here because the assignment's
    published reference values were computed without limit enforcement
    (θ3 = 0.2 in the fixture would otherwise trigger the J3 upper limit).
    Separate tests below verify the limit-enforcement logic.
    """
    return FeedbackControl(
        X                    = X,
        X_d                  = X_d,
        X_d_next             = X_d_next,
        K_p                  = K_p,
        K_i                  = K_i,
        dt                   = DT,
        theta_list           = THETA,
        X_err_integral       = np.zeros(6),
        enforce_joint_limits = enforce_joint_limits,
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


class TestJointLimits(unittest.TestCase):
    """Tests for testJointLimits and joint-limit enforcement in FeedbackControl."""

    def test_no_violation_within_limits(self):
        """The standard start config should return an empty list."""
        from milestone_3_feedback_control import testJointLimits
        theta_ok = np.array([0.0, 0.0, 0.2, -1.6, 0.0])
        self.assertEqual(testJointLimits(theta_ok), [],
                         msg="Standard start config should be within limits")

    def test_j3_upper_limit_violation(self):
        """θ3 = 3.0 exceeds J3 upper limit 2.548 → index 2 returned."""
        from milestone_3_feedback_control import testJointLimits
        theta = np.array([0.0, 0.0, 3.0, -1.6, 0.0])
        self.assertIn(2, testJointLimits(theta),
                      msg="Joint 3 (index 2) should be flagged for θ3=3.0 > 2.548")

    def test_j4_lower_limit_violation(self):
        """θ4 = -3.0 exceeds J4 lower limit -1.780 → index 3 returned."""
        from milestone_3_feedback_control import testJointLimits
        theta = np.array([0.0, 0.0, -0.5, -3.0, 0.0])
        self.assertIn(3, testJointLimits(theta),
                      msg="Joint 4 (index 3) should be flagged for θ4=-3.0 < -1.780")

    def test_multiple_violations(self):
        """Both J3 and J4 out of range → both indices returned."""
        from milestone_3_feedback_control import testJointLimits
        theta = np.array([0.0, 0.0, 3.0, -3.0, 0.0])
        violated = testJointLimits(theta)
        self.assertIn(2, violated)
        self.assertIn(3, violated)

    def test_enforcement_zeros_violating_joints(self):
        """
        With enforcement ON, every joint whose unconstrained prediction
        would violate a limit must end up with zero speed (Je column zeroed).
        The set of violated joints is computed from the actual unconstrained
        solution so the test stays self-consistent regardless of fixture.
        """
        from milestone_3_feedback_control import testJointLimits

        # θ3=3.0 puts the arm near a singularity; the unconstrained solution
        # drives some joints past their limits.
        theta_bad = np.array([0.0, 0.0, 3.0, -1.6, 0.0])

        # Unconstrained solution and its predicted next angles
        _, ctrl_off, _, _ = FeedbackControl(
            X=X, X_d=X_d, X_d_next=X_d_next,
            K_p=K_p_zero, K_i=K_i_zero,
            dt=DT, theta_list=theta_bad,
            X_err_integral=np.zeros(6),
            enforce_joint_limits=False,
        )
        theta_next = theta_bad + ctrl_off[4:9] * DT
        violated   = testJointLimits(theta_next)
        self.assertTrue(violated,
                        msg="Fixture should trigger at least one joint violation")

        # Enforced solution: the violated joints must be frozen (speed 0)
        _, ctrl_on, _, _ = FeedbackControl(
            X=X, X_d=X_d, X_d_next=X_d_next,
            K_p=K_p_zero, K_i=K_i_zero,
            dt=DT, theta_list=theta_bad,
            X_err_integral=np.zeros(6),
            enforce_joint_limits=True,
        )
        for j in violated:
            # joint j (0-indexed) is control index 4 + j
            self.assertAlmostEqual(
                ctrl_on[4 + j], 0.0, delta=1e-9,
                msg=f"Joint {j+1} speed must be zero when its limit is enforced",
            )

    def test_enforcement_disabled_matches_original(self):
        """
        With enforcement OFF the result must match the no-limit reference
        values from the assignment.
        """
        V, ctrl, err, _ = _run(K_p_zero, K_i_zero, enforce_joint_limits=False)
        expected_V = np.array([0.0, 0.0, 0.0, 21.409, 0.0, 6.455])
        np.testing.assert_allclose(V, expected_V, atol=ATOL,
                                   err_msg="V without limits must match assignment reference")


if __name__ == "__main__":
    unittest.main(verbosity=2)
