"""
Tests for Milestone 1 – youBot Kinematic Simulator
====================================================
Verifies the three reference control scenarios from the assignment:

  1. u = ( 10,  10,  10,  10)  →  forward  +0.475 m in x_b
  2. u = (-10,  10, -10,  10)  →  sideways +0.475 m in y_b
  3. u = (-10,  10,  10, -10)  →  CCW spin  +1.234 rad in φ

Each scenario runs for 1 s (100 steps × dt = 0.01 s) from the zero
configuration, with arm joints held at zero.
"""

import unittest
import numpy as np

from milestone_1_youBot_kinematic_simulator import simulate, NextState


# Tolerance used for all assertions (metres / radians)
ATOL = 1e-3

# Shared simulation parameters
DT        = 0.01
N_STEPS   = 100          # 1 second
MAX_SPEED = 12.3         # rad/s
ARM_ZERO  = np.zeros(5)  # arm joint speeds held at zero


def _run(wheel_speeds: list[float]) -> np.ndarray:
    """Run 1 s simulation from all-zero config, return final state."""
    u = np.concatenate([wheel_speeds, ARM_ZERO])   # 9-vector
    controls = np.tile(u, (N_STEPS, 1))
    traj = simulate(np.zeros(13), controls, DT, MAX_SPEED)
    return traj[-1]


class TestNextStateSanity(unittest.TestCase):
    """Basic sanity checks on NextState."""

    def test_zero_controls_no_change(self):
        """Zero controls must leave the config unchanged."""
        q0  = np.array([0.1, 0.5, -0.3, 0.0, 0.1, -0.2, 0.3, 0.0,
                        1.0, 2.0, 3.0, 4.0, 0.0])
        out = NextState(q0, np.zeros(9), dt=0.01, max_speed=12.3)
        np.testing.assert_allclose(out, q0, atol=1e-12,
                                   err_msg="Zero controls changed the config")

    def test_speed_clamping(self):
        """Controls beyond max_speed must be clamped, not silently ignored."""
        q0  = np.zeros(13)
        # Unclamped: 100 rad/s forward for 0.01 s would give x = r*100*0.01 = 0.0475
        # Clamped to 10: x = r*10*0.01 = 0.00475
        u_over  = np.array([100.0, 100.0, 100.0, 100.0, 0, 0, 0, 0, 0])
        u_clamp = np.array([ 10.0,  10.0,  10.0,  10.0, 0, 0, 0, 0, 0])
        out_over  = NextState(q0, u_over,  dt=0.01, max_speed=10.0)
        out_clamp = NextState(q0, u_clamp, dt=0.01, max_speed=10.0)
        np.testing.assert_allclose(out_over, out_clamp, atol=1e-12,
                                   err_msg="Clamping did not work correctly")

    def test_gripper_unchanged(self):
        """Gripper state must pass through NextState unchanged."""
        for grip in (0.0, 1.0):
            q0      = np.zeros(13); q0[12] = grip
            out     = NextState(q0, np.zeros(9), dt=0.01, max_speed=12.3)
            self.assertEqual(out[12], grip,
                             msg=f"Gripper state {grip} was modified by NextState")


class TestReferenceControls(unittest.TestCase):
    """
    Three reference scenarios from the assignment specification.
    All run for 1 s (100 × 0.01 s) from the all-zero configuration.
    """

    # ------------------------------------------------------------------
    # 1. Forward drive
    # ------------------------------------------------------------------
    def test_forward_drive_x_displacement(self):
        """u=(10,10,10,10): x must reach +0.475 m."""
        final = _run([10.0, 10.0, 10.0, 10.0])
        self.assertAlmostEqual(final[1], 0.475, delta=ATOL,
                               msg="Forward drive x displacement incorrect")

    def test_forward_drive_y_zero(self):
        """u=(10,10,10,10): y must remain 0."""
        final = _run([10.0, 10.0, 10.0, 10.0])
        self.assertAlmostEqual(final[2], 0.0, delta=ATOL,
                               msg="Forward drive produced unexpected y drift")

    def test_forward_drive_phi_zero(self):
        """u=(10,10,10,10): φ must remain 0."""
        final = _run([10.0, 10.0, 10.0, 10.0])
        self.assertAlmostEqual(final[0], 0.0, delta=ATOL,
                               msg="Forward drive produced unexpected rotation")

    # ------------------------------------------------------------------
    # 2. Sideways slide
    # ------------------------------------------------------------------
    def test_sideways_y_displacement(self):
        """u=(-10,10,-10,10): y must reach +0.475 m."""
        final = _run([-10.0, 10.0, -10.0, 10.0])
        self.assertAlmostEqual(final[2], 0.475, delta=ATOL,
                               msg="Sideways slide y displacement incorrect")

    def test_sideways_x_zero(self):
        """u=(-10,10,-10,10): x must remain 0."""
        final = _run([-10.0, 10.0, -10.0, 10.0])
        self.assertAlmostEqual(final[1], 0.0, delta=ATOL,
                               msg="Sideways slide produced unexpected x drift")

    def test_sideways_phi_zero(self):
        """u=(-10,10,-10,10): φ must remain 0."""
        final = _run([-10.0, 10.0, -10.0, 10.0])
        self.assertAlmostEqual(final[0], 0.0, delta=ATOL,
                               msg="Sideways slide produced unexpected rotation")

    # ------------------------------------------------------------------
    # 3. CCW spin in place
    # ------------------------------------------------------------------
    def test_spin_phi(self):
        """u=(-10,10,10,-10): φ must reach +1.234 rad."""
        final = _run([-10.0, 10.0, 10.0, -10.0])
        self.assertAlmostEqual(final[0], 1.234, delta=ATOL,
                               msg="CCW spin angle incorrect")

    def test_spin_x_zero(self):
        """u=(-10,10,10,-10): x must remain 0 (spin in place)."""
        final = _run([-10.0, 10.0, 10.0, -10.0])
        self.assertAlmostEqual(final[1], 0.0, delta=ATOL,
                               msg="CCW spin produced unexpected x displacement")

    def test_spin_y_zero(self):
        """u=(-10,10,10,-10): y must remain 0 (spin in place)."""
        final = _run([-10.0, 10.0, 10.0, -10.0])
        self.assertAlmostEqual(final[2], 0.0, delta=ATOL,
                               msg="CCW spin produced unexpected y displacement")


if __name__ == "__main__":
    unittest.main(verbosity=2)
