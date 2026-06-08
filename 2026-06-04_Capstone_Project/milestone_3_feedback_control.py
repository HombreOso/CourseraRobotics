"""
Milestone 3 – Feedforward + PI Feedback Control
=================================================

PURPOSE — what this module does and why
-----------------------------------------
The reference trajectory (Milestone 2) is a pre-planned sequence of
end-effector poses T_se(t).  It describes WHERE the gripper should be at
every instant, but it says nothing about HOW to actuate the wheels and arm
joints to get there.

This module closes the loop:

    ┌──────────────────────────────────────────────────────────────────┐
    │  Reference trajectory                                            │
    │  T_se_d(t), T_se_d_next(t)   ───────────────────────────────►  │
    │                                    FeedbackControl              │
    │  Current actual pose              (this module)                 │
    │  T_se  ──────────────────────────────────────────────────────►  │
    │                                         │                       │
    │                                         ▼                       │
    │                                   controls (9)                  │
    │                              [u1,u2,u3,u4, θ̇1…θ̇5]             │
    │                                         │                       │
    │                                         ▼                       │
    │                                    NextState                    │
    │                               (Milestone 1 simulator)           │
    │                                         │                       │
    │                                         ▼                       │
    │                              new robot state (13)               │
    │                         [φ,x,y, J1…J5, W1…W4, grip] ──────────►(loop)
    └──────────────────────────────────────────────────────────────────┘

IS THE ROBOT MOVED BY FEEDFORWARD OR FEEDBACK?
-----------------------------------------------
Both terms together move the robot — they are added into a single
commanded twist V, which then drives the actuators:

  FEEDFORWARD TERM  [Ad_{X⁻¹Xd}] Vd
  ─────────────────────────────────
  Vd = (1/Δt) log(Xd⁻¹ Xd_next)  is the instantaneous screw velocity
  that the reference trajectory prescribes.  If the robot were perfect
  (exact model, no slip, no integration error), this term alone would
  keep the end-effector on the planned path forever.
  The Adjoint mapping re-expresses Vd from the reference frame {Xd}
  into the actual current frame {X}, because the Jacobian maps speeds
  into velocities expressed in {e} = {X}, not in {Xd}.

  FEEDBACK TERM  Kp·Xerr + Ki·∫Xerr dt
  ──────────────────────────────────────
  Xerr = log(X⁻¹ Xd) is the "gap" twist that, if applied for one unit
  of time, would take the actual pose X exactly to the reference Xd.
  The PI feedback drives this gap to zero.
  Without this term the robot would follow the path approximately but
  errors would grow unboundedly over time.

HOW DO ERRORS ARISE?
---------------------
The Euler integration in NextState introduces a small error every step:

    new_q  ≈  q + Vb·Δt        (first-order; exact only as Δt → 0)

Over hundreds of steps these accumulate.  Additional sources:

  • The mecanum wheel model assumes pure rolling — any slip is unmodelled.
  • The Jacobian pseudoinverse is only the minimum-norm first-order
    approximation; it does not account for nonlinearity between steps.
  • Near arm singularities the Jacobian loses rank and the pseudoinverse
    amplifies noise into very large joint speeds, causing overshoot.
  • Floating-point rounding in the matrix exponential/logarithm.

CONTROL LAW (Modern Robotics Eq. 11.16 / 13.37)
-------------------------------------------------
    V(t) = [Ad_{X⁻¹ Xd}] Vd(t)  +  Kp · Xerr(t)  +  Ki · ∫Xerr dt

where
    X        = T_se            current actual end-effector pose in {s}
    Xd       = T_se_d          current reference pose in {s}
    Xd_next  = T_se_d_next     reference pose one Δt later in {s}
    Xerr     = se3ToVec(log(X⁻¹ Xd))           6-vector error twist
    Vd       = (1/Δt) se3ToVec(log(Xd⁻¹ Xd_next))  feedforward twist
    V        = commanded end-effector twist expressed in {e}

CONVERTING THE TWIST TO ACTUATOR SPEEDS
-----------------------------------------
    [u; θ̇] = Je(θ)⁺ · V          (9-vector, pseudoinverse solution)

Je (6×9) is the mobile-manipulator Jacobian:

    Je = [ J_base (6×4)  |  J_arm (6×5) ]

    J_base = Ad_{T_0e⁻¹ T_b0⁻¹} · F6
      Expresses the chassis velocity contribution in the end-effector
      frame.  F6 maps 4 wheel speeds to a 6D body twist of the chassis;
      the Adjoint then re-expresses that twist in {e}.

    J_arm = JacobianBody(Blist, θ)
      Body Jacobian of the 5-DOF arm: maps joint speeds θ̇ to a 6D
      end-effector twist in {e}.

The pseudoinverse Je⁺ gives the minimum-norm solution for (u, θ̇) that
produces the desired V.  This means wheel and joint speeds are shared
optimally — the arm is used where it has more mechanical advantage, and
the base supplements where the arm would be near-singular.
"""

import csv
import logging
import numpy as np
import modern_robotics as mr
from datetime import datetime
from pathlib import Path

from configurations import T_b0, M_0e, Blist, F6 as _F6


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _build_logger(name: str = "milestone_3") -> logging.Logger:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path  = Path(__file__).parent / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Log file: %s", log_path)
    return logger


logger = _build_logger()


# ---------------------------------------------------------------------------
# Mobile-manipulator Jacobian
# ---------------------------------------------------------------------------

def _mobile_manipulator_jacobian(theta_list: np.ndarray) -> np.ndarray:
    """
    Compute the 6×9 mobile-manipulator Jacobian Je(θ).

    PHYSICAL MEANING
    ----------------
    Je relates the 9 actuator speeds (4 wheels + 5 joints) to the
    resulting 6D end-effector twist in the body frame {e}:

        V_e = Je(θ) · [u; θ̇]

    It has two column blocks:

    J_base (columns 0–3)
        Each column is the end-effector twist produced by spinning one
        wheel at unit speed while all other actuators are still.
        F6 first converts wheel speed to a 6D chassis body twist (in
        {b}); the Adjoint Ad_{T_0e⁻¹ T_b0⁻¹} then changes the reference
        frame from {b} all the way to {e}:

            frame chain:  {e} ← T_0e⁻¹ ← {0} ← T_b0⁻¹ ← {b}

    J_arm (columns 4–8)
        Standard body Jacobian of the 5-DOF arm.  Column i is the
        end-effector twist produced by rotating joint i at unit speed
        with all other joints fixed (screw axis Bi re-expressed for the
        current joint configuration via the adjoint of the product of
        exponentials).

    WHY THE PSEUDOINVERSE?
    ----------------------
    Je is 6×9 (more columns than rows) so infinitely many (u, θ̇)
    produce the same V.  np.linalg.pinv gives the minimum-2-norm
    solution, distributing effort across wheels and joints to avoid
    unnecessarily large individual speeds.

    Parameters
    ----------
    theta_list : (5,) arm joint angles [J1…J5] (rad)

    Returns
    -------
    Je : ndarray (6, 9)
    """
    # Forward kinematics of arm: {0} → {e} for current joint angles
    T_0e = mr.FKinBody(M_0e, Blist, theta_list)

    # Re-express chassis contribution in end-effector frame {e}
    # Ad_{T_0e⁻¹ T_b0⁻¹}: frame chain e ← 0 ← b
    Ad_base = mr.Adjoint(mr.TransInv(T_0e) @ mr.TransInv(T_b0))  # 6×6
    J_base  = Ad_base @ _F6()                                      # 6×4

    # Body Jacobian of the arm (already in {e})
    J_arm = mr.JacobianBody(Blist, theta_list)                     # 6×5

    return np.hstack([J_base, J_arm])                              # 6×9


# ---------------------------------------------------------------------------
# Core control law
# ---------------------------------------------------------------------------

def FeedbackControl(
    X:             np.ndarray,
    X_d:           np.ndarray,
    X_d_next:      np.ndarray,
    K_p:           np.ndarray,
    K_i:           np.ndarray,
    dt:            float,
    theta_list:    np.ndarray,
    X_err_integral: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the commanded twist and actuator speeds for one control timestep.

    STEP-BY-STEP PHYSICS
    ---------------------
    Step 1 — Error twist Xerr
        The SE(3) matrix  X⁻¹ Xd  is the pose of the reference frame
        {Xd} expressed in the actual current frame {X}.  Taking its
        matrix logarithm gives the se(3) element (a 4×4 skew-symmetric
        matrix) whose associated screw motion, applied for one unit of
        time, would take X exactly to Xd.  se3ToVec extracts the 6-vector
        [ω_x, ω_y, ω_z, v_x, v_y, v_z] from that matrix.

        Xerr = se3ToVec( log( X⁻¹ Xd ) )

        A zero Xerr means perfect tracking.  A large Xerr means the robot
        has drifted far from the reference path.

    Step 2 — Integral update
        The integral accumulates Xerr·Δt at every call, giving a running
        estimate of ∫Xerr dt.  This is the "I" part of PI control.
        It corrects steady-state errors: if a constant disturbance (e.g.
        a stuck joint) keeps Xerr non-zero, the integral term grows until
        its contribution to V overwhelms the disturbance.

    Step 3 — Feedforward twist Vd
        Xd⁻¹ Xd_next is the relative motion the reference trajectory
        wants to happen in the next Δt seconds, expressed in the current
        reference frame {Xd}.  Dividing by Δt converts finite rotation/
        translation into an instantaneous twist (first-order approximation
        of the matrix logarithm velocity):

        Vd = (1/Δt) se3ToVec( log( Xd⁻¹ Xd_next ) )

        This twist lives in frame {Xd}.  It tells the robot what motion
        the path demands, independent of any error.

    Step 4 — Control law
        The Adjoint Ad_{X⁻¹ Xd} re-expresses Vd from {Xd} into {X} (the
        actual current end-effector frame).  This is necessary because the
        Jacobian Je maps actuator speeds to twists expressed in {X = e},
        not in {Xd}.

        The full commanded twist is:
            V = Ad_{X⁻¹ Xd} · Vd   (feedforward: follow the plan)
              + Kp · Xerr            (proportional: correct current gap)
              + Ki · ∫Xerr dt        (integral: eliminate steady error)

        Kp and Ki are typically diagonal 6×6 matrices.  Large diagonal
        values give faster error correction but risk oscillation and large
        actuator speeds near singularities.

    Step 5 — Map twist to actuator speeds
        The 6D commanded twist V is distributed across 9 actuators via
        the pseudoinverse of the 6×9 mobile-manipulator Jacobian Je:

            [u; θ̇] = Je(θ)⁺ · V

        This minimum-norm solution keeps all speeds as small as possible.
        When Je is near-singular (arm near a singularity or bad posture),
        the pseudoinverse amplifies V into very large speeds — this is a
        physical warning that the robot configuration needs to be improved.

    Parameters
    ----------
    X              : (4,4) SE(3) – current actual end-effector pose T_se
    X_d            : (4,4) SE(3) – current reference pose T_se_d
    X_d_next       : (4,4) SE(3) – reference pose at next timestep T_se_d_next
    K_p            : (6,6) proportional gain matrix (diagonal recommended)
    K_i            : (6,6) integral gain matrix (diagonal recommended)
    dt             : float – timestep Δt (s), must match trajectory resolution
    theta_list     : (5,) current arm joint angles [J1…J5] (rad)
    X_err_integral : (6,) running integral of Xerr from all previous steps

    Returns
    -------
    V              : (6,) commanded end-effector twist in frame {e}
    controls       : (9,) [u1,u2,u3,u4, θ̇1,…,θ̇5]  wheel + joint speeds
    X_err          : (6,) current error twist (useful for logging / plotting)
    X_err_integral : (6,) updated integral, pass back in on the next call
    """

    # ------------------------------------------------------------------
    # Step 1: Error twist  Xerr = se3ToVec( log( X⁻¹ Xd ) )
    #
    # X⁻¹ Xd  ∈ SE(3): relative transform from actual pose to reference.
    # MatrixLog6 maps it to its se(3) Lie-algebra element (skew matrix).
    # se3ToVec extracts [ω; v] as a 6-vector.
    # ------------------------------------------------------------------
    X_err = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(X) @ X_d))   # (6,)

    # ------------------------------------------------------------------
    # Step 2: Accumulate the integral  ∫Xerr dt  ≈  Σ Xerr·Δt
    #
    # The caller must pass the current integral in and use the returned
    # updated value on the next call — state is owned by the caller.
    # ------------------------------------------------------------------
    X_err_integral = X_err_integral + X_err * dt                # (6,)

    # ------------------------------------------------------------------
    # Step 3: Feedforward reference twist  Vd = (1/Δt) log( Xd⁻¹ Xd_next )
    #
    # Xd⁻¹ Xd_next  is the relative motion of the reference frame in Δt.
    # Scaling by 1/Δt converts the finite screw displacement into an
    # instantaneous body twist in frame {Xd}.
    # ------------------------------------------------------------------
    V_d = (1.0 / dt) * mr.se3ToVec(
        mr.MatrixLog6(mr.TransInv(X_d) @ X_d_next)
    )                                                            # (6,)

    # ------------------------------------------------------------------
    # Step 4: Full PI + feedforward control law
    #
    # Ad_{X⁻¹ Xd}: re-expresses V_d from reference frame {Xd} into the
    # actual end-effector frame {X}.  Without this mapping the feedforward
    # would push in the wrong direction whenever X ≠ Xd.
    # ------------------------------------------------------------------
    Ad_X_inv_Xd = mr.Adjoint(mr.TransInv(X) @ X_d)             # 6×6
    V = (Ad_X_inv_Xd @ V_d          # feedforward: follow the planned path
         + K_p @ X_err              # proportional: shrink the current pose gap
         + K_i @ X_err_integral)    # integral: eliminate persistent steady errors

    # ------------------------------------------------------------------
    # Step 5: Invert the Jacobian to get actuator speeds
    #
    # Je is 6×9 (underdetermined): many (u, θ̇) produce the same V.
    # np.linalg.pinv gives the minimum-norm solution — the actuator
    # speeds closest to zero that still achieve V exactly (in a least-
    # squares sense when Je is numerically rank-deficient).
    # ------------------------------------------------------------------
    Je      = _mobile_manipulator_jacobian(theta_list)          # 6×9
    Je_pinv = np.linalg.pinv(Je)                                # 9×6
    controls = Je_pinv @ V                                      # (9,)

    logger.debug(
        "Xerr=%s  Vd=%s  V=%s  controls=%s",
        np.array2string(X_err,    precision=4),
        np.array2string(V_d,      precision=4),
        np.array2string(V,        precision=4),
        np.array2string(controls, precision=4),
    )

    return V, controls, X_err, X_err_integral


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------

def write_error_csv(
    error_log: list[np.ndarray],
    filepath:  str,
) -> None:
    """
    Write the per-timestep 6-vector Xerr log to a CSV file.

    Each row: [Xerr_1, Xerr_2, Xerr_3, Xerr_4, Xerr_5, Xerr_6]
    Plotting this file over time shows how quickly the controller drives
    the error to zero — the key performance metric for Milestone 3.
    """
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        for row in error_log:
            writer.writerow([f"{v:.6f}" for v in row])
    logger.info("Error CSV written → %s  (%d rows)", filepath, len(error_log))


# ---------------------------------------------------------------------------
# Quick sanity test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("=== Milestone 3 – Feedback Control (sanity test) ===")

    # ------------------------------------------------------------------
    # Test A: Perfect tracking  (X = Xd)
    # Expected: Xerr = 0, V = Vd only (feedforward drives the motion)
    # ------------------------------------------------------------------
    X_a      = np.eye(4)
    X_d_a    = np.eye(4)
    X_d_next_a = np.array([
        [1, 0, 0, 0.01],
        [0, 1, 0, 0   ],
        [0, 0, 1, 0   ],
        [0, 0, 0, 1   ],
    ], dtype=float)

    K_p = np.eye(6) * 2.0
    K_i = np.eye(6) * 0.0
    dt  = 0.01
    theta_list = np.zeros(5)

    V, controls, X_err, _ = FeedbackControl(
        X_a, X_d_a, X_d_next_a, K_p, K_i, dt, theta_list, np.zeros(6)
    )
    logger.info("Test A (perfect tracking):  Xerr=%s (expect ~0)",
                np.array2string(X_err, precision=4))
    logger.info("                            V   =%s",
                np.array2string(V, precision=4))

    # ------------------------------------------------------------------
    # Test B: Non-zero error  (Xd rotated 0.1 rad about z, X = I)
    # Expected: Xerr has non-zero ω_z component; feedback corrects it
    # ------------------------------------------------------------------
    import math
    c, s = math.cos(0.1), math.sin(0.1)
    X_d_rot = np.array([
        [ c, -s, 0, 0],
        [ s,  c, 0, 0],
        [ 0,  0, 1, 0],
        [ 0,  0, 0, 1],
    ], dtype=float)

    V2, ctrl2, err2, _ = FeedbackControl(
        np.eye(4), X_d_rot, X_d_rot,
        np.eye(6) * 5.0, np.eye(6) * 0.1,
        dt, theta_list, np.zeros(6)
    )
    logger.info("Test B (0.1 rad z-error):   Xerr=%s (expect non-zero ω_z)",
                np.array2string(err2, precision=4))
    logger.info("                            V   =%s",
                np.array2string(V2, precision=4))
