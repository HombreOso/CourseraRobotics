"""
Milestone 3 – Feedforward + PI Feedback Control
=================================================
Implements FeedbackControl, the kinematic task-space control law for the
youBot mobile manipulator (Modern Robotics Equations 11.16 / 13.37):

    V(t) = [Ad_{X⁻¹ Xd}] Vd(t)  +  Kp · Xerr(t)  +  Ki · ∫Xerr(t) dt

where
    X        = T_se             current actual end-effector pose in {s}
    Xd       = T_se_d           current reference pose in {s}
    Xd_next  = T_se_d_next      reference pose one Δt later
    Xerr     = se3ToVec(log(X⁻¹ Xd))          6-vector error twist
    Vd       = (1/Δt) se3ToVec(log(Xd⁻¹ Xd_next))  feedforward twist
    V        = commanded end-effector twist in {e}

The commanded twist is converted to wheel + joint speeds via the
pseudoinverse of the mobile-manipulator Jacobian Je(θ):

    [u; θ̇] = Je(θ)⁺ · V

Je is 6×9:
    Je = [ Ad_{T0e⁻¹ T_b0⁻¹} F6 | Jb(θ) ]
         |_______6x4___________|  |_6x5_|

where F6 is the 6×4 mecanum chassis matrix and Jb(θ) is the body
Jacobian of the 5-DOF arm.
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

    Je = [ J_base  |  J_arm ]

    J_base (6×4):
        Ad_{T_0e⁻¹  T_b0⁻¹} @ F6
        Maps 4 wheel speeds to end-effector twist in {e}.

    J_arm (6×5):
        JacobianBody(Blist, theta_list)
        Maps 5 arm-joint speeds to end-effector twist in {e}.

    Parameters
    ----------
    theta_list : (5,) arm joint angles [J1…J5] (rad)

    Returns
    -------
    Je : ndarray (6, 9)
    """
    T_0e  = mr.FKinBody(M_0e, Blist, theta_list)         # {0} → {e}
    T_e0  = mr.TransInv(T_0e)                             # {e} → {0}
    T_0b  = mr.TransInv(T_b0)                             # {0} → {b}  (= T_b0⁻¹)

    # Ad_{T_0e⁻¹ T_b0⁻¹}  (chain: e→0→b expressed in e)
    Ad_base = mr.Adjoint(T_e0 @ T_0b)                    # 6×6

    J_base = Ad_base @ _F6()                              # 6×4
    J_arm  = mr.JacobianBody(Blist, theta_list)           # 6×5

    return np.hstack([J_base, J_arm])                     # 6×9


# ---------------------------------------------------------------------------
# Core control law
# ---------------------------------------------------------------------------

def FeedbackControl(
    X:            np.ndarray,
    X_d:          np.ndarray,
    X_d_next:     np.ndarray,
    K_p:          np.ndarray,
    K_i:          np.ndarray,
    dt:           float,
    theta_list:   np.ndarray,
    X_err_integral: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the commanded twist and joint/wheel speeds for one timestep.

    Parameters
    ----------
    X              : (4,4) SE(3) – current actual end-effector pose T_se
    X_d            : (4,4) SE(3) – current reference pose T_se_d
    X_d_next       : (4,4) SE(3) – reference pose at next timestep T_se_d_next
    K_p            : (6,6) proportional gain matrix (diagonal)
    K_i            : (6,6) integral gain matrix (diagonal)
    dt             : float – timestep Δt (s)
    theta_list     : (5,) arm joint angles [J1…J5] (rad)
    X_err_integral : (6,) running integral of Xerr accumulated so far

    Returns
    -------
    V              : (6,) commanded end-effector twist in {e}
    controls       : (9,) [u1,u2,u3,u4, θ̇1,…,θ̇5]  wheel + joint speeds
    X_err          : (6,) current error twist
    X_err_integral : (6,) updated integral (input + Xerr * dt)
    """
    # ------------------------------------------------------------------
    # 1. Error twist  Xerr = se3ToVec(log(X⁻¹ Xd))
    # ------------------------------------------------------------------
    X_err = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(X) @ X_d))   # (6,)

    # ------------------------------------------------------------------
    # 2. Update integral
    # ------------------------------------------------------------------
    X_err_integral = X_err_integral + X_err * dt                # (6,)

    # ------------------------------------------------------------------
    # 3. Feedforward twist  Vd = (1/Δt) se3ToVec(log(Xd⁻¹ Xd_next))
    # ------------------------------------------------------------------
    V_d = (1.0 / dt) * mr.se3ToVec(
        mr.MatrixLog6(mr.TransInv(X_d) @ X_d_next)
    )                                                            # (6,)

    # ------------------------------------------------------------------
    # 4. Control law (MR Eq. 11.16 / 13.37)
    # ------------------------------------------------------------------
    Ad_X_inv_Xd = mr.Adjoint(mr.TransInv(X) @ X_d)             # 6×6
    V = Ad_X_inv_Xd @ V_d  +  K_p @ X_err  +  K_i @ X_err_integral  # (6,)

    # ------------------------------------------------------------------
    # 5. Map twist to wheel + joint speeds via Je pseudoinverse
    # ------------------------------------------------------------------
    Je       = _mobile_manipulator_jacobian(theta_list)         # 6×9
    Je_pinv  = np.linalg.pinv(Je)                               # 9×6
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

    # Perfect tracking: X = X_d  →  Xerr = 0, V = Vd only
    X   = np.eye(4)
    X_d = np.eye(4)
    X_d_next = np.array([
        [1, 0, 0, 0.01],
        [0, 1, 0, 0   ],
        [0, 0, 1, 0   ],
        [0, 0, 0, 1   ],
    ], dtype=float)

    K_p = np.eye(6) * 2.0
    K_i = np.eye(6) * 0.0
    dt  = 0.01
    theta_list     = np.zeros(5)
    X_err_integral = np.zeros(6)

    V, controls, X_err, X_err_integral = FeedbackControl(
        X, X_d, X_d_next, K_p, K_i, dt, theta_list, X_err_integral
    )

    logger.info("X_err    = %s  (should be ~0)", np.array2string(X_err,    precision=4))
    logger.info("Vd → V   = %s",                 np.array2string(V,        precision=4))
    logger.info("controls = %s",                 np.array2string(controls, precision=4))

    # Non-zero error test: X_d rotated 0.1 rad about z, X = I
    import math
    c, s = math.cos(0.1), math.sin(0.1)
    X_d_rot = np.array([
        [ c, -s, 0, 0],
        [ s,  c, 0, 0],
        [ 0,  0, 1, 0],
        [ 0,  0, 0, 1],
    ], dtype=float)
    K_p2 = np.eye(6) * 5.0
    K_i2 = np.eye(6) * 0.1
    X_err_integral2 = np.zeros(6)

    V2, ctrl2, err2, _ = FeedbackControl(
        np.eye(4), X_d_rot, X_d_rot, K_p2, K_i2,
        dt, theta_list, X_err_integral2
    )
    logger.info("--- Error correction test ---")
    logger.info("Xerr     = %s  (should have non-zero ω_z)", np.array2string(err2,  precision=4))
    logger.info("V        = %s",                              np.array2string(V2,    precision=4))
    logger.info("controls = %s",                              np.array2string(ctrl2, precision=4))
