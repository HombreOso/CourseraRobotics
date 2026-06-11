"""
Capstone Full Program – youBot Mobile Manipulation
===================================================
Integrates all three milestones into a single closed-loop simulation:

    Milestone 1  NextState           – kinematic Euler integration
    Milestone 2  TrajectoryGenerator – 8-segment reference trajectory
    Milestone 3  FeedbackControl     – feedforward + PI control law

Loop structure (N = number of reference configurations):

    for i in 0 … N-2:
        X_d      = reference trajectory row i   (current reference pose)
        X_d_next = reference trajectory row i+1 (next reference pose)
        T_se     = FK(robot state)              (actual end-effector pose)
        V, controls, Xerr, integral = FeedbackControl(...)
        robot_state = NextState(robot_state, controls, dt, max_speed)
        gripper     = reference gripper state at step i+1
        store robot_state + gripper  every k-th step
        store Xerr                   every k-th step

Outputs
-------
  capstone_trajectory.csv   – robot configs for CoppeliaSim (13 cols)
  capstone_Xerr.csv         – error twist history for plotting (6 cols)
"""

import csv
import logging
import numpy as np
import modern_robotics as mr
from datetime import datetime
from pathlib import Path

from configurations import T_b0, M_0e, Blist
from cube_config import T_sc_initial, T_sc_goal
from milestone_1_youBot_kinematic_simulator import NextState
from milestone_2_reference_trajectory_generation import (
    TrajectoryGenerator,
    T_ce_grasp_default,
    T_ce_standoff_default,
)
from milestone_3_feedback_control import FeedbackControl, JOINT_LIMITS


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _build_logger(name: str = "capstone") -> logging.Logger:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir  = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path  = log_dir / f"{name}_{timestamp}.log"

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
# Helpers
# ---------------------------------------------------------------------------

def _robot_state_to_T_se(config: np.ndarray) -> np.ndarray:
    """
    Compute the actual end-effector pose T_se from the 13-element robot state.

    Chain:  T_se = T_sb(φ,x,y)  @  T_b0  @  FKinBody(M_0e, Blist, θ)

    T_sb is the SE(3) pose of the chassis body frame {b} in the space
    frame {s}.  It is constructed from (φ, x, y) plus the fixed chassis
    height z = 0.0963 m.
    """
    phi, x, y   = config[0], config[1], config[2]
    theta_list  = config[3:8]

    c, s = np.cos(phi), np.sin(phi)
    T_sb = np.array([
        [ c, -s, 0, x     ],
        [ s,  c, 0, y     ],
        [ 0,  0, 1, 0.0963],
        [ 0,  0, 0, 1     ],
    ], dtype=float)

    T_0e = mr.FKinBody(M_0e, Blist, theta_list)
    return T_sb @ T_b0 @ T_0e


def _traj_row_to_T_se(row: list[float]) -> np.ndarray:
    """
    Reconstruct a 4×4 SE(3) matrix from a 13-element trajectory row:
        [r11,r12,r13, r21,r22,r23, r31,r32,r33, px,py,pz, gripper]
    """
    R = np.array([
        [row[0], row[1], row[2]],
        [row[3], row[4], row[5]],
        [row[6], row[7], row[8]],
    ])
    p = np.array([row[9], row[10], row[11]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = p
    return T


def _write_csv(rows: list, filepath: str, label: str) -> None:
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow([f"{v:.6f}" for v in row])
    logger.info("%s → %s  (%d rows)", label, filepath, len(rows))


def _log_step(step: int, t: float,
              config:      np.ndarray,
              T_se:        np.ndarray,
              X_d:         np.ndarray,
              X_err:       np.ndarray,
              controls:    np.ndarray,
              config_next: np.ndarray) -> None:
    """
    Write one full control iteration to the DEBUG log.

    Layout per step
    ---------------
    ─── step NNNN  t=T.TTTs ──────────
      START config  : φ x y  J=[...]  W=[...]  grip
      T_se START    : 4×4 matrix  (FK of START config)
      T_se REF(X_d) : 4×4 matrix  (reference trajectory)
      X_err         : [ω v]  |Xerr|=N
      controls J_dot: [...]
      controls W_dot: [...]
      END   config  : φ x y  J=[...]  W=[...]  grip
      T_se END      : 4×4 matrix  (FK of END config)
      LIMIT VIOLATED: Jk=val (lo..hi)               ← WARNING on console too
    """
    def log_T(label: str, T: np.ndarray) -> None:
        """Log a 4x4 SE(3) matrix as four rows."""
        logger.debug("  %s:", label)
        for row in T:
            logger.debug("    [ %8.4f  %8.4f  %8.4f  %8.4f ]",
                         row[0], row[1], row[2], row[3])

    def fmt_arr(a: np.ndarray, prec: int = 4) -> str:
        return "[" + "  ".join(f"{v:+.{prec}f}" for v in a) + "]"

    J_cur  = config[3:8]
    W_cur  = config[8:12]
    J_next = config_next[3:8]
    W_next = config_next[8:12]

    T_se_next = _robot_state_to_T_se(config_next)

    logger.debug("─── step %4d  t=%7.3f s ───────────────────────────────────────────",
                 step, t)
    logger.debug("  START config  : φ=%+.4f x=%+.4f y=%+.4f  J=%s  W=%s  grip=%d",
                 config[0], config[1], config[2],
                 fmt_arr(J_cur), fmt_arr(W_cur), int(config[12]))
    log_T("T_se START (actual)", T_se)
    log_T("T_se REF   (X_d)   ", X_d)
    logger.debug("  X_err         : %s  |Xerr|=%8.4f",
                 fmt_arr(X_err), float(np.linalg.norm(X_err)))
    logger.debug("  controls J_dot: %s", fmt_arr(controls[:5]))
    logger.debug("  controls W_dot: %s", fmt_arr(controls[5:]))
    logger.debug("  END   config  : φ=%+.4f x=%+.4f y=%+.4f  J=%s  W=%s  grip=%d",
                 config_next[0], config_next[1], config_next[2],
                 fmt_arr(J_next), fmt_arr(W_next), int(config_next[12]))
    log_T("T_se END   (after NextState)", T_se_next)

    # Joint limit violations in the NEW configuration
    violations = []
    for k, (theta, (lo, hi)) in enumerate(zip(J_next, JOINT_LIMITS)):
        if theta < lo or theta > hi:
            violations.append(f"J{k+1}={theta:+.4f} ({lo:.3f}..{hi:.3f})")
    if violations:
        msg = "  LIMIT VIOLATED: " + "  ".join(violations)
        logger.debug(msg)
        logger.warning("step %4d  LIMIT VIOLATED: %s", step, "  ".join(violations))


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def build_trajectory(
    initial_robot_config: np.ndarray,
    k:     int   = 1,
    v_max: float = 0.5,
    w_max: float = 1.0,
) -> list:
    """
    Generate the 8-segment reference trajectory from the robot's initial
    end-effector pose (computed via FK from `initial_robot_config`).

    Use this separately when you want to fix the trajectory and then run
    the simulation with a *different* (e.g. perturbed) starting config.

    Returns
    -------
    trajectory : list of N×13 rows
    """
    T_se_initial = _robot_state_to_T_se(initial_robot_config)
    logger.info("build_trajectory: T_se_initial from FK:\n%s",
                np.array2string(T_se_initial, precision=4))
    return TrajectoryGenerator(
        T_se_initial   = T_se_initial,
        T_sc_initial   = T_sc_initial,
        T_sc_final     = T_sc_goal,
        T_ce_grasp     = T_ce_grasp_default,
        k              = k,
        v_max          = v_max,
        w_max          = w_max,
        method         = 5,
    )


def run_capstone(
    initial_robot_config: np.ndarray,
    K_p:        np.ndarray,
    K_i:        np.ndarray,
    k:          int   = 1,
    max_speed:  float = np.inf,
    v_max:      float = 0.5,
    w_max:      float = 1.0,
    traj_csv:   str   = "capstone_trajectory.csv",
    err_csv:    str   = "capstone_Xerr.csv",
    trajectory: list  = None,
) -> tuple[list, list]:
    """
    Run the full capstone closed-loop simulation.

    Parameters
    ----------
    initial_robot_config : (13,) starting state [φ,x,y,J1…J5,W1…W4,grip]
    K_p        : (6,6) proportional gain matrix
    K_i        : (6,6) integral gain matrix
    k          : reference configs per 0.01 s  (1 = one config per 10 ms)
    max_speed  : actuator speed clamp (rad/s); np.inf = no limit
    v_max      : max linear speed for trajectory generation (m/s)
    w_max      : max angular speed for trajectory generation (rad/s)
    traj_csv   : output path for robot configuration trajectory
    err_csv    : output path for Xerr log
    trajectory : pre-built trajectory (list of rows).  If None, a new
                 trajectory is generated from the FK of initial_robot_config.
                 Pass a pre-built trajectory to test with a perturbed start.

    Returns
    -------
    config_log : list of (13,) arrays  (one per stored timestep)
    error_log  : list of (6,)  arrays  (one per stored timestep)
    """
    dt = 0.01 / k   # time between reference configurations (s)

    # ------------------------------------------------------------------
    # Step 1 – Obtain reference trajectory
    # ------------------------------------------------------------------
    if trajectory is None:
        # Generate from the FK of the initial robot configuration (perfect start)
        T_se_initial = _robot_state_to_T_se(initial_robot_config)
        logger.info("Initial T_se from FK:\n%s",
                    np.array2string(T_se_initial, precision=4))
        logger.info("Generating reference trajectory …")
        trajectory = TrajectoryGenerator(
            T_se_initial   = T_se_initial,
            T_sc_initial   = T_sc_initial,
            T_sc_final     = T_sc_goal,
            T_ce_grasp     = T_ce_grasp_default,
            T_ce_standoff  = T_ce_standoff_default,
            k              = k,
            v_max          = v_max,
            w_max          = w_max,
            method         = 5,
        )
    else:
        logger.info("Using pre-built trajectory (%d rows).", len(trajectory))
        # Log how far the robot's actual start is from the trajectory start
        T_se_actual = _robot_state_to_T_se(initial_robot_config)
        T_se_ref    = _traj_row_to_T_se(trajectory[0])
        X_err0 = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(T_se_actual) @ T_se_ref))
        logger.info("Initial Xerr (actual vs reference): %s  |Xerr|=%.4f",
                    np.array2string(X_err0, precision=4),
                    float(np.linalg.norm(X_err0)))

    N = len(trajectory)
    logger.info("Reference trajectory: %d configurations  (%.2f s total)",
                N, (N - 1) * dt)

    # ------------------------------------------------------------------
    # Step 3 – Closed-loop simulation  (N-1 iterations)
    # ------------------------------------------------------------------
    robot_config   = np.array(initial_robot_config, dtype=float)
    X_err_integral = np.zeros(6)

    # Include the initial config in the output
    config_log: list[np.ndarray] = [robot_config.copy()]
    error_log:  list[np.ndarray] = []

    logger.info("Starting control loop: %d iterations …", N - 1)

    for i in range(N - 1):
        # ---- reference poses for this iteration ----
        X_d      = _traj_row_to_T_se(trajectory[i])
        X_d_next = _traj_row_to_T_se(trajectory[i + 1])

        # ---- actual end-effector pose from current robot state ----
        T_se = _robot_state_to_T_se(robot_config)

        # ---- arm joint angles for Jacobian ----
        theta_list = robot_config[3:8]

        # ---- Milestone 3: feedback + feedforward control ----
        _V, controls, X_err, X_err_integral = FeedbackControl(
            X              = T_se,
            X_d            = X_d,
            X_d_next       = X_d_next,
            K_p            = K_p,
            K_i            = K_i,
            dt             = dt,
            theta_list     = theta_list,
            X_err_integral = X_err_integral,
        )

        # ---- Milestone 1: integrate robot state one timestep ----
        config_before = robot_config.copy()
        robot_config  = NextState(robot_config, controls, dt, max_speed)

        # ---- gripper state comes from the reference trajectory ----
        robot_config[12] = trajectory[i + 1][12]

        # ---- per-step debug log ----
        _log_step(
            step        = i,
            t           = i * dt,
            config      = config_before,
            T_se        = T_se,
            X_d         = X_d,
            X_err       = X_err,
            controls    = controls,
            config_next = robot_config,
        )

        # ---- store every k-th step ----
        if (i + 1) % k == 0:
            config_log.append(robot_config.copy())
            error_log.append(X_err.copy())

        if (i + 1) % max(1, (N // 10)) == 0:
            logger.info("  step %5d / %d   |Xerr|=%.4f",
                        i + 1, N - 1, float(np.linalg.norm(X_err)))

    logger.info("Control loop complete.  Final |Xerr| = %.4f",
                float(np.linalg.norm(error_log[-1])) if error_log else 0.0)

    # ------------------------------------------------------------------
    # Step 4 – Write output files
    # ------------------------------------------------------------------
    _write_csv(config_log, traj_csv, "Robot config CSV")
    _write_csv(error_log,  err_csv,  "Xerr CSV        ")

    return config_log, error_log


# ---------------------------------------------------------------------------
# Entry point – default capstone scenario
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Capstone Full Program – youBot Mobile Manipulation")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Shared: "perfect" initial robot configuration
    # Arm joints θ = (0, 0, 0.2, -1.6, 0) place the end-effector at a
    # reasonable pose above the robot base.  The reference trajectory is
    # generated FROM this configuration, so Xerr = 0 at t = 0.
    # ------------------------------------------------------------------
    perfect_config = np.array([
        0.0,   # chassis φ
        0.0,   # chassis x
        0.0,   # chassis y
        0.0,   # J1
        0.0,   # J2
        0.2,   # J3
       -1.6,   # J4
        0.0,   # J5
        0.0,   # W1
        0.0,   # W2
        0.0,   # W3
        0.0,   # W4
        0.0,   # gripper (open)
    ])

    # Build the reference trajectory once from the perfect start.
    # Both feedforward tests reuse this same trajectory.
    ref_trajectory = build_trajectory(perfect_config, k=1, v_max=0.5, w_max=1.0)

    K_zero = np.zeros((6, 6))   # feedforward-only (Tests A & B)
    K_p    = np.eye(6) * 2.0    # proportional gain  (Test C)
    K_i    = np.eye(6) * 0.2    # integral gain      (Test C)

    # ==================================================================
    # Test A – Feedforward only, perfect initial condition
    # Expected: robot follows the trajectory closely; Xerr stays ~0
    # because the Euler integration error is the only source of drift.
    # ==================================================================
    logger.info("")
    logger.info("--- Test A: feedforward only, perfect start ---")
    run_capstone(
        initial_robot_config = perfect_config,
        K_p        = K_zero,
        K_i        = K_zero,
        k          = 1,
        max_speed  = np.inf,
        trajectory = ref_trajectory,
        traj_csv   = "capstone_testA_trajectory.csv",
        err_csv    = "capstone_testA_Xerr.csv",
    )

    # ==================================================================
    # Test B – Feedforward only, deliberate initial error
    # The robot starts with chassis offset (+0.1 m in x, +0.1 m in y,
    # +0.1 rad in φ) so Xerr ≠ 0 at t = 0.
    # Expected: the error does NOT shrink — the feedforward term only
    # tracks the planned path; without feedback the gap persists.
    # This makes sense physically: the feedforward has no knowledge of
    # where the robot actually is; it blindly follows the plan.
    # ==================================================================
    logger.info("")
    logger.info("--- Test B: feedforward only, perturbed start ---")
    perturbed_config = perfect_config.copy()
    perturbed_config[0] += 0.1    # φ offset
    perturbed_config[1] += 0.1    # x offset
    perturbed_config[2] += 0.1    # y offset

    run_capstone(
        initial_robot_config = perturbed_config,
        K_p        = K_zero,
        K_i        = K_zero,
        k          = 1,
        max_speed  = np.inf,
        trajectory = ref_trajectory,   # same trajectory as Test A
        traj_csv   = "capstone_testB_trajectory.csv",
        err_csv    = "capstone_testB_Xerr.csv",
    )

    # ==================================================================
    # Test C – PI feedback, perturbed start (full controller)
    # Expected: Xerr starts non-zero but decays toward 0 as the
    # proportional term drives the robot back onto the reference path.
    # ==================================================================
    logger.info("")
    logger.info("--- Test C: PI feedback (Kp=2·I), perturbed start ---")
    run_capstone(
        initial_robot_config = perturbed_config,
        K_p        = K_p,
        K_i        = K_i,
        k          = 1,
        max_speed  = np.inf,
        trajectory = ref_trajectory,   # same trajectory as Tests A & B
        traj_csv   = "capstone_testC_trajectory.csv",
        err_csv    = "capstone_testC_Xerr.csv",
    )

    logger.info("")
    logger.info("All three tests complete.  Load the CSV files into")
    logger.info("CoppeliaSim Scene 6 and compare the animations.")
