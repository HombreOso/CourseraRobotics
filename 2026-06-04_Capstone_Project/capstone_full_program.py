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
from milestone_3_feedback_control import FeedbackControl


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _build_logger(name: str = "capstone") -> logging.Logger:
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


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_capstone(
    initial_robot_config: np.ndarray,
    K_p:       np.ndarray,
    K_i:       np.ndarray,
    k:         int   = 1,
    max_speed: float = np.inf,
    v_max:     float = 0.5,
    w_max:     float = 1.0,
    traj_csv:  str   = "capstone_trajectory.csv",
    err_csv:   str   = "capstone_Xerr.csv",
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
    v_max      : max linear speed for trajectory segment durations (m/s)
    w_max      : max angular speed for trajectory segment durations (rad/s)
    traj_csv   : output path for robot configuration trajectory
    err_csv    : output path for Xerr log

    Returns
    -------
    config_log : list of (13,) arrays  (one per stored timestep)
    error_log  : list of (6,)  arrays  (one per stored timestep)
    """
    dt = 0.01 / k   # time between reference configurations (s)

    # ------------------------------------------------------------------
    # Step 1 – Compute the actual initial end-effector pose from FK
    # ------------------------------------------------------------------
    T_se_initial = _robot_state_to_T_se(initial_robot_config)
    logger.info("Initial T_se computed from FK:\n%s",
                np.array2string(T_se_initial, precision=4))

    # ------------------------------------------------------------------
    # Step 2 – Generate the 8-segment reference trajectory (Milestone 2)
    # ------------------------------------------------------------------
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
        robot_config = NextState(robot_config, controls, dt, max_speed)

        # ---- gripper state comes from the reference trajectory ----
        robot_config[12] = trajectory[i + 1][12]

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
    # Initial robot configuration
    # All chassis variables at zero; arm joints chosen so the end-effector
    # starts above and slightly in front of the robot base.
    # Joint angles (rad): θ1=0, θ2=0, θ3=0.2, θ4=-1.6, θ5=0
    # ------------------------------------------------------------------
    initial_config = np.array([
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

    # ------------------------------------------------------------------
    # PI gains  — diagonal 6×6
    # Start with moderate proportional gain and zero integral.
    # Tune K_i > 0 to eliminate steady-state errors if needed.
    # ------------------------------------------------------------------
    Kp_scalar = 2.0
    Ki_scalar = 0.0

    K_p = np.eye(6) * Kp_scalar
    K_i = np.eye(6) * Ki_scalar

    logger.info("Gains: Kp=%.1f·I  Ki=%.1f·I", Kp_scalar, Ki_scalar)

    config_log, error_log = run_capstone(
        initial_robot_config = initial_config,
        K_p       = K_p,
        K_i       = K_i,
        k         = 1,
        max_speed = np.inf,
        v_max     = 0.5,
        w_max     = 1.0,
        traj_csv  = "capstone_trajectory.csv",
        err_csv   = "capstone_Xerr.csv",
    )

    logger.info("Simulation complete.")
    logger.info("  Configuration rows : %d", len(config_log))
    logger.info("  Error rows         : %d", len(error_log))
    logger.info("  Final robot config : %s",
                np.array2string(config_log[-1], precision=4))
