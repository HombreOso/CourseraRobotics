"""
Milestone 1 – youBot Kinematic Simulator
=========================================
Implements NextState, a first-order Euler kinematic simulator for the
youBot mobile manipulator, and writes the resulting trajectory to a CSV
file in the format required by the CoppeliaSim scene.

State vector (13 elements)
--------------------------
  [0]     chassis φ  (rad)
  [1]     chassis x  (m)
  [2]     chassis y  (m)
  [3-7]   arm joint angles J1–J5 (rad)
  [8-11]  wheel angles W1–W4 (rad)
  [12]    gripper state  (0 = open, 1 = closed)

Controls vector (9 elements)
-----------------------------
  [0-3]   wheel angular speeds u1–u4 (rad/s)
  [4-8]   arm joint speeds θ̇1–θ̇5 (rad/s)
"""

import csv
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

from configurations import F, r, l, w


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _build_logger(name: str = "milestone_1") -> logging.Logger:
    """
    Return a logger that writes to both the console and a timestamped .log
    file in the same directory as this script.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir   = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path  = log_dir / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler – DEBUG and above
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler – INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Log file: %s", log_path)
    return logger


logger = _build_logger()


# ---------------------------------------------------------------------------
# Core kinematics
# ---------------------------------------------------------------------------

def _odometry(q: np.ndarray, wheel_speeds: np.ndarray, dt: float) -> np.ndarray:
    """
    Update chassis configuration q = (φ, x, y) using first-order odometry
    (Modern Robotics Chapter 13.4).

    Parameters
    ----------
    q            : current chassis config [φ, x, y]
    wheel_speeds : 4-vector of wheel angular speeds [u1, u2, u3, u4] (rad/s)
    dt           : timestep (s)

    Returns
    -------
    q_new : updated chassis config [φ, x, y]
    """
    delta_theta = wheel_speeds * dt          # wheel angle increments

    # Body-frame planar twist [ω_bz, v_bx, v_by]
    Vb = F @ delta_theta                     # shape (3,)
    w_bz = Vb[0]
    v_bx = Vb[1]
    v_by = Vb[2]

    if abs(w_bz) < 1e-9:
        # Straight-line motion: no rotation correction needed
        delta_qb = np.array([w_bz, v_bx, v_by])
    else:
        delta_qb = np.array([
            w_bz,
            (v_bx * np.sin(w_bz) + v_by * (np.cos(w_bz) - 1.0)) / w_bz,
            (v_by * np.sin(w_bz) + v_bx * (1.0 - np.cos(w_bz))) / w_bz,
        ])

    phi = q[0]
    # Rotate body-frame displacement into space frame
    R_sb = np.array([
        [1.0,           0.0,          0.0],
        [0.0,  np.cos(phi), -np.sin(phi)],
        [0.0,  np.sin(phi),  np.cos(phi)],
    ])
    delta_q = R_sb @ delta_qb

    return q + delta_q


def NextState(
    current_config: np.ndarray,
    controls: np.ndarray,
    dt: float,
    max_speed: float,
) -> np.ndarray:
    """
    Compute the youBot configuration one timestep later (first-order Euler).

    Parameters
    ----------
    current_config : 13-vector
        [φ, x, y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper]
    controls : 9-vector
        [u1, u2, u3, u4, θ̇1, θ̇2, θ̇3, θ̇4, θ̇5]
    dt : float
        Timestep Δt (s)
    max_speed : float
        Maximum absolute angular speed for all actuators (rad/s).
        Pass np.inf to disable clamping.

    Returns
    -------
    new_config : 13-vector
        Updated configuration at time t + Δt.
    """
    current_config = np.asarray(current_config, dtype=float)
    controls = np.asarray(controls, dtype=float)

    # Clamp controls to [-max_speed, max_speed]
    controls = np.clip(controls, -max_speed, max_speed)

    wheel_speeds = controls[0:4]   # u1–u4
    joint_speeds = controls[4:9]   # θ̇1–θ̇5

    # 1. Update arm joint angles
    old_joints  = current_config[3:8]
    new_joints  = old_joints + joint_speeds * dt

    # 2. Update wheel angles
    old_wheels  = current_config[8:12]
    new_wheels  = old_wheels + wheel_speeds * dt

    # 3. Update chassis via odometry
    old_q   = current_config[0:3]
    new_q   = _odometry(old_q, wheel_speeds, dt)

    # Gripper state is unchanged by NextState
    gripper = current_config[12]

    return np.concatenate([new_q, new_joints, new_wheels, [gripper]])


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def write_csv(trajectory: list[np.ndarray], filepath: str) -> None:
    """
    Write a list of 13-element state vectors to a CSV file.

    Each row:  φ, x, y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper
    """
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        for state in trajectory:
            writer.writerow([f"{v:.6f}" for v in state])
    logger.info("CSV written  → %s  (%d rows)", filepath, len(trajectory))


def simulate(
    initial_config: np.ndarray,
    controls_sequence: np.ndarray,
    dt: float,
    max_speed: float,
) -> list[np.ndarray]:
    """
    Run the simulator for len(controls_sequence) timesteps.

    Parameters
    ----------
    initial_config    : 13-vector, starting state
    controls_sequence : (N, 9) array of control inputs
    dt                : timestep (s)
    max_speed         : speed clamp (rad/s)

    Returns
    -------
    trajectory : list of N+1 state vectors (includes the initial config)
    """
    trajectory = [np.asarray(initial_config, dtype=float).copy()]
    config = trajectory[0].copy()

    logger.info(
        "simulate  N=%d steps  dt=%.4f s  max_speed=%.3f rad/s",
        len(controls_sequence), dt, max_speed,
    )
    logger.debug("Initial config: %s", np.array2string(trajectory[0], precision=5))

    for step, u in enumerate(controls_sequence):
        config = NextState(config, u, dt, max_speed)
        trajectory.append(config.copy())
        logger.debug("step %4d  config: %s", step + 1,
                     np.array2string(config, precision=5))

    logger.info("simulate  done.  Final config: %s",
                np.array2string(config, precision=5))
    return trajectory


# ---------------------------------------------------------------------------
# Quick test  –  straight-forward drive for 1 s then stop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("=== Milestone 1 – youBot kinematic simulator ===")

    # Initial configuration: everything at zero, gripper open
    q0 = np.zeros(13)

    # Drive all four wheels forward at 10 rad/s for 100 steps of 0.01 s
    u_drive = np.array([10.0, 10.0, 10.0, 10.0,  0.0, 0.0, 0.0, 0.0, 0.0])
    N_drive = 100
    controls = np.tile(u_drive, (N_drive, 1))

    traj = simulate(q0, controls, dt=0.01, max_speed=12.3)

    out_path = "milestone_1_test_output.csv"
    write_csv(traj, out_path)

    labels = ["phi", "x", "y", "J1", "J2", "J3", "J4", "J5",
              "W1", "W2", "W3", "W4", "grip"]
    summary = "  ".join(f"{lbl}={val: .5f}"
                        for lbl, val in zip(labels, traj[-1]))
    logger.info("Final state: %s", summary)
