"""
compare_trajectory.py
=====================
Compare every row in a capstone trajectory CSV against the reference trajectory.

For each timestep i the script:
  1. Reads robot config from the CSV  ->  FK  ->  T_se_actual[i]
  2. Reads row i from the reference trajectory  ->  T_se_ref[i]
  3. Reports position error, rotation error, and full 6-D Xerr

CSV format (robot config, 13 cols):
  phi, x, y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper

Reference trajectory row format (13 cols):
  r11,r12,r13, r21,r22,r23, r31,r32,r33, px, py, pz, gripper

Output files  (timestamped, placed next to the input CSV)
-------------
  <stem>_comparison_<timestamp>.log   full per-step table + summary
  <stem>_comparison_<timestamp>.csv   numeric table for plotting

Usage
-----
  py compare_trajectory.py                           # uses built-in defaults
  py compare_trajectory.py capstone_testA_trajectory.csv
"""

import csv
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import modern_robotics as mr

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from cube_config import T_sc_initial, T_sc_goal
from milestone_2_reference_trajectory_generation import (
    TrajectoryGenerator,
    T_ce_grasp_default,
    T_ce_standoff_default,
)
from capstone_full_program import _robot_state_to_T_se


# ── logging setup ─────────────────────────────────────────────────────────────

def _build_logger(stem: str) -> logging.Logger:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir  = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path  = log_dir / f"{stem}_comparison_{timestamp}.log"

    logger = logging.getLogger(f"compare.{stem}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter("%(message)s")   # plain text – no timestamp clutter

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Log file: %s", log_path)
    return logger


# ── helpers ───────────────────────────────────────────────────────────────────

def _row_to_T(row: list[float]) -> np.ndarray:
    """Reconstruct 4x4 T from a 13-element reference-trajectory row."""
    T = np.eye(4)
    T[:3, :3] = np.array(row[:9], dtype=float).reshape(3, 3)
    T[:3,  3] = np.array(row[9:12], dtype=float)
    return T


def _rotation_angle(R_rel: np.ndarray) -> float:
    """Angle of a relative rotation matrix (radians)."""
    cos_a = (np.trace(R_rel) - 1.0) / 2.0
    return math.acos(float(np.clip(cos_a, -1.0, 1.0)))


def _xerr(T_actual: np.ndarray, T_ref: np.ndarray) -> np.ndarray:
    """6-D error twist: se3ToVec(log(T_actual^-1 * T_ref))."""
    return mr.se3ToVec(mr.MatrixLog6(mr.TransInv(T_actual) @ T_ref))


def _fmt_T(T: np.ndarray) -> list[str]:
    """Return a 4x4 matrix as four formatted strings (one per row)."""
    lines = []
    for row in T:
        lines.append(f"    [ {row[0]:8.4f}  {row[1]:8.4f}  {row[2]:8.4f}  {row[3]:8.4f} ]")
    return lines


def load_robot_csv(path: str) -> list[np.ndarray]:
    """Read a capstone trajectory CSV -> list of (13,) config arrays."""
    configs = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if row:
                configs.append(np.array([float(v) for v in row]))
    return configs


# ── main comparison ───────────────────────────────────────────────────────────

def compare(
    robot_csv:      str,
    ref_trajectory: list[list[float]] | None = None,
    perfect_config: np.ndarray | None = None,
    k:     int   = 1,
    v_max: float = 0.5,
    w_max: float = 1.0,
    dt:    float = 0.01,
    out_csv: str | None = None,
) -> None:
    """
    Compare every timestep in `robot_csv` against `ref_trajectory`.

    Parameters
    ----------
    robot_csv      : path to capstone_testX_trajectory.csv
    ref_trajectory : pre-built reference trajectory list; if None, it is
                     regenerated from FK of `perfect_config`
    perfect_config : (13,) initial robot config used to build the trajectory
                     (only needed when ref_trajectory is None)
    k, v_max, w_max: trajectory generation params (must match the original run)
    dt             : timestep in seconds (0.01 / k)
    out_csv        : path for the numeric comparison CSV (auto-named if None)
    """
    stem   = Path(robot_csv).stem
    logger = _build_logger(stem)
    SEP    = "-" * 118

    logger.info("")
    logger.info("Loaded robot CSV : %s", robot_csv)

    configs = load_robot_csv(robot_csv)
    N = len(configs)
    logger.info("Rows             : %d", N)

    # ── build reference trajectory if not supplied ────────────────────────────
    if ref_trajectory is None:
        if perfect_config is None:
            raise ValueError("Supply either ref_trajectory or perfect_config.")
        T_se_initial = _robot_state_to_T_se(perfect_config)
        logger.info("Building reference trajectory from FK of perfect_config ...")
        ref_trajectory = TrajectoryGenerator(
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
        logger.info("Reference trajectory : %d rows", len(ref_trajectory))

    if N != len(ref_trajectory):
        logger.warning("WARNING: CSV has %d rows but reference has %d rows "
                       "-- truncating to min.", N, len(ref_trajectory))
        N = min(N, len(ref_trajectory))

    # ── table header ─────────────────────────────────────────────────────────
    COL_HDR = (f"{'step':>5}  {'t':>6}  "
               f"{'p_actual (x,y,z)':>32}  {'p_ref (x,y,z)':>32}  "
               f"{'pos_err':>8}  {'rot_deg':>7}  {'|Xerr|':>7}  "
               f"{'grip_A':>6}  {'grip_R':>6}")
    logger.info("")
    logger.info(SEP)
    logger.info(COL_HDR)
    logger.info(SEP)

    # ── per-step comparison ───────────────────────────────────────────────────
    rows_out      = []
    max_pos_err   = 0.0
    max_rot_err   = 0.0
    max_xerr_norm = 0.0
    worst_step    = 0

    for i in range(N):
        t = i * dt

        T_actual = _robot_state_to_T_se(configs[i])
        T_ref    = _row_to_T(ref_trajectory[i])

        p_a   = T_actual[:3, 3]
        p_r   = T_ref[:3, 3]
        R_rel = T_actual[:3, :3].T @ T_ref[:3, :3]

        pos_err   = float(np.linalg.norm(p_a - p_r))
        rot_err   = _rotation_angle(R_rel)
        rot_deg   = math.degrees(rot_err)
        Xerr      = _xerr(T_actual, T_ref)
        xerr_norm = float(np.linalg.norm(Xerr))

        grip_a = int(round(configs[i][12]))
        grip_r = int(round(ref_trajectory[i][12]))

        if xerr_norm > max_xerr_norm:
            max_xerr_norm = xerr_norm
            worst_step    = i
        max_pos_err = max(max_pos_err, pos_err)
        max_rot_err = max(max_rot_err, rot_deg)

        high = pos_err > 0.05 or rot_deg > 5.0
        flag = "  *** HIGH ERROR" if high else ""

        # print every 10th row; always print high-error rows
        if i % 10 == 0 or high:
            logger.info("%5d  %6.3f  [%+7.4f %+7.4f %+7.4f]  "
                        "[%+7.4f %+7.4f %+7.4f]  "
                        "%8.4f  %7.3f  %7.4f  %6d  %6d%s",
                        i, t,
                        p_a[0], p_a[1], p_a[2],
                        p_r[0], p_r[1], p_r[2],
                        pos_err, rot_deg, xerr_norm,
                        grip_a, grip_r, flag)

        # every step: log full T matrices at DEBUG level (file only)
        logger.debug("  step %d  T_actual:", i)
        for line in _fmt_T(T_actual):
            logger.debug(line)
        logger.debug("  step %d  T_ref:", i)
        for line in _fmt_T(T_ref):
            logger.debug(line)
        logger.debug("  step %d  Xerr: [%s]  |Xerr|=%.4f",
                     i,
                     "  ".join(f"{v:+.4f}" for v in Xerr),
                     xerr_norm)

        rows_out.append([
            i, f"{t:.3f}",
            f"{p_a[0]:.6f}", f"{p_a[1]:.6f}", f"{p_a[2]:.6f}",
            f"{p_r[0]:.6f}", f"{p_r[1]:.6f}", f"{p_r[2]:.6f}",
            f"{pos_err:.6f}", f"{rot_deg:.4f}", f"{xerr_norm:.6f}",
            grip_a, grip_r,
        ])

    # ── summary ───────────────────────────────────────────────────────────────
    logger.info(SEP)
    logger.info("")
    logger.info("Summary over %d steps:", N)
    logger.info("  Max position error : %.4f m", max_pos_err)
    logger.info("  Max rotation error : %.3f deg", max_rot_err)
    logger.info("  Max |Xerr|         : %.4f  (step %d, t=%.3f s)",
                max_xerr_norm, worst_step, worst_step * dt)

    # ── write comparison CSV ──────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if out_csv is None:
        out_csv = str(Path(__file__).parent / "logs" / f"{stem}_comparison_{timestamp}.csv")

    HEADER = [
        "step", "t",
        "px_actual", "py_actual", "pz_actual",
        "px_ref",    "py_ref",    "pz_ref",
        "pos_err", "rot_err_deg", "xerr_norm",
        "grip_actual", "grip_ref",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows_out)
    logger.info("  Comparison CSV   : %s  (%d rows)", out_csv, len(rows_out))


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    robot_csv_path = sys.argv[1] if len(sys.argv) > 1 else "capstone_testA_trajectory.csv"

    perfect_config = np.array([
        0.0,   # chassis phi
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
        0.0,   # gripper
    ])

    compare(
        robot_csv      = robot_csv_path,
        perfect_config = perfect_config,
        k              = 1,
        v_max          = 0.5,
        w_max          = 1.0,
        dt             = 0.01,
    )
