"""
Milestone 2 – Reference Trajectory Generation
===============================================
Implements TrajectoryGenerator, which produces the eight-segment reference
trajectory for the youBot end-effector frame {e}.

Eight segments
--------------
  1. T_se_initial      → standoff above cube (initial position)
  2. Standoff (initial) → grasp pose at cube (initial position)
  3. Gripper CLOSE  (hold 0.625 s)
  4. Grasp pose        → standoff above cube (initial position)
  5. Standoff (initial) → standoff above cube (final/goal position)
  6. Standoff (final)  → grasp pose at cube (final position)
  7. Gripper OPEN   (hold 0.625 s)
  8. Grasp pose        → standoff above cube (final position)

Output row format (13 values per row)
--------------------------------------
  r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py, pz, gripper_state
"""

import csv
import logging
import math
import numpy as np
import modern_robotics as mr
from datetime import datetime
from pathlib import Path

from cube_config import T_sc_initial, T_sc_goal


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _build_logger(name: str = "milestone_2") -> logging.Logger:
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
# Internal helpers
# ---------------------------------------------------------------------------

def _se3_distance(T_start: np.ndarray, T_end: np.ndarray) -> tuple[float, float]:
    """
    Return (linear distance, rotation angle) between two SE(3) transforms.
    Used to compute segment durations automatically.
    """
    dp    = np.linalg.norm(T_end[:3, 3] - T_start[:3, 3])
    R_rel = T_start[:3, :3].T @ T_end[:3, :3]
    # Angle of rotation: arccos((trace - 1) / 2), clamped for numerical safety
    cos_angle = (np.trace(R_rel) - 1.0) / 2.0
    angle = math.acos(float(np.clip(cos_angle, -1.0, 1.0)))
    return dp, angle


def _segment_duration(
    T_start: np.ndarray,
    T_end:   np.ndarray,
    v_max:   float,
    w_max:   float,
) -> float:
    """
    Choose segment duration as max(dist/v_max, angle/w_max),
    rounded up to the nearest 0.01 s.
    """
    dp, angle = _se3_distance(T_start, T_end)
    Tf = max(dp / v_max, angle / w_max, 0.01)   # at least one step
    Tf = math.ceil(Tf / 0.01) * 0.01
    return round(Tf, 6)


def _log_T(label: str, T: np.ndarray) -> None:
    """Log a 4×4 SE(3) matrix with a label, one row per line."""
    logger.debug("  %s:", label)
    for row in T:
        logger.debug("    [ %8.4f  %8.4f  %8.4f  %8.4f ]",
                     row[0], row[1], row[2], row[3])


def _T_to_row(T: np.ndarray, gripper: int) -> list[float]:
    """Flatten an SE(3) matrix + gripper state into a 13-element row."""
    R = T[:3, :3]
    p = T[:3, 3]
    return [
        R[0, 0], R[0, 1], R[0, 2],
        R[1, 0], R[1, 1], R[1, 2],
        R[2, 0], R[2, 1], R[2, 2],
        p[0],    p[1],    p[2],
        float(gripper),
    ]


def _motion_segment(
    T_start:  np.ndarray,
    T_end:    np.ndarray,
    Tf:       float,
    k:        int,
    method:   int,
    gripper:  int,
    include_start: bool = False,
    cartesian: bool = False,
) -> list[list[float]]:
    """
    Generate rows for one motion segment.

    Parameters
    ----------
    include_start : bool
        If True, include the first (start) configuration in the output.
        Only True for the very first segment; all others omit it to avoid
        duplicate rows at segment boundaries.
    cartesian : bool
        If True, use mr.CartesianTrajectory (translation and rotation
        interpolated separately → straight-line tool-tip path).
        If False (default), use mr.ScrewTrajectory (smooth screw motion).

        Use cartesian=True for straight vertical approach/depart segments
        (standoff → grasp and grasp → standoff) so the gripper descends
        and ascends in a straight line and cannot collide with the cube sides.
        Use cartesian=False for large sweeping motions between distant poses.
    """
    dt   = 0.01 / k
    N    = max(2, round(Tf / dt) + 1)

    if cartesian:
        traj = mr.CartesianTrajectory(T_start, T_end, Tf, N, method)
    else:
        traj = mr.ScrewTrajectory(T_start, T_end, Tf, N, method)

    start_idx = 0 if include_start else 1
    return [_T_to_row(T, gripper) for T in traj[start_idx:]]


def _gripper_segment(
    T_hold:  np.ndarray,
    gripper: int,
    k:       int,
    duration: float = 0.625,
) -> list[list[float]]:
    """
    Hold a fixed pose for `duration` seconds while the gripper opens/closes.
    The first row is omitted (boundary with the previous segment).
    """
    dt = 0.01 / k
    N  = max(1, round(duration / dt))
    row = _T_to_row(T_hold, gripper)
    return [row] * N


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def TrajectoryGenerator(
    T_se_initial:  np.ndarray,
    T_sc_initial:  np.ndarray,
    T_sc_final:    np.ndarray,
    T_ce_grasp:    np.ndarray,
    T_ce_standoff: np.ndarray = None,
    k:             int   = 1,
    v_max:         float = 0.5,
    w_max:         float = 1.0,
    method:        int   = 5,
) -> list[list[float]]:
    """
    Generate the eight-segment reference trajectory for the end-effector.

    Parameters
    ----------
    T_se_initial  : SE(3) – initial end-effector pose in space frame {s}
    T_sc_initial  : SE(3) – initial cube pose in {s}
    T_sc_final    : SE(3) – desired final cube pose in {s}
    T_ce_grasp    : SE(3) – grasp pose of {e} relative to cube frame {c}
    T_ce_standoff : SE(3) – standoff pose of {e} relative to {c}
    k             : int   – reference configs per 0.01 s (≥ 1)
    v_max         : float – max linear speed (m/s) for auto-duration (default 0.5)
    w_max         : float – max angular speed (rad/s) for auto-duration (default 1.0)
    method        : int   – time-scaling: 3 = cubic, 5 = quintic (default 5)

    Returns
    -------
    trajectory : list of N×13 rows
        Each row: [r11,r12,r13, r21,r22,r23, r31,r32,r33, px,py,pz, gripper]
    """
    # ------------------------------------------------------------------
    # Derived key poses  (all in space frame {s})
    # ------------------------------------------------------------------
    if T_ce_standoff is None:
        T_ce_standoff = T_ce_standoff_default

    T_se_standoff_i = T_sc_initial @ T_ce_standoff   # standoff above initial cube
    T_se_grasp_i    = T_sc_initial @ T_ce_grasp       # grasp at initial cube
    T_se_standoff_f = T_sc_final   @ T_ce_standoff   # standoff above final cube
    T_se_grasp_f    = T_sc_final   @ T_ce_grasp       # grasp at final cube

    logger.debug("=== Input transforms ===")
    _log_T("T_se_initial  (EE start, space frame)",    T_se_initial)
    _log_T("T_sc_initial  (cube initial, space frame)", T_sc_initial)
    _log_T("T_sc_final    (cube goal, space frame)",    T_sc_final)
    _log_T("T_ce_grasp    (grasp relative to cube)",    T_ce_grasp)
    _log_T("T_ce_standoff (standoff relative to cube)", T_ce_standoff)

    logger.debug("=== Derived waypoints (T_se in space frame) ===")
    _log_T("T_se_standoff_i  (standoff above initial cube)", T_se_standoff_i)
    _log_T("T_se_grasp_i     (grasp at initial cube)",       T_se_grasp_i)
    _log_T("T_se_standoff_f  (standoff above final cube)",   T_se_standoff_f)
    _log_T("T_se_grasp_f     (grasp at final cube)",         T_se_grasp_f)

    def dur(A, B):
        return _segment_duration(A, B, v_max, w_max)

    # cartesian=True for straight-line approach/depart (avoids cube collision);
    # cartesian=False (ScrewTrajectory) for larger sweeping motions.
    segments_meta = [
        # name,                              T_start,          T_end,             Tf,                                    grip, cartesian
        ("Seg 1 – initial → standoff_i",    T_se_initial,     T_se_standoff_i,  dur(T_se_initial,    T_se_standoff_i), 0,    False),
        ("Seg 2 – standoff_i → grasp_i",    T_se_standoff_i,  T_se_grasp_i,     dur(T_se_standoff_i, T_se_grasp_i),   0,    True ),
        ("Seg 3 – gripper CLOSE (0.625 s)", None,             None,              0.625,                                1,    False),
        ("Seg 4 – grasp_i → standoff_i",    T_se_grasp_i,     T_se_standoff_i,  dur(T_se_grasp_i,    T_se_standoff_i),1,    True ),
        ("Seg 5 – standoff_i → standoff_f", T_se_standoff_i,  T_se_standoff_f,  dur(T_se_standoff_i, T_se_standoff_f),1,    False),
        ("Seg 6 – standoff_f → grasp_f",    T_se_standoff_f,  T_se_grasp_f,     dur(T_se_standoff_f, T_se_grasp_f),  1,    True ),
        ("Seg 7 – gripper OPEN  (0.625 s)", None,             None,              0.625,                                0,    False),
        ("Seg 8 – grasp_f → standoff_f",    T_se_grasp_f,     T_se_standoff_f,  dur(T_se_grasp_f,    T_se_standoff_f),0,    True ),
    ]

    logger.info("TrajectoryGenerator  k=%d  v_max=%.2f m/s  w_max=%.2f rad/s  method=%d",
                k, v_max, w_max, method)

    # Gripper segments (3 & 7) are identified by T_start=None
    gripper_indices = {2, 6}

    trajectory: list[list[float]] = []

    for idx, (name, T_start, T_end, Tf, grip, use_cartesian) in enumerate(segments_meta):
        is_gripper = idx in gripper_indices
        is_first   = (idx == 0)

        if is_gripper:
            T_hold = T_se_grasp_i if idx == 2 else T_se_grasp_f
            rows   = _gripper_segment(T_hold, grip, k, duration=Tf)
        else:
            rows = _motion_segment(T_start, T_end, Tf, k, method, grip,
                                   include_start=is_first, cartesian=use_cartesian)

        traj_type = "cartesian" if (not is_gripper and use_cartesian) else ("screw" if not is_gripper else "hold")
        logger.info("  %-38s  Tf=%.3f s  rows=%4d  [%s]", name, Tf, len(rows), traj_type)

        if not is_gripper:
            T_log_start = np.eye(4); T_log_start[:3, :3] = np.array(rows[0][:9]).reshape(3, 3);  T_log_start[:3, 3] = rows[0][9:12]
            T_log_end   = np.eye(4); T_log_end[:3,   :3] = np.array(rows[-1][:9]).reshape(3, 3); T_log_end[:3,   3] = rows[-1][9:12]
            _log_T(f"    T_se start", T_log_start)
            _log_T(f"    T_se end  ", T_log_end)

        trajectory.extend(rows)

    logger.info("TrajectoryGenerator  total rows = %d", len(trajectory))
    return trajectory


def write_trajectory_csv(trajectory: list[list[float]], filepath: str) -> None:
    """Write the trajectory to a CSV file (no header)."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        for row in trajectory:
            writer.writerow([f"{v:.6f}" for v in row])
    logger.info("CSV written → %s  (%d rows)", filepath, len(trajectory))


# ---------------------------------------------------------------------------
# Default end-effector poses relative to cube frame {c}
# ---------------------------------------------------------------------------

# Grasp: end-effector approaches from below the cube's +z (tilted down to grab)
# Rotation: flip z-axis of {e} to point into the cube top face
T_ce_grasp_default = np.array([
    [ -1,  0,  0,  0    ],
    [ 0,  1,  0,  0    ],
    [0,  0,  -1,  0.013    ],
    [ 0,  0,  0,  1    ],
], dtype=float)

# Standoff: same orientation, lifted 0.15 m above the grasp point (0.018 + 0.15 = 0.168 m)
T_ce_standoff_default = np.array([
    [ -1,  0,  0,  0     ],
    [  0,  1,  0,  0     ],
    [  0,  0, -1,  0.168 ],
    [  0,  0,  0,  1     ],
], dtype=float)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("=== Milestone 2 – Reference Trajectory Generation ===")

    # Initial end-effector pose (arbitrary starting position above the robot)
    T_se_initial = np.array([
        [ 0,  0,  1,  0   ],
        [ 0,  1,  0,  0   ],
        [-1,  0,  0,  0.5 ],
        [ 0,  0,  0,  1   ],
    ], dtype=float)

    traj = TrajectoryGenerator(
        T_se_initial   = T_se_initial,
        T_sc_initial   = T_sc_initial,
        T_sc_final     = T_sc_goal,
        T_ce_grasp     = T_ce_grasp_default,
        T_ce_standoff  = T_ce_standoff_default,
        k              = 1,
        v_max          = 0.5,
        w_max          = 1.0,
        method         = 5,
    )

    out_path = "milestone_2_trajectory.csv"
    write_trajectory_csv(traj, out_path)

    logger.info("Done.  %d trajectory rows written.", len(traj))
