# Milestone 2 – Reference Trajectory Generation

Implements `TrajectoryGenerator`, which produces an eight-segment reference
trajectory for the youBot end-effector frame {e} expressed in the space
frame {s}.  The output is a CSV file consumed by CoppeliaSim (Scene 6) and
later by the feedback controller in Milestone 3.

---

## Files

| File | Purpose |
|------|---------|
| `milestone_2_reference_trajectory_generation.py` | `TrajectoryGenerator`, `write_trajectory_csv`, default `T_ce_grasp` and `T_ce_standoff` |
| `configurations.py` | YouBot constants: `T_b0`, `M_0e`, `Blist` (needed to compute `T_se_initial` from a real robot state) |
| `cube_config.py` | `T_sc_initial`, `T_sc_goal` — cube poses in {s} taken from the assignment spec |

---

## Where each input comes from

### `T_se_initial` — initial end-effector pose in {s}

This is the **only input that is not directly given by the assignment**.
It must be computed from the actual starting robot configuration
`(φ, x, y, J1…J5)` using forward kinematics:

```
T_sb  =  SE(3) chassis pose in {s}  (from Milestone 1 odometry)

         ⎡ cos φ  -sin φ  0   x ⎤
T_sb  =  ⎢ sin φ   cos φ  0   y ⎥
         ⎢  0       0     1  0.0963 ⎥
         ⎣  0       0     0   1 ⎦

T_se  =  T_sb  @  T_b0  @  FKinBody(M_0e, Blist, theta_list)
```

| Symbol | Source | Description |
|--------|--------|-------------|
| `T_sb` | Constructed from `(φ, x, y)` | Space-to-body chassis transform; z = 0.0963 m (chassis height) |
| `T_b0` | `configurations.py` | Fixed transform from chassis {b} to arm base {0}; translation (0.1662, 0, 0.0026) m |
| `M_0e` | `configurations.py` | Home config of {e} relative to arm base {0}; translation (0.033, 0, 0.6546) m |
| `Blist` | `configurations.py` | 6×5 screw axes of the 5 joints in end-effector frame {e} |
| `theta_list` | Robot state | Current 5 arm joint angles J1–J5 |

**In the demo `__main__` block**, `T_se_initial` is set to an explicit
placeholder that places the end-effector 0.5 m above the base with the
gripper pointing in the −z direction:

```python
T_se_initial = np.array([
    [ 0,  0,  1,  0   ],   # x_e = z_s
    [ 0,  1,  0,  0   ],   # y_e = y_s
    [-1,  0,  0,  0.5 ],   # z_e = -x_s,  origin 0.5 m above floor
    [ 0,  0,  0,  1   ],
])
```

For a real run, replace this with the FK computation above using the
robot's actual starting joint angles.

---

### `T_sc_initial` and `T_sc_final` — cube poses in {s}

Both come from **`cube_config.py`**, which encodes values stated in the
assignment specification:

| Variable | (x, y, z) m | Rotation | Source |
|----------|-------------|----------|--------|
| `T_sc_initial` | (1, 0, 0.025) | Identity (aligned with {s}) | Assignment spec |
| `T_sc_goal` | (0, −1, 0.025) | −π/2 about ẑ_s | Assignment spec |

The z-offset of 0.025 m places the cube centre at half the cube height
(5 cm cube → 2.5 cm above the floor).

---

### `T_ce_grasp` — grasp pose of {e} relative to cube frame {c}

Defined in the module as `T_ce_grasp_default`.  It specifies how the
end-effector must be oriented **at the moment of gripping**.

```
         ⎡  0   0   1   0 ⎤
T_ce  =  ⎢  0   1   0   0 ⎥
         ⎢ -1   0   0   0 ⎥
         ⎣  0   0   0   1 ⎦
```

**Explanation of the rotation:**

The cube frame {c} has its z-axis pointing up.  The end-effector frame
{e} has its z-axis pointing *out of the gripper* (the approach direction).
To approach from above, we need z_e to point *downward* in {c}, i.e.
`z_e = −z_c`.  The columns of the rotation matrix are the axes of {e}
expressed in {c}:

```
x_e = z_c   (first column)
y_e = y_c   (second column, unchanged)
z_e = -x_c  (third column, the approach direction points into cube top)
```

The translation is zero: the gripper origin meets the cube origin at the
grasp point (the cube is centred between the fingers).

---

### `T_ce_standoff` — standoff pose of {e} above the cube

Defined in the module as `T_ce_standoff_default`.  Same orientation as
`T_ce_grasp`; the translation adds **0.15 m** along the cube's z-axis
(i.e. 15 cm straight above the cube centre):

```
         ⎡  0   0   1   0    ⎤
T_ce  =  ⎢  0   1   0   0    ⎥
         ⎢ -1   0   0   0.15 ⎥
         ⎣  0   0   0   1    ⎦
```

This guarantees the robot clears the cube's top face during approach and
departure without collision.

---

## How segment durations are computed

No durations are hard-coded.  For each motion segment the function calls
`_segment_duration(T_start, T_end, v_max, w_max)`:

```
dist  = ‖ p_end − p_start ‖          (Euclidean translation distance)
angle = arccos( (tr(R_rel) − 1) / 2 ) (rotation angle of R_start^T R_end)

Tf = max( dist / v_max,  angle / w_max,  0.01 )
Tf = ceil( Tf / 0.01 ) × 0.01         (round up to nearest 10 ms)
```

Default limits: `v_max = 0.5 m/s`, `w_max = 1.0 rad/s`.

Gripper segments (open/close) are always **0.625 s** regardless of
distance (the robot holds still while the fingers move).

---

## The eight trajectory segments

All waypoints in the space frame {s} are computed by composing cube and
relative-frame transforms:

```
T_se_standoff_i = T_sc_initial @ T_ce_standoff
T_se_grasp_i    = T_sc_initial @ T_ce_grasp
T_se_standoff_f = T_sc_final   @ T_ce_standoff
T_se_grasp_f    = T_sc_final   @ T_ce_grasp
```

| # | From → To | Gripper | Duration |
|---|-----------|---------|---------|
| 1 | `T_se_initial` → `T_se_standoff_i` | 0 (open) | auto |
| 2 | `T_se_standoff_i` → `T_se_grasp_i` | 0 (open) | auto |
| 3 | Hold `T_se_grasp_i` (gripper **closes**) | 1 (closed) | 0.625 s |
| 4 | `T_se_grasp_i` → `T_se_standoff_i` | 1 (closed) | auto |
| 5 | `T_se_standoff_i` → `T_se_standoff_f` | 1 (closed) | auto |
| 6 | `T_se_standoff_f` → `T_se_grasp_f` | 1 (closed) | auto |
| 7 | Hold `T_se_grasp_f` (gripper **opens**) | 0 (open) | 0.625 s |
| 8 | `T_se_grasp_f` → `T_se_standoff_f` | 0 (open) | auto |

Segments 1–8 are concatenated without duplicating boundary rows (each
segment's first configuration is skipped because it is identical to the
last configuration of the previous segment).

Each motion segment uses `mr.ScrewTrajectory` (quintic time scaling,
`method=5`) for smooth acceleration and deceleration at every endpoint.

---

## CSV output format

Each row contains **13 comma-separated values**:

```
r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py, pz, gripper_state
```

These are the top three rows of the 4×4 SE(3) matrix `T_se` plus the
gripper state (0 = open, 1 = closed).  Load this file in CoppeliaSim
Scene 6 (the CSV Animation youBot scene) to visualise the trajectory.

---

## Running the demo

From the `2026-06-04_Capstone_Project` directory:

```bash
python milestone_2_reference_trajectory_generation.py
```

Expected console output (durations depend on `v_max`/`w_max`):

```
INFO  Log file: milestone_2_<timestamp>.log
INFO  === Milestone 2 – Reference Trajectory Generation ===
INFO  TrajectoryGenerator  k=1  v_max=0.50 m/s  w_max=1.00 rad/s  method=5
INFO    Seg 1 – initial → standoff_i         Tf=2.110 s  rows=212
INFO    Seg 2 – standoff_i → grasp_i         Tf=0.300 s  rows=30
INFO    Seg 3 – gripper CLOSE (0.625 s)      Tf=0.625 s  rows=62
INFO    Seg 4 – grasp_i → standoff_i         Tf=0.300 s  rows=30
INFO    Seg 5 – standoff_i → standoff_f      Tf=2.830 s  rows=283
INFO    Seg 6 – standoff_f → grasp_f         Tf=0.300 s  rows=30
INFO    Seg 7 – gripper OPEN  (0.625 s)      Tf=0.625 s  rows=62
INFO    Seg 8 – grasp_f → standoff_f         Tf=0.300 s  rows=30
INFO  TrajectoryGenerator  total rows = 739
INFO  CSV written → milestone_2_trajectory.csv  (739 rows)
```

A timestamped log file `milestone_2_<yyyy-MM-dd_HH-mm-ss>.log` is written
alongside the script containing the same output.

---

## Using `TrajectoryGenerator` from another module

```python
import numpy as np
import modern_robotics as mr
from configurations import T_b0, M_0e, Blist
from cube_config import T_sc_initial, T_sc_goal
from milestone_2_reference_trajectory_generation import (
    TrajectoryGenerator, write_trajectory_csv,
    T_ce_grasp_default, T_ce_standoff_default,
)

# --- Compute T_se_initial from the real robot starting state ---
phi, x, y = 0.0, 0.0, 0.0          # chassis pose
theta_list = np.array([0, -0.2, -1.6, -1.57, 0])  # arm joint angles

T_sb = np.array([
    [np.cos(phi), -np.sin(phi), 0, x     ],
    [np.sin(phi),  np.cos(phi), 0, y     ],
    [0,            0,           1, 0.0963],
    [0,            0,           0, 1     ],
])
T_se_initial = T_sb @ T_b0 @ mr.FKinBody(M_0e, Blist, theta_list)

# --- Generate trajectory ---
traj = TrajectoryGenerator(
    T_se_initial   = T_se_initial,
    T_sc_initial   = T_sc_initial,
    T_sc_final     = T_sc_goal,
    T_ce_grasp     = T_ce_grasp_default,
    T_ce_standoff  = T_ce_standoff_default,
    k              = 1,       # 1 config per 0.01 s
    v_max          = 0.5,     # m/s
    w_max          = 1.0,     # rad/s
    method         = 5,       # quintic time scaling
)

write_trajectory_csv(traj, "my_trajectory.csv")
```

### `TrajectoryGenerator` parameter reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `T_se_initial` | `ndarray (4,4)` | — | Initial end-effector pose in {s} |
| `T_sc_initial` | `ndarray (4,4)` | — | Initial cube pose in {s} |
| `T_sc_final` | `ndarray (4,4)` | — | Goal cube pose in {s} |
| `T_ce_grasp` | `ndarray (4,4)` | — | Grasp pose of {e} in cube frame {c} |
| `T_ce_standoff` | `ndarray (4,4)` | — | Standoff pose of {e} in cube frame {c} |
| `k` | `int` | `1` | Reference configs per 0.01 s (controller frequency multiplier) |
| `v_max` | `float` | `0.5` | Max linear speed (m/s) used for auto-duration |
| `w_max` | `float` | `1.0` | Max angular speed (rad/s) used for auto-duration |
| `method` | `int` | `5` | Time-scaling: `3` = cubic, `5` = quintic |
