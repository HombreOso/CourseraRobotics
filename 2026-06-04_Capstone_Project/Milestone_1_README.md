# Milestone 1 – youBot Kinematic Simulator

First-order Euler kinematic simulator for the youBot mobile manipulator.
Produces a 13-column CSV trajectory file compatible with the CoppeliaSim
Capstone scene (Scene 6).

---

## Files

| File | Purpose |
|------|---------|
| `configurations.py` | YouBot physical constants (`r`, `l`, `w`), fixed transforms (`T_b0`, `M_0e`), screw axes `Blist`, chassis matrix `F` |
| `milestone_1_youBot_kinematic_simulator.py` | `NextState`, `simulate`, `write_csv` functions + built-in quick-test |
| `test_milestone_1.py` | 12 unit tests covering sanity checks and all three reference control scenarios |
| `cube_config.py` | Cube initial / goal SE(3) configurations |

---

## Prerequisites

Python 3.10+ with the following packages:

```
numpy
modern_robotics
```

Install with:

```bash
pip install numpy modern-robotics
```

---

## State and control vectors

**State vector — 13 elements**

| Index | Name | Unit |
|-------|------|------|
| 0 | Chassis heading φ | rad |
| 1 | Chassis x | m |
| 2 | Chassis y | m |
| 3–7 | Arm joint angles J1–J5 | rad |
| 8–11 | Wheel angles W1–W4 | rad |
| 12 | Gripper state (0 = open, 1 = closed) | — |

**Controls vector — 9 elements**

| Index | Name | Unit |
|-------|------|------|
| 0–3 | Wheel angular speeds u1–u4 | rad/s |
| 4–8 | Arm joint speeds θ̇1–θ̇5 | rad/s |

---

## Running the built-in quick test

From the `2026-06-04_Capstone_Project` directory:

```bash
python milestone_1_youBot_kinematic_simulator.py
```

This drives all four wheels forward at 10 rad/s for 1 s (100 steps × dt = 0.01 s)
and writes the trajectory to `milestone_1_test_output.csv`.

Expected console output:

```
2026-xx-xx xx:xx:xx  INFO  Log file: milestone_1_<timestamp>.log
2026-xx-xx xx:xx:xx  INFO  === Milestone 1 – youBot kinematic simulator ===
2026-xx-xx xx:xx:xx  INFO  simulate  N=100 steps  dt=0.0100 s  max_speed=12.300 rad/s
2026-xx-xx xx:xx:xx  INFO  simulate  done.  Final config: [ 0.  0.475  0.  ...]
2026-xx-xx xx:xx:xx  INFO  CSV written → milestone_1_test_output.csv  (101 rows)
2026-xx-xx xx:xx:xx  INFO  Final state: phi= 0.00000  x= 0.47500  y= 0.00000 ...
```

A timestamped log file `milestone_1_<yyyy-MM-dd_HH-mm-ss>.log` is created
alongside the script. It contains full DEBUG-level per-step output.

---

## Running the unit tests

```bash
python -m unittest test_milestone_1 -v
```

### Test groups

#### `TestNextStateSanity`

| Test | What it verifies |
|------|-----------------|
| `test_zero_controls_no_change` | All-zero controls leave the state exactly unchanged |
| `test_speed_clamping` | Controls exceeding `max_speed` are clamped before integration |
| `test_gripper_unchanged` | Gripper state (0 or 1) passes through `NextState` unmodified |

#### `TestReferenceControls` — assignment reference scenarios (1 s, 100 × 0.01 s steps)

| # | Wheel speeds `u` | Expected result | Tolerance |
|---|-----------------|-----------------|-----------|
| 1 | `( 10,  10,  10,  10)` | x = +0.475 m, y = 0, φ = 0 | ±1 mm / ±1 mrad |
| 2 | `(-10,  10, -10,  10)` | y = +0.475 m, x = 0, φ = 0 | ±1 mm / ±1 mrad |
| 3 | `(-10,  10,  10, -10)` | φ = +1.234 rad, x = 0, y = 0 | ±1 mm / ±1 mrad |

All 12 tests should report **OK**.

---

## Using `NextState` and `simulate` programmatically

```python
import numpy as np
from milestone_1_youBot_kinematic_simulator import NextState, simulate, write_csv

# Single step
config = np.zeros(13)
controls = np.array([10.0, 10.0, 10.0, 10.0,  0.0, 0.0, 0.0, 0.0, 0.0])
new_config = NextState(config, controls, dt=0.01, max_speed=12.3)

# Full trajectory
controls_seq = np.tile(controls, (100, 1))   # 100 steps
traj = simulate(config, controls_seq, dt=0.01, max_speed=12.3)

# Write CoppeliaSim-compatible CSV
write_csv(traj, "my_trajectory.csv")
```

### `NextState` parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `current_config` | `ndarray (13,)` | Current robot state |
| `controls` | `ndarray (9,)` | Wheel + joint speeds |
| `dt` | `float` | Timestep in seconds |
| `max_speed` | `float` | Speed clamp in rad/s — pass `np.inf` to disable |

---

## CSV output format

Each row contains 13 comma-separated values:

```
chassis_phi, chassis_x, chassis_y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper_state
```

Load this file in the CoppeliaSim Capstone scene (Scene 6) to visualise the
trajectory.

---

## YouBot physical constants (from `configurations.py`)

| Symbol | Value | Description |
|--------|-------|-------------|
| `r` | 0.0475 m | Wheel radius |
| `l` | 0.235 m | Half forward/backward wheelbase |
| `w` | 0.150 m | Half side-to-side wheelbase |
| `T_b0` | 4×4 SE(3) | Chassis frame {b} → arm base frame {0} |
| `M_0e` | 4×4 SE(3) | Arm base {0} → end-effector {e} at home config |
