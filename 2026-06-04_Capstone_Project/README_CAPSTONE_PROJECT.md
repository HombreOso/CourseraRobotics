# Capstone Project — youBot Mobile Manipulation

Modern Robotics Specialization (Coursera) — Final Project  
KUKA youBot: pick a cube from one location and place it at another, using a
closed-loop feedforward + PI controller.

---

## What the project does

The robot starts with the arm in a neutral pose.  It has to approach a 5 cm cube
sitting on the floor, pick it up, and place it at a goal position roughly 1.5 m
away.  The end-effector follows an 8-segment trajectory (move to standoff → lower
to grasp → close gripper → lift → carry → lower to goal → open gripper → lift away).
The whole thing runs as a simulation: the program generates a CSV file of robot
configurations that you load into CoppeliaSim Scene 6 to watch the robot move.

The controller is:

```
V = Ad_{X⁻¹ Xd} · V_d  +  Kp · Xerr  +  Ki · ∫ Xerr dt
```

where `X` is the actual end-effector pose (computed by FK from the current robot
state), `Xd` is the reference pose at this timestep, and `V_d` is the reference
twist to the next timestep.  The nine actuator speeds (5 arm joints + 4 mecanum
wheels) come from inverting the 6×9 mobile-manipulator Jacobian `Je`.

---

## File overview

```
capstone_full_program.py              main script — runs all four tests
milestone_1_youBot_kinematic_simulator.py   NextState: Euler integration
milestone_2_reference_trajectory_generation.py  TrajectoryGenerator: 8-segment EE path
milestone_3_feedback_control.py       FeedbackControl: V, controls, Xerr
configurations.py                     robot physical constants (r, l, w, Blist, T_b0, M_0e)
cube_config.py                        cube initial and goal poses
aa_tuning.py                          all tuning parameters in one place
logs/                                 timestamped log files from each run
```

---

## How to run

```
py capstone_full_program.py
```

This runs four tests back-to-back and writes eight CSV files (four trajectory +
four Xerr).  Load any `capstone_test*_trajectory.csv` into CoppeliaSim Scene 6.

### Tests

| Test | Controller | Initial condition | Purpose |
|---|---|---|---|
| A | Feedforward only | Perfect (Xerr = 0 at t=0) | Baseline; shows trajectory generation is correct |
| B | Feedforward only | Perturbed (+0.1 m x, +0.1 m y, +0.1 rad φ) | Confirms open-loop doesn't recover |
| C | PI (Kp = 2·I, Ki = 0.2·I) | Same perturbation | Feedback corrects the error |
| D | PI + nonlinear SE(2) chassis | Same perturbation | Faster heading correction |

### Cube positions (edit `cube_config.py` to change)

```python
T_sc_initial  at (1.0, 0.0, 0.025) m, no rotation
T_sc_goal     at (0.0, 1.5, 0.025) m, rotated –π/2 about ẑ
```

---

## Three milestones

### Milestone 1 — `NextState`

Euler-integrates the robot state one timestep:

```
θ(t+dt)  =  θ(t)  +  θ_dot · dt       (arm joints and wheel angles)
(φ, x, y)(t+dt)  =  odometry using  F · u · dt
```

The chassis odometry uses the mecanum wheel geometry matrix `F` (3×4), which maps
wheel speeds to the body-frame chassis twist `[ωz, vx, vy]`.

### Milestone 2 — `TrajectoryGenerator`

Generates a 700-ish row reference trajectory by chaining eight motion segments.
The main decision here is which interpolation method to use per segment:

- **Screw trajectory** (`mr.ScrewTrajectory`) for the big transit moves (segment 1
  moving to initial standoff, segment 5 carrying the cube to the goal).
- **Cartesian trajectory** (`mr.CartesianTrajectory`) for approach and depart
  segments (standoff → grasp and back).  Cartesian gives a straight vertical descent
  which is what you actually want when lowering the gripper onto a cube.  Screw
  interpolation would curve the path and possibly miss.

Standoff height is set 0.15 m above the grasp point (`T_ce_standoff` z = 0.168 m,
grasp point z = 0.018 m).

### Milestone 3 — `FeedbackControl`

Returns the commanded body twist `V` and the nine actuator speeds `controls`.
The Jacobian inverse uses the **damped least-squares pseudoinverse**:

```
J⁺ = Jᵀ (J Jᵀ + λ² I)⁻¹
```

instead of the standard Moore-Penrose.  See the "Problems I ran into" section for
why this matters.

---

## Tuning parameters (`aa_tuning.py`)

```python
k_p_diag_value = 2.0    # Kp = k_p * I₆
k_i_diag_value = 0.2    # Ki = k_i * I₆

# nonlinear SE(2) chassis correction (Test D only)
k_nl_phi = 1.5
k_nl_x   = 1.5
k_nl_y   = 1.5

max_speed      = inf     # actuator clamp (rad/s) — disabled; λ handles it
lambda_damping = 0.005   # damped pseudoinverse damping factor
ff_error_scale = 0.0     # feedforward blending — not used, left for experiments
```

---

## Problems I ran into and how I fixed them

### 1. Wrong trajectory interpolation for approach/depart

The original code used `mr.ScrewTrajectory` for every segment.  Screw
interpolation works great for large 6D motions but it curves the path — the tool
moves along a helical arc instead of going straight down.  For lowering a gripper
onto a cube you really want a straight line.  I switched segments 2, 4, 6, 8
to `mr.CartesianTrajectory` and the approach paths look correct in CoppeliaSim.

Also had a bug where the standoff z-height was the same as the grasp height
(0.018 m), so the standoff and grasp poses were nearly identical.  The approach
segment had basically zero length and collapsed to a single row.  Fixed by setting
the standoff z to 0.168 m (0.018 + 0.15 lift).

### 2. Test C looked identical to Test B

When I first ran the PI test, the error never went down — the Xerr plot was a
flat line, same as the feedforward test.  Turned out the gains were hardcoded
as zero in the test C call.  Classic.  Fixed by centralising gains in
`aa_tuning.py` so all tests pull from the same variables.

### 3. Chassis driving way past the reference path

In CoppeliaSim, Tests B, C, D showed the chassis
shooting 2× further than it should before (maybe) coming back.  Wheel speeds were
hitting 196 rad/s in Test A (with a perfect start!) and 143 rad/s in Test B.
Physical youBot wheels spin at maybe 15–20 rad/s in practice.

**Why it happened — Adjoint amplification**

The feedforward term `Ad_{X⁻¹ Xd} · V_d` has a cross-product inside the Adjoint:
`[p] R ω`, where `p` is the translational offset between actual and reference EE
frames.  When the robot starts off-path, this cross term amplifies the desired
velocity.  The bigger the initial error, the larger the amplification, which then
drives the chassis further off, which makes the Adjoint term even bigger.  It is
a positive feedback loop hiding inside the feedforward.

**Why it happened — Jacobian singularities**

I ran a quick diagnostic over the full trajectory and measured the minimum
singular value of `Je` at each step.  About 6% of steps had σ_min below 0.01,
with the worst case at 0.00126.  With the standard pseudoinverse, the gain at
that step is 1/0.00126 ≈ **794**.  Combine that with the Adjoint amplification
and you get 196 rad/s.

**The fix — damped least-squares pseudoinverse**

The Modern Robotics textbook (Chapter 6) discusses this.  Instead of:

```
J⁺ = (Jᵀ J)⁻¹ Jᵀ    (breaks near singularities)
```

use:

```
J⁺ = Jᵀ (J Jᵀ + λ² I)⁻¹
```

The effective gain for a singular value σ is `σ / (σ² + λ²)` instead of `1/σ`.
At the worst singularity with λ = 0.005, the gain goes from 794 down to 47.
For typical steps (σ ≈ 0.015), the gain changes by less than 10%, so tracking
accuracy is barely affected.

After this change, peak wheel speeds in Test A dropped from 196 rad/s to 27.5 rad/s
and the chassis trajectories in B, C, D look physically reasonable.

I also tried:
- **Hard `max_speed` clamp** at 12 rad/s: worked for stability but destroyed
  tracking — Test A final |Xerr| jumped from 0.27 to 1.24 because critical moments
  in the trajectory got clipped.
- **Feedforward blending** (`ff_error_scale`): blends the Adjoint-corrected
  feedforward with a plain `V_d` based on |Xerr|.  Reduces runaway but also weakens
  feedforward on perfectly on-path steps, so Test A suffered too.

The damped pseudoinverse was the cleanest fix.  I left `max_speed = inf` and
`ff_error_scale = 0.0` as defaults.

### 4. Chassis moving in wrong direction after perturbation

With the perturbed start the chassis was rotating the wrong way before correcting.
The standard SE(3) feedback expresses errors in world frame, so when the chassis
heading φ is off by 0.1 rad the proportional correction pushes along the wrong
axis.

The textbook has eq. 13.31 for a unicycle (differential-drive) robot.  The youBot
has a holonomic (mecanum) base, so the formula doesn't apply directly — it has
`tan(φ_e)/cos(φ_e)` which goes to infinity at ±90° heading error, a singularity
that doesn't exist for an omnidirectional base.

What I took from 13.31 is the key idea: **express position error in the reference
heading frame**, not the world frame.  For the holonomic case that becomes:

```
φ_e     = φ − φ_d
[x_e, y_e] = R(φ_d)ᵀ · [x − x_d, y − y_d]    (rotate error into {d} frame)

ω_corr  = −k_φ · φ_e
v_corr  = R(φ_e) · [−k_x · x_e, −k_y · y_e]  (rotate correction back to {b})
```

No singularities, three independent DOF, holonomic.  The desired chassis heading
φ_d is extracted from the reference EE pose `T_se_d` by running the FK in reverse
for the chassis component.

This is Test D and it gives the lowest final error of the four tests (|Xerr| =
0.008 vs 0.012 for Test C).

---

## Results

| Test | Final \|Xerr\| | Peak wheel speed | Notes |
|---|---|---|---|
| A — feedforward, perfect | 0.229 | 27.5 rad/s | Drift from Euler integration only |
| B — feedforward, perturbed | 0.247 | ~30 rad/s | Stays off as expected |
| C — PI, perturbed | 0.012 | ~30 rad/s | Good convergence |
| D — PI + SE(2), perturbed | **0.008** | ~30 rad/s | Fastest convergence |

Test A result before fixing the pseudoinverse was |Xerr| = 0.269 with peak speeds
of 196 rad/s.  Same trajectory, same gains — the only change was λ = 0.005.

---

## Dependencies

```
numpy
modern_robotics   (pip install modern-robotics)
```

Python 3.10+.  Run with `py` on Windows or `python3` on Linux/Mac.
