# Capstone Project — Engineering Changes Log

Modern Robotics Capstone: youBot Mobile Manipulation  
_(pick-and-place with 8-segment EE trajectory + PI feedback control)_

---

## Summary of results

| Test | Controller | Final \|Xerr\| | Max chassis x | Notes |
|---|---|---|---|---|
| A | Feedforward, perfect start | **0.229** | 0.415 m | Baseline |
| B | Feedforward, perturbed start | 0.247 | 0.672 m | Expected — no feedback |
| C | PI (Kp=2·I, Ki=0.2·I), perturbed | **0.012** | 0.717 m | Strong convergence |
| D | PI + nonlinear SE(2) chassis, perturbed | **0.008** | 0.713 m | Best convergence |

Perturbed start: chassis offset +0.1 m x, +0.1 m y, +0.1 rad φ at t = 0.

---

## Issue 1 — Trajectory generation used wrong screw type

**Symptom:** Approach/depart motions (standoff → grasp and back) traced curved
arcs instead of straight-line tool paths, which is incorrect for picking up a cube.

**Root cause:** All eight segments used `mr.ScrewTrajectory`, which interpolates
along a screw axis and curves both rotation and translation simultaneously.
Straight vertical descents should use `mr.CartesianTrajectory` (linear
interpolation of position + SLERP of orientation).

**Fix (`milestone_2_reference_trajectory_generation.py`):**
- Added `cartesian: bool` flag to `_motion_segment`.
- Segments 2, 4, 6, 8 (standoff ↔ grasp transitions) now use
  `mr.CartesianTrajectory`; segments 1, 5 keep `mr.ScrewTrajectory`.
- Corrected `T_ce_standoff_default` z-offset from 0.018 m to **0.168 m**
  (0.018 grasp height + 0.15 m lift) so the standoff is actually above the
  grasp point.
- Fixed standoff pose calculation from incorrect matrix addition to correct
  kinematic composition:
  ```
  T_se_standoff = T_sc @ T_ce_standoff   # was: T_se_grasp + [offset matrix]
  ```

---

## Issue 2 — `K_p` and `K_i` were zero for Test C

**Symptom:** Test C (PI feedback) produced the same trajectory as Test B
(feedforward only) — the error never converged.

**Root cause:** The `__main__` block hard-coded
`K_p = np.eye(6) * 0.0` and `K_i = np.eye(6) * 0.0` for Test C.

**Fix (`capstone_full_program.py` / `aa_tuning.py`):**
- Gains are now loaded from `aa_tuning.py` (`k_p_diag_value = 2.0`,
  `k_i_diag_value = 0.2`).
- All four test blocks reference the same source of truth.

---

## Issue 3 — Chassis driven far past the reference path in perturbed tests

**Symptom (observed in CoppeliaSim):** In Tests B, C, D the chassis drove 2–2.5× further
in x than in Test A.  Wheel speeds reached **143 rad/s** in Test B and
**196 rad/s** even in Test A (perfect start).

### Root cause A — Adjoint amplification (Feedforward runaway)

The feedforward term is:

```
V = Ad_{X⁻¹ Xd} · V_d  +  Kp · Xerr  +  Ki · ∫Xerr dt
```

When the robot is off-path (X ≠ Xd), the Adjoint matrix Ad_{X⁻¹ Xd} has a
cross-product coupling term **[p] R ω** where p is the translational EE offset
and ω is the angular part of V_d.  As the error grows (no feedback in Test B),
this term amplifies V_d — a **positive-feedback loop inside the feedforward
itself**.  Tests B, C, D all suffered because the initial perturbation was large
enough to trigger the runaway before the PI could react.

### Root cause B — Near-singular Jacobian (affects all tests)

Statistical analysis of the 6×9 mobile-manipulator Jacobian Je along the full
trajectory revealed:

| Statistic | σ_min value |
|---|---|
| Minimum | 0.00126 |
| 5th percentile | 0.0093 |
| Median | 0.0147 |
| Steps with σ_min < 0.01 | 43 out of 702 (6 %) |

The exact Moore-Penrose pseudoinverse has gain **1/σ_min = 794** at the most
singular step.  Combined with the Adjoint amplification, this produced wheel
speeds up to **196 rad/s** even in the on-path Test A.

Attempts to fix with `max_speed`:
- `max_speed = 12`: clipped 22 % of Test A steps → Test A |Xerr| jumped from
  0.269 to **1.244** (too tight).
- `max_speed = 20`: clipped 2.3 % of steps → Test A |Xerr| = 0.612 (still
  hurts critical trajectory moments).
- Pure `ff_error_scale` blending: reduces feedforward based on |Xerr|, but
  degrades on-path accuracy because even the perfect-start test accumulates
  small integration error that then weakens the feedforward.

### Fix — Damped least-squares pseudoinverse (Modern Robotics, Ch. 6)

Replace both `np.linalg.pinv(Je)` calls in `FeedbackControl` with:

```
J⁺ = Jᵀ (J Jᵀ + λ² I)⁻¹
```

Effect on gain `σ/(σ² + λ²)`:

| σ_min | Exact gain | Damped gain (λ=0.005) | Change |
|---|---|---|---|
| 0.00126 (most singular) | 794 | 47 | −94 % |
| 0.0093 (5th pct) | 108 | 54 | −50 % |
| 0.0147 (median) | 68 | 61 | −10 % |

Choosing **λ = 0.005** (tuned in `aa_tuning.py`):
- Barely affects the 94 % of steps with σ_min ≈ median (< 10 % gain change).
- Caps the 6 % near-singular steps from 196 rad/s → **27.5 rad/s** (peak after fix).
- Actually **improves** Test A tracking (0.269 → **0.229**) because near-singular
  over-corrections no longer shoot the arm past the target.

`max_speed` is left at `float('inf')` — the Jacobian regularisation alone is
sufficient; no hard clamp is needed.

---

## Issue 4 — Chassis heading in wrong direction (Tests B, C, D)

**Symptom (observed in CoppeliaSim):** After perturbation, the chassis base
rotated to the wrong heading before converging, making the trajectory look
physically wrong.

**Root cause:** The standard SE(3) feedback expresses chassis position error in
the **world frame**.  When heading error φ_e is non-zero, the proportional
correction Kp · Xerr drives the chassis along the world-frame axis instead of
the reference heading direction, creating the wrong-direction transient.

**Analysis — formula 13.31 (MR book) applicability:**

| Property | MR eq. 13.31 (unicycle) | YouBot chassis |
|---|---|---|
| Chassis type | Differential-drive (nonholonomic) | Mecanum (holonomic, 3 DOF) |
| Singularity in control law | `tan(φ_e)/cos(φ_e)` → ∞ at φ_e = ±π/2 | Not present |
| Directly applicable | **No** | — |

Formula 13.31 cannot be applied directly because it assumes the robot cannot
strafe.  However, its key principle — **express position error in the reference
heading frame {d} before applying feedback** — is directly transferable.

**Fix — Holonomic SE(2) nonlinear correction
(`milestone_3_feedback_control.py: nonlinear_chassis_se2_correction`):**

_Step 1_ — Error in reference frame (same as MR eq. 13.30):
```
φ_e = φ − φ_d
[x_e]   [ cos φ_d   sin φ_d ] [x − x_d]
[y_e] = [−sin φ_d   cos φ_d ] [y − y_d]
```

_Step 2_ — Holonomic correction (no singularity, 3 independent DOF):
```
ω_corr  = −k_φ · φ_e
Δv_x(d) = −k_x · x_e        (correction in reference frame {d})
Δv_y(d) = −k_y · y_e
```

_Step 3_ — Rotate correction from {d} back to chassis body frame {b}:
```
[v_bx]   [cos φ_e  −sin φ_e] [Δv_x(d)]
[v_by] = [sin φ_e   cos φ_e] [Δv_y(d)]
```

The desired chassis pose (φ_d, x_d, y_d) is obtained from the reference EE pose
via `_desired_chassis_pose(T_se_d, theta_list)`.

**Result:** Test D (PI + nonlinear SE(2)) reduces final |Xerr| from 0.012 → **0.008**
compared to Test C (PI only), and the chassis correction converges in the correct
direction in CoppeliaSim.

---

## New files and changes

| File | Change |
|---|---|
| `aa_tuning.py` | **New.** Single source of truth for all tunable parameters: `k_p_diag_value`, `k_i_diag_value`, `k_nl_phi/x/y`, `max_speed`, `lambda_damping`, `ff_error_scale`. |
| `compare_trajectory.py` | **New.** Compares actual robot CSV trajectory against the reference, reporting position error, rotation error, and |Xerr| at every step. Outputs a comparison CSV and a timestamped log file. |
| `milestone_2_reference_trajectory_generation.py` | Cartesian/screw segment selection; standoff height fix; T matrix logging; `T_ce_standoff` parameter added to `TrajectoryGenerator`. |
| `milestone_3_feedback_control.py` | `nonlinear_chassis_se2_correction()`, `_desired_chassis_pose()`, damped pseudoinverse (`lambda_damping`), `ff_error_scale` blending, detailed step logging. |
| `capstone_full_program.py` | All parameters centralised to `aa_tuning.py`; `_log_step()` added; Tests A–D with shared `_common` dict; `run_capstone` accepts `nonlinear_chassis`, `lambda_damping`, `ff_error_scale`. |
| All `_build_logger` functions | Log files now written to `logs/` subdirectory. |

---

## Key parameters (`aa_tuning.py`)

```python
k_p_diag_value = 2.0    # Kp = k_p * I₆  (proportional gain, Tests C & D)
k_i_diag_value = 0.2    # Ki = k_i * I₆  (integral gain)
k_nl_phi = 1.5          # nonlinear chassis SE(2) heading gain
k_nl_x   = 1.5          # nonlinear chassis SE(2) x gain
k_nl_y   = 1.5          # nonlinear chassis SE(2) y gain
max_speed      = inf     # actuator clamp in NextState (rad/s); inf = disabled
lambda_damping = 0.005   # damped pseudoinverse λ (MR Ch. 6)
ff_error_scale = 0.0     # feedforward Adjoint blending (disabled; prefer λ)
```

---

## CoppeliaSim output files

| File | Description |
|---|---|
| `capstone_testA_trajectory.csv` | Feedforward only, perfect start |
| `capstone_testB_trajectory.csv` | Feedforward only, perturbed start |
| `capstone_testC_trajectory.csv` | PI feedback, perturbed start |
| `capstone_testD_trajectory.csv` | PI + nonlinear SE(2) chassis, perturbed start |
| `capstone_test*_Xerr.csv` | 6D error twist history for each test |

Load any trajectory CSV into **CoppeliaSim Scene 6** to visualise.
