import csv
import math
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import modern_robotics as mr
from scipy.optimize import linprog
from dataclasses import dataclass
from np_utils import format_numpy_compact, pprint_np

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Wheel:
    phi: float
    r_i: float
    gamma_i: float
    beta_i: float
    x_i: float
    y_i: float

# ---------------------------------------------------------------------------

def h_i(phi: float, r_i: float, gamma_i: float, beta_i: float,
        x_i: float, y_i: float) -> np.ndarray:
    """
    Equation (13.6): wheel kinematic constraint row vector h_i(phi).

    Returns a (1, 3) row vector (transposed column vector):
        1 / (r_i * cos(gamma_i)) * [x_i*sin(b+g) - y_i*cos(b+g),
                                     cos(b+g+phi),
                                     sin(b+g+phi)]
    where b = beta_i, g = gamma_i.
    """
    bg = beta_i + gamma_i
    col = np.array([
        x_i * np.sin(bg) - y_i * np.cos(bg),
        np.cos(bg + phi),
        np.sin(bg + phi),
    ])
    return (1.0 / (r_i * np.cos(gamma_i))) * col.transpose()

CSV_PATH = "chapter_13_wheels_configs.csv"

def read_wheels(path: str = CSV_PATH) -> list[Wheel]:
    """Read wheel configurations from a CSV file.

    Numeric cells may contain arithmetic expressions (e.g. '3*3.14/4'),
    which are evaluated safely with only basic math operations permitted.
    """
    wheels = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wheels.append(Wheel(
                phi=eval(row["phi_radians"]),
                r_i=eval(row["r_i"]),
                gamma_i=eval(row["gamma_i_radians"]),
                beta_i=eval(row["beta_i_radians"]),
                x_i=eval(row["x_i"]),
                y_i=eval(row["y_i"]),
            ))
    return wheels

def construct_matrix_H(wheels: list[Wheel]) -> np.ndarray:
    """
    Construct the matrix H from the list of wheels.
    """
    return np.array([h_i(wheel.phi, wheel.r_i, wheel.gamma_i, wheel.beta_i, wheel.x_i, wheel.y_i) for wheel in wheels])

if __name__ == "__main__":
    # Load the wheels from the CSV file.
    wheels = read_wheels()
    print(wheels)

    # Construct the matrix H.
    H = construct_matrix_H(wheels)
    pprint_np(label="H", arr=H)

    twist_body_frame = np.array([1,0,0])

    controls_vector = H @ twist_body_frame
    pprint_np(label="controls_vector", arr=controls_vector)

    another_twist_body_frame = np.array([1,2,3])
    another_controls_vector = H @ another_twist_body_frame
    pprint_np(label="another_controls_vector", arr=another_controls_vector)

    u1_max = 10
    u1_min = -10
    u2_max = 10
    u2_min = -10
    u3_max = 10
    u3_min = -10
    u4_max = 10
    u4_min = -10



    u1 = np.clip(controls_vector[0], u1_min, u1_max)
    u2 = np.clip(controls_vector[1], u2_min, u2_max)
    u3 = np.clip(controls_vector[2], u3_min, u3_max)
    u4 = np.clip(controls_vector[3], u4_min, u4_max)

    pprint_np(label="u1", arr=u1)
    pprint_np(label="u2", arr=u2)
    pprint_np(label="u3", arr=u3)
    pprint_np(label="u4", arr=u4)

    # ---------------------------------------------------------------------------
    # Solve:  -10 <= [-11.313708, 5.6568542, 0.] @ [0, v_x, v_y] <= 10
    # ---------------------------------------------------------------------------
    row = np.array([-11.313708,2.8284271,-2.8284271])
    u_min, u_max = -10.0, 10.0

    # Dot product with [0, v_x, v_y]:
    #   row[0]*0 + row[1]*v_x + row[2]*v_y  =  5.6568542 * v_x
    # row[2] == 0, so v_y is unconstrained by this row.
    coeff_vx = row[1]   # 5.6568542
    coeff_vy = row[2]   # 0.0  → v_y does not appear

    # Solve bounds on v_x:  u_min <= coeff_vx * v_x <= u_max
    vx_min = u_min / coeff_vx
    vx_max = u_max / coeff_vx

    print(f"\nConstraint row : {row}")
    print(f"Twist          : [0, v_x, v_y]")
    print(f"Effective expr : {coeff_vx:.7f} * v_x  (v_y coefficient = {coeff_vy})")
    print(f"Feasible v_x   : [{vx_min:.6f}, {vx_max:.6f}]")
    print(f"Feasible v_y   : unconstrained (coefficient is zero)")

    # ---------------------------------------------------------------------------
    # Q3: Maximum linear chassis speed  sqrt(v_x^2 + v_y^2)
    #     subject to  -10 <= H @ [0, v_x, v_y] <= 10   (omega_z = 0)
    # ---------------------------------------------------------------------------
    # Strategy: parametrise the direction of motion by angle theta.
    #   v_x = s * cos(theta),  v_y = s * sin(theta)
    #   Wheel speeds: u = s * (H @ [0, cos(theta), sin(theta)])
    #   |u_i| <= 10  =>  s <= 10 / |H_i @ dir|  for every wheel i
    #   s_max(theta) = min_i( 10 / |H_i @ dir| )   (ignoring zero coefficients)
    #   Answer = max over theta of s_max(theta)

    u_limit = 10.0
    N = 100_000
    thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # Build direction matrix (N, 3) with first column always 0 (omega_z = 0)
    directions = np.column_stack([
        np.zeros(N),
        np.cos(thetas),
        np.sin(thetas),
    ])

    wheel_coeffs_all = H @ directions.T     # (n_wheels, N): speed per unit s per wheel

    with np.errstate(divide="ignore", invalid="ignore"):
        bounds_all = np.where(
            np.abs(wheel_coeffs_all) > 1e-9,
            u_limit / np.abs(wheel_coeffs_all),
            np.inf,
        )

    # Tightest constraint across all wheels for each direction
    s_max_all = np.min(bounds_all, axis=0)  # (N,)

    best_idx   = np.argmax(s_max_all)
    max_speed  = s_max_all[best_idx]
    best_theta = thetas[best_idx]

    print(f"\n--- Q3: Maximum linear chassis speed ---")
    print(f"H (wheel constraint matrix):\n{H}")
    print(f"Third column of H (v_y sensitivity): {H[:, 2]}")
    print(f"Best direction : theta = {np.degrees(best_theta):.2f} deg "
          f"(vx={np.cos(best_theta):.4f}, vy={np.sin(best_theta):.4f})")
    print(f"Max speed sqrt(vx^2 + vy^2) = {max_speed:.4f}")
    if np.isinf(max_speed):
        print("  NOTE: The v_y direction is uncontrolled (all H rows have ~zero v_y sensitivity).")
        print("    Max speed in the pure v_x direction: "
              f"{u_limit / np.max(np.abs(H[:, 1])):.4f}")

    # ---------------------------------------------------------------------------
    # Analysis of the two-constraint system (omega_z = 0):
    #   -10 <=  2.8284271*vx - 2.8284271*vy <= 10   (A)
    #   -10 <= -2.8284271*vx + 2.8284271*vy <= 10   (B)
    # ---------------------------------------------------------------------------
    # Observation: (B) == -(A), so both reduce to ONE constraint:
    #   |2.8284271 * (vx - vy)| <= 10
    #   |vx - vy| <= 10 / 2.8284271
    #
    # The feasible region is an infinite diagonal strip in (vx, vy) space.
    # vx, vy individually are UNBOUNDED; only their difference is constrained.
    # Max speed sqrt(vx^2+vy^2) is also UNBOUNDED (take vx=vy -> infinity).
    # ---------------------------------------------------------------------------

    c_A = np.array([ 2.8284271, -2.8284271])   # coefficients [vx, vy] for row A
    c_B = np.array([-2.8284271,  2.8284271])   # coefficients [vx, vy] for row B
    u_lim = 10.0

    # Verify the symmetry analytically
    are_negatives = np.allclose(c_A, -c_B)
    diff_bound = u_lim / abs(c_A[0])           # |vx - vy| <= this value

    print(f"\n--- Two-constraint analysis ---")
    print(f"Row A coefficients : {c_A}")
    print(f"Row B coefficients : {c_B}")
    print(f"Row B == -Row A    : {are_negatives}  (redundant constraint)")
    print(f"Effective constraint: |vx - vy| <= {diff_bound:.6f}")

    # Use linprog to formally confirm bounds (scipy convention: minimize c @ x)
    # Constraints in linprog form: A_ub @ x <= b_ub
    A_ub = np.array([
         c_A,   #  c_A @ [vx,vy] <=  u_lim
        -c_A,   # -c_A @ [vx,vy] <=  u_lim  (lower bound of A)
         c_B,   #  c_B @ [vx,vy] <=  u_lim
        -c_B,   # -c_B @ [vx,vy] <=  u_lim
    ])
    b_ub = np.array([u_lim, u_lim, u_lim, u_lim])

    for label, obj in [("vx_max", [-1, 0]), ("vx_min", [1, 0]),
                       ("vy_max", [0, -1]), ("vy_min", [0,  1])]:
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=[(None,None),(None,None)])
        if res.status == 3:
            val = "unbounded"
        elif res.status == 0:
            val = f"{-res.fun:.6f}" if obj[0] < 0 or obj[1] < 0 else f"{res.fun:.6f}"
        else:
            val = f"status={res.status}"
        print(f"  {label:8s}: {val}")

    print(f"  max speed sqrt(vx^2+vy^2): unbounded "
          f"(feasible set is infinite along vx=vy direction)")