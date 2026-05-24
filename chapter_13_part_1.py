import csv
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
    return (1.0 / (r_i * np.cos(gamma_i))) * col.reshape(1, 3)

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
    return np.array([h_i(wheel.phi, wheel.r_i, wheel.gamma_i, wheel.beta_i, wheel.x_i, wheel.y_i) for wheel in wheels]).squeeze()

if __name__ == "__main__":
    # Load the wheels from the CSV file.
    wheels = read_wheels()
    print(wheels)

    # Construct the matrix H.
    H = construct_matrix_H(wheels)
    pprint_np(label="H", arr=H)