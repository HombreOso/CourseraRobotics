import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import modern_robotics as mr
from scipy.optimize import linprog
from dataclasses import dataclass

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
