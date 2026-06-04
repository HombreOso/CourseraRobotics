import numpy as np

# ---------------------------------------------------------------------------
# YouBot physical constants
# ---------------------------------------------------------------------------

# Wheel / chassis geometry
r = 0.0475          # wheel radius (m)
l = 0.47 / 2        # half forward-backward wheelbase (m)  → 0.235
w = 0.30 / 2        # half side-to-side wheelbase (m)      → 0.15

# ---------------------------------------------------------------------------
# Fixed transforms
# ---------------------------------------------------------------------------

# SE(3): chassis body frame {b} → arm base frame {0}
T_b0 = np.array([
    [1, 0, 0, 0.1662],
    [0, 1, 0, 0     ],
    [0, 0, 1, 0.0026],
    [0, 0, 0, 1     ],
], dtype=float)

# SE(3): arm base frame {0} → end-effector frame {e} at home config (all θ = 0)
M_0e = np.array([
    [1, 0, 0, 0.033 ],
    [0, 1, 0, 0     ],
    [0, 0, 1, 0.6546],
    [0, 0, 0, 1     ],
], dtype=float)

# ---------------------------------------------------------------------------
# Screw axes in the end-effector frame {e} at the home configuration
# Columns of Blist correspond to joints 1–5.
# Each screw axis: [ω_x, ω_y, ω_z, v_x, v_y, v_z]
# ---------------------------------------------------------------------------
B1 = np.array([ 0,  0,  1,    0,  0.033, 0])
B2 = np.array([ 0, -1,  0, -0.5076, 0,   0])
B3 = np.array([ 0, -1,  0, -0.3526, 0,   0])
B4 = np.array([ 0, -1,  0, -0.2176, 0,   0])
B5 = np.array([ 0,  0,  1,    0,    0,   0])

Blist = np.column_stack([B1, B2, B3, B4, B5])   # shape (6, 5)

# ---------------------------------------------------------------------------
# Chassis velocity matrix F for the 4-wheel omnidirectional (mecanum) base.
#
# Maps wheel speeds u = [u1, u2, u3, u4] to the body-frame chassis twist
# [ω_z, v_x, v_y].
#
#        r/4 * [ -1/(l+w)   1/(l+w)   1/(l+w)  -1/(l+w) ]
#  F  =        [  1         1         1         1        ]
#              [ -1         1        -1         1        ]
# ---------------------------------------------------------------------------
F = (r / 4) * np.array([
    [-(1/(l+w)),  (1/(l+w)),  (1/(l+w)), -(1/(l+w))],
    [  1,          1,          1,          1        ],
    [ -1,          1,         -1,          1        ],
], dtype=float)   # shape (3, 4)

# 6-row version used for the mobile-base Jacobian contribution
def F6() -> np.ndarray:
    """Return the (6 x 4) extended chassis velocity matrix."""
    zeros = np.zeros((1, 4))
    return np.vstack([zeros, zeros, F, zeros])   # shape (6, 4)

# ---------------------------------------------------------------------------
# Gripper geometry (meters)
# ---------------------------------------------------------------------------
d1_min = 0.02   # minimum opening distance between finger tips
d1_max = 0.07   # maximum opening distance between finger tips
d2     = 0.035  # interior length of the fingers
d3     = 0.043  # distance from finger base to end-effector frame {e}

