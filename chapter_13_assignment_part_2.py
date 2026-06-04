import numpy as np
import csv
import time
import math
import modern_robotics as mr

from np_utils import format_numpy_compact, pprint_np

# ---------------------------------------------------------------------------

def H_phi(phi: float = 0, r: float = 0.5, d: float = 2) -> np.ndarray:
    """
    Wheel-chassis kinematic matrix H(phi) for a differential-drive robot.

    Parameters
    ----------
    phi : float  – chassis heading angle (radians)
    r   : float  – wheel radius
    d   : float  – half the wheelbase (distance from chassis centre to each wheel)

    Returns
    -------
    H : np.ndarray, shape (5, 2)
        Maps wheel speeds [u_R, u_L] to the chassis twist components
        [omega_z, v_x, v_y, ...].

        H = [ -r/2d       r/2d      ]
            [  r/2*cos(φ)  r/2*cos(φ)]
            [  r/2*sin(φ)  r/2*sin(φ)]
            [  1           0         ]
            [  0           1         ]
    """
    return np.array([
        [-r / (2*d),          r / (2*d)         ],
        [ (r/2) * np.cos(phi), (r/2) * np.cos(phi)],
        [ (r/2) * np.sin(phi), (r/2) * np.sin(phi)],
        [ 1.0,                 0.0               ],
        [ 0.0,                 1.0               ],
    ])

# ---------------------------------------------------------------------------

def F(r: float, d: float) -> np.ndarray:
    """
    Chassis velocity matrix F for a differential-drive robot.

    F = r * [ -1/(2d)   1/(2d) ]
            [  1/2      1/2    ]
            [  0        0      ]

    Maps wheel speeds [u_R, u_L] to body-frame chassis twist [omega_z, v_x, v_y].

    Parameters
    ----------
    r : float – wheel radius
    d : float – half the wheelbase
    """
    return r * np.array([
        [-1 / (2*d),  1 / (2*d)],
        [ 1/2,        1/2      ],
        [ 0.0,        0.0      ],
    ])

def F6(r: float, d: float) -> np.ndarray:
    """
    6D chassis velocity matrix F_6 for a differential-drive robot.

    Extends F (3×m) to the full 6D body twist by padding with zero rows:

        F_6 = [ 0_m ]   <- omega_x  (zero: chassis doesn't roll)
              [ 0_m ]   <- omega_y  (zero: chassis doesn't pitch)
              [  F  ]   <- omega_z, v_x, v_y  (from F)
              [ 0_m ]   <- v_z      (zero: chassis stays on ground)

    With m=2 wheels this gives a (6 x 2) matrix.
    """
    F_mat = F(r, d)          # (3, m)
    m = F_mat.shape[1]       # number of wheels = 2
    zeros = np.zeros((1, m))
    return np.vstack([zeros, zeros, F_mat, zeros])   # (6, m)

# ---------------------------------------------------------------------------
link_length_unit = 3

T_0e = np.array([[0, 1, 0, -3], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

body_Jacobian = np.array([0, 0, -1, -3, 0, 0])

pprint_np(label="T_0e", arr=T_0e)

T_b0 = np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

inverse_T0e = mr.TransInv(T_0e)
inverse_Tb0 = mr.TransInv(T_b0)

adjoint_inv_T0e_Tb0 = mr.Adjoint(inverse_T0e @ inverse_Tb0)

pprint_np(label="adjoint_inv_T0e_Tb0", arr=adjoint_inv_T0e_Tb0)
r = 0.5
d = 2
F6 = F6(r, d)

J_base = adjoint_inv_T0e_Tb0 @ F6

J_e1 = J_base[:, 0]
J_e2 = J_base[:, 1] 

pprint_np(label="J_e1", arr=J_e1)
pprint_np(label="J_e2", arr=J_e2)

B1 = np.array([0, 0, 1, -3, 0, 0])

B_List = B1.reshape(6, 1)   # shape (6, n_joints) as required by modern_robotics

pprint_np(label="B_List", arr=B_List)

theta_list = np.array([math.pi/2])

M = np.array([[1, 0, 0, -3], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

Body_Jacobian = mr.JacobianBody(B_List, theta_list)
pprint_np(label="Body_Jacobian", arr=Body_Jacobian)
