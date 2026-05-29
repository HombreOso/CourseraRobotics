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
link_length_unit = 3

q1 = np.array([link_length_unit, 0, 0])
s1 = np.array([0, 0, 1])
h1 = 0
screw_axis_1 = mr.ScrewToAxis(q1, s1, h1)

joint_angles_config_question_1 = np.array([math.pi/2])

M = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

T_0e = mr.FKinBody(M, screw_axis_1, joint_angles_config_question_1)

pprint_np(label="T_0e", arr=T_0e)

T_b0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]])

inverse_T0e = mr.TransInv(T_0e)
inverse_Tb0 = mr.TransInv(T_b0)

adjoint_inv_T0e_Tb0 = mr.Adjoint(inverse_T0e @ inverse_Tb0)

pprint_np(label="adjoint_inv_T0e_Tb0", arr=adjoint_inv_T0e_Tb0)

F6 = H_phi()

J_base = adjoint_inv_T0e_Tb0 @ F6


