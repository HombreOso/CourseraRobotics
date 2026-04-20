import modern_robotics as mr
import numpy as np
import math

from np_utils import format_numpy_compact, pprint_np

# preliminaries

link_length_unit = 1
L1 = L2 = L3 = link_length_unit

T_sd = np.array([
    [-0.585, -0.811, 0, 0.076],
    [ 0.811, -0.585, 0, 2.608],
    [ 0,      0,     1, 0    ],
    [ 0,      0,     0, 1    ]
])

theta_0 = np.array([math.pi/4, math.pi/4, math.pi/4])

tolerance_w = 1e-3
tolerance_v = 1e-4

M = np.array([[1, 0, 0, 3],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])

B1 = np.array([0, 0, 1, 0, 3, 0])
B2 = np.array([0, 0, 1, 0, 2, 0])
B3 = np.array([0, 0, 1, 0, 1, 0])

B_list = np.column_stack([B1, B2, B3])

pprint_np(label="B_list", arr=B_list)

theta_list, success = mr.IKinBody(B_list, M, T_sd, theta_0, tolerance_w, tolerance_v)

pprint_np(label="theta_list", arr=theta_list)