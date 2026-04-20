import modern_robotics as mr
import numpy as np
import math

from np_utils import format_numpy_compact, pprint_np


# question 1:

force_question_1 = 2  # Newtons
wrench_tip_s = np.array([0, 0, 0, force_question_1, 0, 0])

link_length_unit = 1

theta_list = np.array([0, math.pi/4, 0])

q1 = np.array([0, 0, 0])
s1 = np.array([0, 0, 1])
h1 = 0
screw_axis_1 = mr.ScrewToAxis(q1, s1, h1)

q2 = np.array([link_length_unit, 0, 0])
s2 = np.array([0, 0, 1])
h2 = 0
screw_axis_2 = mr.ScrewToAxis(q2, s2, h2)

q3 = np.array([3, 0, 0])
s3 = np.array([0, 0, 1])
h3 = 0
screw_axis_3 = mr.ScrewToAxis(q3, s3, h3)

pprint_np(label="screw_axis_1", arr=screw_axis_1)
pprint_np(label="screw_axis_2", arr=screw_axis_2)
pprint_np(label="screw_axis_3", arr=screw_axis_3)

Slist = np.column_stack([screw_axis_1, screw_axis_2, screw_axis_3])

Jacobian_space = mr.JacobianSpace(Slist, theta_list)

pprint_np(label="Jacobian_space", arr=Jacobian_space)

transposed_Jacobian_space = Jacobian_space.transpose()

torques_joints = transposed_Jacobian_space @ wrench_tip_s

pprint_np(label="torques_joints", arr=torques_joints)


# question 2:

L1 = L2 = L3 = L4 = link_length_unit

# θ1=0, θ2=0, θ3=π/2, θ4=−π/2
theta_list_q2 = np.array([0, 0, math.pi / 2, -math.pi / 2])

F_b = np.array([0, 0, 10, 10, 10, 0])

s4   = math.sin(theta_list_q2[3])
s34  = math.sin(theta_list_q2[2] + theta_list_q2[3])
s234 = math.sin(theta_list_q2[1] + theta_list_q2[2] + theta_list_q2[3])

c4   = math.cos(theta_list_q2[3])
c34  = math.cos(theta_list_q2[2] + theta_list_q2[3])
c234 = math.cos(theta_list_q2[1] + theta_list_q2[2] + theta_list_q2[3])

# 6×4 body Jacobian; rows: (ω_bx, ω_by, ω_bz, v_bx, v_by, v_bz)
Jacobian_body_col1 = np.array([0, 0, 1,
                                L3*s4 + L2*s34 + L1*s234,
                                L4 + L3*c4 + L2*c34 + L1*c234,
                                0])

Jacobian_body_col2 = np.array([0, 0, 1,
                                L3*s4 + L2*s34,
                                L4 + L3*c4 + L2*c34,
                                0])

Jacobian_body_col3 = np.array([0, 0, 1,
                                L3*s4,
                                L4 + L3*c4,
                                0])

Jacobian_body_col4 = np.array([0, 0, 1,
                                0,
                                L4,
                                0])

Jacobian_body = np.column_stack([
    Jacobian_body_col1,
    Jacobian_body_col2,
    Jacobian_body_col3,
    Jacobian_body_col4,
])

pprint_np(label="Jacobian_body", arr=Jacobian_body)

torques_joints = Jacobian_body.T @ F_b

pprint_np(label="torques_joints", arr=torques_joints)


# question 3:

S1_q3 = np.array([0, 0, 1, 0, 0, 0], dtype=float)
S2_q3 = np.array([1, 0, 0, 0, 2, 0], dtype=float)
S3_q3 = np.array([0, 0, 0, 0, 1, 0], dtype=float)

Slist_q3 = np.column_stack([S1_q3, S2_q3, S3_q3])

theta_list_q3 = np.array([math.pi / 2, math.pi / 2, 1])

Jacobian_space_q3 = mr.JacobianSpace(Slist_q3, theta_list_q3)

pprint_np(label="Jacobian_space_q3", arr=Jacobian_space_q3)


# question 4:

B1_q4 = np.array([0,  1, 0, 3, 0, 0], dtype=float)
B2_q4 = np.array([-1, 0, 0, 0, 3, 0], dtype=float)
B3_q4 = np.array([0,  0, 0, 0, 0, 1], dtype=float)

Blist_q4 = np.column_stack([B1_q4, B2_q4, B3_q4])

theta_list_q4 = np.array([math.pi / 2, math.pi / 2, 1])

Jacobian_body_q4 = mr.JacobianBody(Blist_q4, theta_list_q4)

pprint_np(label="Jacobian_body_q4", arr=Jacobian_body_q4)


# question 5:

Jacobian_body_q5 = np.array([
    [ 0,     -1,     0,      0,     -1,     0,     0    ],
    [ 0,      0,     1,      0,      0,     1,     0    ],
    [ 1,      0,     0,      1,      0,     0,     1    ],
    [-0.105,  0,     0.006, -0.045,  0,     0.006, 0    ],
    [-0.889,  0.006, 0,     -0.844,  0.006, 0,     0    ],
    [ 0,     -0.105, 0.889,  0,      0,     0,     0    ],
])

pprint_np(label="Jacobian_body_q5", arr=Jacobian_body_q5)

Jacobian_linear_velocity_part_q5 = Jacobian_body_q5[3:6, :]

pprint_np(label="Jacobian_linear_velocity_part_q5", arr=Jacobian_linear_velocity_part_q5)

A = Jacobian_linear_velocity_part_q5 @ Jacobian_linear_velocity_part_q5.transpose()

pprint_np(label="A", arr=A)

eigenvalues, eigenvectors = np.linalg.eig(A)

pprint_np(label="eigenvalues", arr=eigenvalues)
pprint_np(label="eigenvectors", arr=eigenvectors)

max_eigenvalue = np.max(eigenvalues)
max_eigenvalue_axis = eigenvectors[:, np.argmax(eigenvalues)]
longest_manipulability_semiaxis_length = math.sqrt(max_eigenvalue)

pprint_np(label="max_eigenvalue", arr=max_eigenvalue)
pprint_np(label="max_eigenvalue_axis", arr=max_eigenvalue_axis)

# check if unit eigenvector
if np.linalg.norm(max_eigenvalue_axis) != 1:
    print("max_eigenvalue_axis is not a unit eigenvector")
else:
    print("max_eigenvalue_axis is a unit eigenvector")

pprint_np(label="longest_manipulability_semiaxis_length", arr=longest_manipulability_semiaxis_length)