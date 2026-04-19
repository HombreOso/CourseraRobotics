import modern_robotics as mr
import numpy as np
import math

from np_utils import format_numpy_compact, pprint_np


# question 1:

# configuration of end effector in {s} frame when all joints are in zero position
M = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

# link length unit
link_length_unit = 1

# translation displacement of end effector in {s} frame when all joints are in zero position
p_s_0 = np.array([\
link_length_unit*(1.0 + math.sqrt(3.0) + 1.0), \
0,\
link_length_unit*(-1.0 + math.sqrt(3.0) + 2.0)])

M[0:3, 3] = p_s_0
pprint_np(label="M", arr=M)

# question 2:

# configuration of end effector in {s} frame when all joints are in zero posi
# link length unit
link_length_unit = 1

# question 2:

s1 = np.array([0, 0, 1])
q1 = np.array([link_length_unit, 0, 0])
h1 = 0

screw_axis_1 = mr.ScrewToAxis(q1, s1, h1)

s2 = np.array([0, 1, 0])
q2 = np.array([link_length_unit, 0, 0])
h2 = 0
screw_axis_2 = mr.ScrewToAxis(q2, s2, h2)

s3 = np.array([0, 1, 0])
q3 = np.array([link_length_unit*(1+math.sqrt(3)), 0, -1])
h3 = 0
screw_axis_3 = mr.ScrewToAxis(q3, s3, h3)

s4 = np.array([0, 1, 0])
q4 = np.array([link_length_unit*(math.sqrt(3)+2), 0, link_length_unit*(math.sqrt(3)-1)])
h4 = 0
screw_axis_4 = mr.ScrewToAxis(q4, s4, h4)

screw_axis_5 = np.array([0, 0, 0, 0, 0, 1])
s6 = np.array([0, 0, 1])
q6 = np.array([link_length_unit*(math.sqrt(3)+2), 0, link_length_unit*(math.sqrt(3)-1+2)])
h6 = 0
screw_axis_6 = mr.ScrewToAxis(q6, s6, h6)

joint_angles_config_question_3 = np.array([-math.pi/2, math.pi/2, math.pi/3, -math.pi/4, 1, math.pi/6])

pprint_np(label="screw_axis_1", arr=screw_axis_1)
pprint_np(label="screw_axis_2", arr=screw_axis_2)
pprint_np(label="screw_axis_3", arr=screw_axis_3)
pprint_np(label="screw_axis_4", arr=screw_axis_4)
pprint_np(label="screw_axis_5", arr=screw_axis_5)
pprint_np(label="screw_axis_6", arr=screw_axis_6)   

Slist = np.column_stack(
    [
        screw_axis_1,
        screw_axis_2,
        screw_axis_3,
        screw_axis_4,
        screw_axis_5,
        screw_axis_6,
    ]
)

pprint_np(label="Slist", arr=Slist)

T_question_2 = mr.FKinSpace(M, Slist, joint_angles_config_question_3)

pprint_np(label="T_question_2", arr=T_question_2)

adjoint_M_inverse = mr.Adjoint(mr.TransInv(M))

pprint_np(label="adjoint_M_inverse", arr=adjoint_M_inverse)

body_screw_axis_1 = adjoint_M_inverse @ screw_axis_1
body_screw_axis_2 = adjoint_M_inverse @ screw_axis_2
body_screw_axis_3 = adjoint_M_inverse @ screw_axis_3
body_screw_axis_4 = adjoint_M_inverse @ screw_axis_4
body_screw_axis_5 = adjoint_M_inverse @ screw_axis_5
body_screw_axis_6 = adjoint_M_inverse @ screw_axis_6

pprint_np(label="body_screw_axis_1", arr=body_screw_axis_1)
pprint_np(label="body_screw_axis_2", arr=body_screw_axis_2)
pprint_np(label="body_screw_axis_3", arr=body_screw_axis_3)
pprint_np(label="body_screw_axis_4", arr=body_screw_axis_4)
pprint_np(label="body_screw_axis_5", arr=body_screw_axis_5)
pprint_np(label="body_screw_axis_6", arr=body_screw_axis_6)

body_Slist = np.column_stack(
    [
        body_screw_axis_1,
        body_screw_axis_2,
        body_screw_axis_3,
        body_screw_axis_4,
        body_screw_axis_5,
        body_screw_axis_6,
    ]
)

pprint_np(label="body_Slist", arr=body_Slist)

body_T_question_2 = mr.FKinBody(M, body_Slist, joint_angles_config_question_3)

pprint_np(label="body_T_question_2", arr=body_T_question_2)