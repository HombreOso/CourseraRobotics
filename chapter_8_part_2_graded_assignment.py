import numpy as np

import math

from np_utils import format_numpy_compact, pprint_np

import modern_robotics as mr



M01 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]]
M12 = [[0, 0, 1, 0.28], [0, 1, 0, 0.13585], [-1, 0, 0, 0], [0, 0, 0, 1]]
M23 = [[1, 0, 0, 0], [0, 1, 0, -0.1197], [0, 0, 1, 0.395], [0, 0, 0, 1]]
M34 = [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.14225], [0, 0, 0, 1]]
M45 = [[1, 0, 0, 0], [0, 1, 0, 0.093], [0, 0, 1, 0], [0, 0, 0, 1]]
M56 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.09465], [0, 0, 0, 1]]
M67 = [[1, 0, 0, 0], [0, 0, 1, 0.0823], [0, -1, 0, 0], [0, 0, 0, 1]]
G1 = np.diag([0.010267495893, 0.010267495893,  0.00666, 3.7, 3.7, 3.7])
G2 = np.diag([0.22689067591, 0.22689067591, 0.0151074, 8.393, 8.393, 8.393])
G3 = np.diag([0.049443313556, 0.049443313556, 0.004095, 2.275, 2.275, 2.275])
G4 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
G5 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
G6 = np.diag([0.0171364731454, 0.0171364731454, 0.033822, 0.1879, 0.1879, 0.1879])
Glist = [G1, G2, G3, G4, G5, G6]
Mlist = [M01, M12, M23, M34, M45, M56, M67] 
Slist = [[0,         0,         0,         0,        0,        0],
         [0,         1,         1,         1,        0,        1],
         [1,         0,         0,         0,       -1,        0],
         [0, -0.089159, -0.089159, -0.089159, -0.10915, 0.005491],
         [0,         0,         0,         0,  0.81725,        0],
         [0,         0,     0.425,   0.81725,        0,  0.81725]]

theta_angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3])
theta_velocity = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
theta_acceleration = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
gravity = np.array([0, 0, -9.81])
tau_question_5 = np.array([0.0128, -41.1477, -3.7809, 0.0323, 0.0370, 0.1034])
wrench_tip = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

mass_matrix = mr.MassMatrix(theta_angles, Mlist, Glist, Slist)

pprint_np(label="mass_matrix", arr=mass_matrix)

coriolis_centripetal_components = mr.VelQuadraticForces(theta_angles, theta_velocity, Mlist, Glist, Slist)

pprint_np(label="coriolis_centripetal_components", arr=coriolis_centripetal_components)

torques_to_overcome_gravity_forces = mr.GravityForces(theta_angles, gravity, Mlist, Glist, Slist)

pprint_np(label="torques_to_overcome_gravity_forces", arr=torques_to_overcome_gravity_forces)


# forces to create the wrench Ftip at the end effector
JTFtip = mr.EndEffectorForces(theta_angles, wrench_tip, Mlist, Glist, Slist)

pprint_np(label="JTFtip", arr=JTFtip)

accelerations_question_5 = mr.ForwardDynamics(theta_angles, theta_velocity, tau_question_5, gravity, wrench_tip, Mlist, Glist, Slist)

pprint_np(label="accelerations_question_5", arr=accelerations_question_5)