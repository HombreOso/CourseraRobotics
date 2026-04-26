import modern_robotics as mr
import numpy as np
import math

from np_utils import format_numpy_compact, pprint_np

# question 1:

diameter_cylinder = 4.0 / 100.0 # meters
length_cylinder = 20.0 / 100.0 # meters
diameter_sphere = 20.0 / 100.0 # meters

density_dumpbell = 5_600.0 # kg/m^3

mass_sphere = density_dumpbell * (4/3 * math.pi * (diameter_sphere / 2) ** 3)
mass_cylinder = density_dumpbell * math.pi * (diameter_cylinder / 2) ** 2 * length_cylinder


mass_dumpbell = 2 * mass_sphere + mass_cylinder

# intertia matrix of sphere in sphere frame

diagonal_component_sphere = mass_sphere * 2 * (diameter_sphere / 2) ** 2 / 5
inertia_matrix_sphere_own_frame = np.array([
    [diagonal_component_sphere, 0, 0],
    [0, diagonal_component_sphere, 0],
    [0, 0, diagonal_component_sphere]
])

radius_cylinder = diameter_cylinder / 2
cylinder_axis_component = mass_cylinder * radius_cylinder ** 2 / 2
cylinder_round_component = mass_cylinder * ( 3 * radius_cylinder ** 2 + length_cylinder ** 2) / 12
inertia_matrix_cylinder_own_frame = np.array([
    [cylinder_round_component, 0, 0],
    [0, cylinder_round_component, 0],
    [0, 0, cylinder_axis_component]
])

def parallel_axis_theorem(inertia_own_frame, mass, q):
    qt_q = q.T @ q
    q_qt = q @ q.T
    return inertia_own_frame + mass * (qt_q * np.eye(3) - q_qt)

q_left_sphere = np.array([0, 0, -length_cylinder / 2 - diameter_sphere / 2])
q_right_sphere = np.array([0, 0, length_cylinder / 2 + diameter_sphere / 2])

inertia_left_sphere_in_dumpbell_frame = parallel_axis_theorem(
    inertia_matrix_sphere_own_frame, mass_dumpbell, q_left_sphere)

inertia_right_sphere_in_dumpbell_frame = parallel_axis_theorem(
    inertia_matrix_sphere_own_frame, mass_dumpbell, q_right_sphere)

resulting_inertia = inertia_left_sphere_in_dumpbell_frame + \
    inertia_right_sphere_in_dumpbell_frame + \
    inertia_matrix_cylinder_own_frame

print("resulting_inertia =\n", resulting_inertia)

pprint_np(label="resulting_inertia", arr=resulting_inertia)

# question 5
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

theta             = np.array([0, math.pi/6, math.pi/4, math.pi/3, math.pi/2, 2*math.pi/3])
theta_velocity    = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
theta_acceleration = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
gravity           = np.array([0, 0, -9.81])
wrench_tip        = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

torques = mr.InverseDynamics(thetalist=theta, dthetalist=theta_velocity, ddthetalist=theta_acceleration, g=gravity, Ftip=wrench_tip, Mlist=Mlist, \
                    Glist=Glist, Slist=Slist)

pprint_np(label="torques", arr=torques)


# question 2:

theta_list = np.array([0, math.pi/6, math.pi/4, math.pi/3, math.pi/2, 2*math.pi/3])
theta_velocity_list = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
theta_acceleration_list = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
