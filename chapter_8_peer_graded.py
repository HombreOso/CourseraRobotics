import numpy as np
import csv

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

gravity = np.array([0, 0, -9.81])

import modern_robotics as mr
theta_list_start_part_1 = np.array([0,0,0,0,0,0])
theta_list_start_part_2 = np.array([0,-1,0,0,0,0])
simulation_duration_seconds_part_2 = 5

integration_time_steps_per_second = 100
integration_time_step_count_part_1 = 3 * integration_time_steps_per_second
integration_time_step_count_part_2 = 5 * integration_time_steps_per_second

integration_time_step = 1 / integration_time_steps_per_second # Hertz

theta_list_part_1 = [theta_list_start_part_1]
theta_list_part_2 = [theta_list_start_part_2]

current_theta_list_part_1 = theta_list_start_part_1
current_theta_list_part_2 = theta_list_start_part_2
current_dtheta_list_part_1 = np.zeros(6)
current_dtheta_list_part_2 = np.zeros(6)
current_ddtheta_list_part_1 = np.zeros(6)
current_ddtheta_list_part_2 = np.zeros(6)
constant_tau_list_part_1 = np.zeros(6)
constant_tau_list_part_2 = np.zeros(6)
constant_wrench_tip_part_1 = constant_wrench_tip_part_2 = np.zeros(6)



# PART 1:
for i in range(integration_time_step_count_part_1):
    # calculate the acceleration of joints using forward dynamics
    current_ddtheta_list_part_1 = mr.ForwardDynamics(
        thetalist=current_theta_list_part_1, 
        dthetalist=current_dtheta_list_part_1, 
        taulist=constant_tau_list_part_1, 
        g=gravity, 
        Ftip=constant_wrench_tip_part_1, 
        Mlist=Mlist, 
        Glist=Glist, 
        Slist=Slist)
    # calculate the next theta and velocity using Euler step
    [current_theta_list_part_1, current_dtheta_list_part_1] = mr.EulerStep(
        thetalist=current_theta_list_part_1, 
        dthetalist=current_dtheta_list_part_1, 
        ddthetalist=current_ddtheta_list_part_1, 
        dt=integration_time_step)

    theta_list_part_1.append(current_theta_list_part_1)

# PART 2:
for i in range(integration_time_step_count_part_2):

    # calculate the acceleration of joints using forward dynamics
    current_ddtheta_list_part_2 = mr.ForwardDynamics(
        thetalist=current_theta_list_part_2, 
        dthetalist=current_dtheta_list_part_2, 
        taulist=constant_tau_list_part_2, 
        g=gravity, 
        Ftip=constant_wrench_tip_part_2, 
        Mlist=Mlist, 
        Glist=Glist, 
        Slist=Slist)
    # calculate the next theta and velocity using Euler step
    [current_theta_list_part_2, current_dtheta_list_part_2] = mr.EulerStep(
        thetalist=current_theta_list_part_2, 
        dthetalist=current_dtheta_list_part_2, 
        ddthetalist=current_ddtheta_list_part_2, 
        dt=integration_time_step)

    theta_list_part_2.append(current_theta_list_part_2)

# save theta lists to csv
with open("chapter_8_peer_graded_assignment_part_1_thetas.csv", "w", newline='') as f:
    csv.writer(f).writerows(theta_list_part_1)
with open("chapter_8_peer_graded_assignment_part_2_thetas.csv", "w", newline='') as f:
    csv.writer(f).writerows(theta_list_part_2)