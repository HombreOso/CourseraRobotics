import numpy as np
import csv
import time
import modern_robotics as mr
from np_utils import format_numpy_compact, pprint_np


X_start = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

X_end = np.array([
    [0, 0, 1, 1],
    [1, 0, 0, 2],
    [0, 1, 0, 3],
    [0, 0, 0, 1]
])

time_scaling = mr.QuinticTimeScaling(Tf = 5, t = 3)

pprint_np(time_scaling, label="Time scaling")

# question 6:
screw_trajectory = mr.ScrewTrajectory(Xstart=X_start, Xend=X_end, Tf=5, N=10, method=3)
print("Screw trajectory shape: ", np.array(screw_trajectory).shape)
pprint_np(arr=screw_trajectory[8], label="Screw trajectory nineth matrix")

# question 7:
cartesian_trajectory = mr.CartesianTrajectory(X_start, X_end, Tf=5, N=10, method=5)
print("Cartesian trajectory shape: ", np.array(cartesian_trajectory).shape)
pprint_np(arr=cartesian_trajectory[8], label="Cartesian trajectory nineth matrix")
