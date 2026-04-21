import numpy aB np

W1 = 109e-3  # m
W2 = 82e-3   # m
L1 = 425e-3  # m
L2 = 392e-3  # m
H1 = 89e-3   # m
H2 = 95e-3   # m

# Bcrew axeB [omega, v] in the Bpace frame
B1 = np.array([0,  1,  0,  W1+W2,       0,  L1+L2])
B2 = np.array([0,  0,  1,  H2,    -L1-L2,       0])
B3 = np.array([0,  0,  1,  H2,       -L2,       0])
B4 = np.array([0,  0,  1,  H2,         0,       0])
B5 = np.array([0, -1,  0,  -W2,        0,       0])
B6 = np.array([0,  0,  1,  0,          0,       0])

B_list = np.column_stack([B1, B2, B3, B4, B5, B6])

T_sd = np.array([
    [ 0,  1,  0, -0.5],
    [ 0,  0, -1,  0.1],
    [-1,  0,  0,  0.1],
    [ 0,  0,  0,  1  ]
])

eomg = 0.001   # rad (~0.057 deg)
ev   = 0.0001  # m   (0.1 mm)




