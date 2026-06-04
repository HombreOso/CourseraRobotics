import numpy as np
# ---------------------------------------------------------------------------
# Cube configurations in the space frame {s}
# Cube: 5 cm × 5 cm × 5 cm; frame {c} at cube centre, axes aligned with edges.
# ---------------------------------------------------------------------------

# Default initial pose: (x, y, z) = (1, 0, 0.025), axes aligned with {s}
T_sc_initial = np.array([
    [1, 0, 0, 1    ],
    [0, 1, 0, 0    ],
    [0, 0, 1, 0.025],
    [0, 0, 0, 1    ],
], dtype=float)

# Default goal pose: (x, y, z) = (0, -1, 0.025), rotated -π/2 about ẑ_s
T_sc_goal = np.array([
    [ 0, 1, 0,  0    ],
    [-1, 0, 0, -1    ],
    [ 0, 0, 1,  0.025],
    [ 0, 0, 0,  1    ],
], dtype=float)