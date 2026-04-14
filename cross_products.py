import numpy as np
import modern_robotics as mr


def unit(v):
    """Return a normalized copy of a vector."""
    return v / np.linalg.norm(v)


def frame_from_xy(x_axis, y_axis):
    """Build a right-handed rotation matrix from x and y axes."""
    x_axis = unit(x_axis)
    y_axis = unit(y_axis)
    z_axis = unit(np.cross(x_axis, y_axis))
    # Recompute y so the three axes are mutually orthogonal.
    y_axis = unit(np.cross(z_axis, x_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


# Axes of frame {a} expressed in frame {s}
X_a = np.array([0.0, 0.0, 1.0])
Y_a = np.array([-1.0, 0.0, 0.0])
R_sa = frame_from_xy(X_a, Y_a)

print("R_sa =\n", R_sa)


# Axes of frame {b} expressed in frame {s}
X_b = np.array([1.0, 0.0, 0.0])
Y_b = np.array([0.0, 0.0, -1.0])
R_sb = frame_from_xy(X_b, Y_b)

print("R_sb =\n", R_sb)


# Origins of {a} and {b} expressed in frame {s}
p_a_s = np.array([0.0, 0.0, 1.0])
p_b_s = np.array([0.0, 2.0, 0.0])

T_sa = mr.RpToTrans(R_sa, p_a_s)
T_sb = mr.RpToTrans(R_sb, p_b_s)

# Relative transform of frame {b} expressed in frame {a}
T_ab = mr.TransInv(T_sa) @ T_sb

print("T_sa =\n", T_sa)
print("T_sb =\n", T_sb)
print("T_ab =\n", T_ab)

se3_Tsa = mr.MatrixLog6(T_sa)
S_theta_Tsa = mr.se3ToVec(se3_Tsa)



print("MatrixLog6(T_sa) =\n", se3_Tsa)
print("se3ToVec(MatrixLog6(T_sa)) =\n", S_theta_Tsa)

se3_Tsa = mr.MatrixLog6(T_sa)
S_theta = mr.se3ToVec(se3_Tsa)

S, theta = mr.AxisAng6(S_theta)

print("S =", S)
print("theta =", theta)






T_bs = mr.TransInv(T_sb)
print("T_bs =\n", T_bs)

# question 5

p_b = np.array([1,2, 3, 1])
p_s = T_sb @ p_b

print("p_s =\n", p_s)

# question 7

twist_s = np.array([3,2,1,-1,-2,-3])
Ad_Tsb = mr.Adjoint(T_sb)
twist_b = Ad_Tsb @ twist_s

Ad_Tsa = mr.Adjoint(T_sa)
twist_a = Ad_Tsa @ twist_s

print("Ad_Tsa =\n", Ad_Tsa)
print("twist_a =\n", twist_a)

print("Ad_Tsb =\n", Ad_Tsb)
print("twist_b =\n", twist_b)

# question 8

twist_b = np.array([3,2,1,-1,-2,-3])
Ad_Tbs = mr.Adjoint(T_bs)
twist_s = Ad_Tbs @ twist_b

print("Ad_Tbs =\n", Ad_Tbs)
print("twist_s =\n", twist_s)


# question 9
omega_theta_non_skewed = np.array([0,1,2])
velocity_non_skewed = np.array([[3,0,0]])

velocity_skewed_transposed = velocity_non_skewed.transpose()

S_theta_non_skewed = np.hstack((omega_theta_non_skewed, velocity_non_skewed))

omega_theta_skewed = mr.VecToso3(omega_theta_non_skewed)

print("omega_theta_skewed =\n", omega_theta_skewed)

print("velocity_non_skewed.transpose() =\n", velocity_non_skewed.transpose())

S_skewed_theta = np.vstack((np.hstack((omega_theta_skewed, velocity_non_skewed.transpose())), np.zeros((1,4))))

print("S_skewed_theta =\n", S_skewed_theta)

exponential_theta = mr.MatrixExp6(S_skewed_theta)

print("exponential_theta =\n", exponential_theta)