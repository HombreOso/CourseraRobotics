import numpy as np
import modern_robotics as mr

from se3_matrix_exponential import chop_small


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
T_as = mr.TransInv(T_sa)
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
Ad_Tas = mr.Adjoint(T_as)
twist_a = Ad_Tas @ twist_s

print("twist_a =\n", twist_a)

Wrench_b = np.array([1,0,0,2,1,0])

Ad_Tsb = mr.Adjoint(T_sb)

transposed_Ad_Tsb = Ad_Tsb.transpose()

Wrench_s = transposed_Ad_Tsb @ Wrench_b

print("Wrench_s =\n", Wrench_s)

# question 11: inverse of T via TransInv (SE(3): T^{-1} = [R^T, -R^T p; 0 0 0 1])
T = np.array(
    [
        [0.0, -1.0, 0.0, 3.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
T_inv = mr.TransInv(T)
print("T =\n", T)
print("TransInv(T) =\n", T_inv)

# question 12: VecTose3

twist_question_12 = np.array([1,0,0,0,2,3])

se3_question_12 = mr.VecTose3(twist_question_12)

print("VecTose3(twist_question_12) =\n", se3_question_12)


# question 13: screwToAxis

s_hat_question_13 = np.array([1,0,0])
p_question_13 = np.array([0,0,2])
pitch_question_13 = 1

s_question_13 = mr.ScrewToAxis(p_question_13, s_hat_question_13, pitch_question_13)

print("screwToAxis(p_question_13, s_hat_question_13, pitch_question_13) =\n", s_question_13)

# question 14: T = exp([S]θ) via MatrixExp6 (exponential coordinates in se(3))
# [S]θ has [ω]θ in the 3x3 block (here ω = [0,0,1], θ = π/2) and last column vθ.
S_theta_q14 = np.array(
    [
        [0.0, -np.pi / 2.0, 0.0, 3.0 * np.pi / 4.0],
        [np.pi / 2.0, 0.0, 0.0, -3.0 * np.pi / 4.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
)
T_q14 = chop_small(mr.MatrixExp6(S_theta_q14))
print("[S]*theta (question 14) =\n", S_theta_q14)
print("MatrixExp6([S]*theta) = T =\n", T_q14)

# question 15: matrix logarithm [S]*theta = MatrixLog6(T) in se(3)
T_q15 = np.array(
    [
        [0.0, -1.0, 0.0, 3.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
S_theta_q15 = chop_small(mr.MatrixLog6(T_q15))
print("T (question 15) =\n", T_q15)
print("MatrixLog6(T) = [S]*theta =\n", S_theta_q15)

print("twist_a =\n", twist_a)
