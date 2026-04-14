"""
Matrix exponential exp([V]) for se(3), i.e. 4x4 twists as in *Modern Robotics*
(Lynch & Park), Chapter 3 (exponential coordinates of rigid-body motion).

Notation (book):
  A twist in se(3) is written as a 4x4 matrix

      [V] = | [ω]  v |
            |  0    0 |

  where [ω] is the 3x3 skew-symmetric matrix of angular part ω ∈ R^3, and
  v ∈ R^3 is the linear part (last column of the top 3x4 block).

  Exponential coordinates: often you store [ω]θ in the upper-left and a
  3-vector in the last column that scales consistently with that angle
  (as in mr.MatrixExp6). This function matches modern_robotics.MatrixExp6.

Reference: same structure as Northwestern's modern_robotics.core.MatrixExp6.
"""

from __future__ import annotations

import numpy as np


def _skew(w: np.ndarray) -> np.ndarray:
    """Skew-symmetric [w] for w = (wx, wy, wz)."""
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    return np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]])


def _vee_so3(so3: np.ndarray) -> np.ndarray:
    """Inverse of skew: vector ω from [ω]."""
    return np.array([so3[2, 1], so3[0, 2], so3[1, 0]])


def _near_zero(x: float, eps: float = 1e-6) -> bool:
    return abs(x) < eps


def chop_small(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Set entries with |entry| < eps to exactly 0 (cleans float noise like 1e-16).

    Use a modest eps: large enough to kill roundoff in sin/cos/Rodrigues, small
    enough not to wipe meaningful small values in normal robotics magnitudes.
    """
    a = np.asarray(x, dtype=float)
    out = np.where(np.abs(a) < eps, 0.0, a)
    return out


def matrix_exp_so3(omega_hat: np.ndarray) -> np.ndarray:
    """
    exp([ω]) for so(3), with [ω] any 3x3 skew matrix.

    If ω_hat = θ * [k] for unit axis k, this returns rotation by angle θ
    about k (Rodrigues' formula applied with that combined angle).

    Book: exp([ω]) = I + sinθ/θ * [ω] + (1-cosθ)/θ^2 * [ω]^2 when θ = ||ω|| ≠ 0;
    implemented equivalently via normalized [k] = [ω]/θ.
    """
    omega_hat = np.asarray(omega_hat, dtype=float)
    w = _vee_so3(omega_hat)
    theta = float(np.linalg.norm(w))
    if _near_zero(theta):
        return np.eye(3)
    k_hat = omega_hat / theta
    R = (
        np.eye(3)
        + np.sin(theta) * k_hat
        + (1.0 - np.cos(theta)) * (k_hat @ k_hat)
    )
    return chop_small(R)


def matrix_exp_se3(se3mat: np.ndarray) -> np.ndarray:
    """
    exp([V]) : se(3) → SE(3), same convention as modern_robotics.MatrixExp6.

    Input `se3mat` is 4x4 with bottom row [0,0,0,0] and

        se3mat[:3,:3] = [ω]   (may encode θ and axis together)
        se3mat[:3, 3]  = v

    Cases (book / library):
      1) ||ω_vec|| ≈ 0 with ω_vec = vee([ω]): pure translation.
         exp([V]) = | I  v |
                     | 0  1 |
         (here v is already the displacement in the coordinates used.)

      2) Otherwise: let θ = ||ω_vec||, [k] = [ω]/θ (unit-screw angular part).
         R = exp([ω])  (Rodrigues with angle θ),
         p = ( I θ + (1-cosθ)[k] + (θ - sinθ)[k]^2 ) * (v / θ).

         That (v/θ) factor is because the library expects the *last column*
         to scale like the same θ as in [ω], i.e. consistent exponential
         coordinates (see MR source for MatrixExp6).
    """
    se3mat = np.asarray(se3mat, dtype=float)
    omega_hat = se3mat[0:3, 0:3]
    v = se3mat[0:3, 3].copy()

    omega_vec = _vee_so3(omega_hat)
    if _near_zero(float(np.linalg.norm(omega_vec))):
        out = np.eye(4)
        out[0:3, 3] = chop_small(v.reshape(3))
        return out

    theta = float(np.linalg.norm(omega_vec))
    k_hat = omega_hat / theta  # unit-magnitude skew factor [k], ||k||=1

    R = matrix_exp_so3(omega_hat)
    # G(θ) from the book for unit k: ∫_0^θ exp([k]s) ds
    #     = I θ + (1-cosθ)[k] + (θ - sinθ)[k]^2
    G = (
        np.eye(3) * theta
        + (1.0 - np.cos(theta)) * k_hat
        + (theta - np.sin(theta)) * (k_hat @ k_hat)
    )
    p = G @ (v / theta)

    out = np.eye(4)
    out[0:3, 0:3] = R
    out[0:3, 3] = p
    return chop_small(out)


# --- Optional: normalized screw axis S = (ω, q) and distance θ along screw ---


def matrix_exp_screw(S: np.ndarray, theta: float) -> np.ndarray:
    """
    exp([S] θ) where S is the 6-vector screw axis ω (||ω||=1 or ω=0, ||v||=1)
    in *body* or *space* form depending how you build [S].

    S = [ω1, ω2, ω3, v1, v2, v3]; then [S] has [ω] on top-left and v on top-right.

    For ||ω|| = 1:
        R = I + sinθ [ω] + (1-cosθ) [ω]^2
        p = (I θ + (1-cosθ)[ω] + (θ-sinθ)[ω]^2) v

    For ||ω|| = 0 (prismatic, ||v||=1):
        R = I,  p = v θ.
    """
    S = np.asarray(S, dtype=float).reshape(6)
    w, vv = S[0:3], S[3:6]
    w_hat = _skew(w)
    if _near_zero(float(np.linalg.norm(w))):
        T = np.eye(4)
        T[0:3, 3] = chop_small((vv * theta).reshape(3))
        return T
    R = (
        np.eye(3)
        + np.sin(theta) * w_hat
        + (1.0 - np.cos(theta)) * (w_hat @ w_hat)
    )
    G = (
        np.eye(3) * theta
        + (1.0 - np.cos(theta)) * w_hat
        + (theta - np.sin(theta)) * (w_hat @ w_hat)
    )
    p = G @ vv
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = p
    return chop_small(T)


if __name__ == "__main__":
    import modern_robotics as mr

    omega = np.array([0.0, 1.0, 2.0])
    vel = np.array([[3.0, 0.0, 0.0]]).T
    omega_hat = mr.VecToso3(omega)
    se3 = np.r_[np.c_[omega_hat, vel], np.zeros((1, 4))]

    T_us = matrix_exp_se3(se3)
    T_mr = mr.MatrixExp6(se3)

    print("T_us =\n", T_us)
    print("T_mr =\n", T_mr)
    print("max abs diff vs modern_robotics.MatrixExp6:", np.max(np.abs(T_us - T_mr)))
