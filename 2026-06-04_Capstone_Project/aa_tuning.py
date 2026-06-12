k_p_diag_value = 2.0
k_i_diag_value = 0.2
k_nl_phi = 1.5
k_nl_x = 1.5
k_nl_y = 1.5

# Maximum actuator speed (rad/s) applied to all 9 actuators in NextState.
# Acts as a hardware clamp AFTER the Jacobian inverse.  With damped pseudoinverse
# (lambda_damping below) the need for a tight clamp is reduced; set to inf to
# let the Jacobian regularisation do all the work.
max_speed = float('inf')

# Damped least-squares pseudoinverse (Modern Robotics, Ch. 6):
#     J⁺ = Jᵀ (JJᵀ + λ²I)⁻¹
# λ = 0  → exact Moore-Penrose pseudoinverse (original behaviour).
# λ = 0.05 → bounds the Jacobian gain to ≤ 1/(2λ) = 10 near singularities,
#             eliminating the 196 rad/s runaway while barely affecting
#             well-conditioned configurations.
lambda_damping = 0.005

# Feedforward Adjoint-blending gain (set to 0 to disable, see FeedbackControl
# docstring).  Prefer lambda_damping; this is left at 0 by default.
ff_error_scale = 0.0
