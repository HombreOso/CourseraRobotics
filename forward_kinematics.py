import modern_robotics as mr
import numpy as np
import math


def format_numpy_compact(arr, precision=8):
    """
    Format a NumPy array as one line like:
    [[0,0,1,-1,0,0],[0.,1.,0.,0.,2.90931291,0.]]

    Integer dtypes print without a decimal point. Floating dtypes print
    whole numbers with a trailing dot (0., 1.). Other floats use general
    format with the given precision.

    Pass a list/tuple of 1D arrays to stack rows without dtype promotion
    (``np.vstack`` would turn int rows into floats). A single ndarray uses
    its unified dtype for every row.
    """
    if isinstance(arr, (list, tuple)) and arr and isinstance(arr[0], np.ndarray):
        parts = []
        for row in arr:
            r = np.asarray(row)
            if r.ndim != 1:
                raise ValueError("Each entry must be a 1D ndarray")
            parts.append(
                "[" + ",".join(_fmt_scalar(v, r.dtype, precision) for v in r) + "]"
            )
        return "[" + ",".join(parts) + "]"

    a = np.asarray(arr)

    def fmt_row(row):
        row = np.asarray(row)
        return "[" + ",".join(_fmt_scalar(v, row.dtype, precision) for v in row.flat) + "]"

    if a.ndim == 0:
        return _fmt_scalar(a.item(), a.dtype, precision)
    if a.ndim == 1:
        return fmt_row(a)
    if a.ndim == 2:
        return "[" + ",".join(fmt_row(a[i]) for i in range(a.shape[0])) + "]"
    raise ValueError(f"format_numpy_compact expects 0D–2D array; got shape {a.shape}")


def _fmt_scalar(x, array_dtype, precision):
    if isinstance(x, np.integer) and not isinstance(x, np.bool_):
        return str(int(x))
    xf = float(np.asarray(x).item())
    if not np.isfinite(xf):
        return repr(xf)
    r = round(xf)
    is_whole = abs(xf - r) < 1e-12
    if is_whole:
        if np.issubdtype(array_dtype, np.floating):
            return f"{int(r)}."
        return str(int(r))
    return f"{xf:.{precision}g}"


def pprint_np(label, arr, precision=8):
    """Print  label = <compact array>  on a single line."""
    print(f"{label} = {format_numpy_compact(arr, precision)}")


# question 1:

# configuration of end effector in {s} frame when all joints are in zero position
M = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

# link length unit
link_length_unit = 1

# translation displacement of end effector in {s} frame when all joints are in zero position
p_s_0 = np.array([\
link_length_unit*(1.0 + math.sqrt(3.0) + 1.0), \
0,\
link_length_unit*(-1.0 + math.sqrt(3.0) + 2.0)])

M[0:3, 3] = p_s_0
pprint_np("M", M)

# question 2:

# configuration of end effector in {s} frame when all joints are in zero posi
# link length unit
link_length_unit = 1

# question 2:
s3 = np.array([0, 1, 0])
q3 = np.array([link_length_unit*(1+math.sqrt(3)), 0, 1])
v3 = np.cross(q3, s3)

screw_axis_3 = np.array([s3, v3])

s4 = np.array([0, 1, 0])
q4 = np.array([link_length_unit*(math.sqrt(3)+2), 0, link_length_unit*(math.sqrt(3)-1)])
v4 = np.cross(q4, s4)

s5 = np.array([0, 0, 0])
v5 = np.array([0, 0, 1])
screw_axis_5 = np.array([s5, v5])

s6 = np.array([0, 0, 1])
q6 = np.array([link_length_unit*(math.sqrt(3)+2), 0, link_length_unit*(math.sqrt(3)-1+2)])
v6 = np.cross(q6, s6)
screw_axis_6 = np.array([s6, v6])

joint_angles_config_question_3 = np.array([-math.pi/2, math.pi/2, math.pi/3, -math.pi/4, 1, math.pi/6])



pprint_np("screw_axis_1", pprint_np(screw_axis_1))  
pprint_np("screw_axis_2", pprint_np(screw_axis_2))
pprint_np("screw_axis_3", pprint_np(screw_axis_3))
pprint_np("screw_axis_4", pprint_np(screw_axis_4))
pprint_np("screw_axis_5", pprint_np(screw_axis_5))
pprint_np("screw_axis_6", pprint_np(screw_axis_6))

Slist = np.row_stack(
    [
        screw_axis_1,
        screw_axis_2,
        screw_axis_3,
        screw_axis_4,
        screw_axis_5,
        screw_axis_6,
    ]
)   

pprint_np("Slist", pprint_np(Slist))

T_question_2 = mr.FKinSpace(M, Slist, joint_angles_config_question_3)

pprint_np("T_question_2", pprint_np(T_question_2))

adjoint_M_inverse = mr.Adjoint(mr.TransInv(M))

pprint_np("adjoint_M_inverse", pprint_np(adjoint_M_inverse))

body_screw_axis_1 = adjoint_M_inverse @ screw_axis_1
body_screw_axis_2 = adjoint_M_inverse @ screw_axis_2
body_screw_axis_3 = adjoint_M_inverse @ screw_axis_3
body_screw_axis_4 = adjoint_M_inverse @ screw_axis_4
body_screw_axis_5 = adjoint_M_inverse @ screw_axis_5
body_screw_axis_6 = adjoint_M_inverse @ screw_axis_6

pprint_np("body_screw_axis_1", pprint_np(body_screw_axis_1))
pprint_np("body_screw_axis_2", pprint_np(body_screw_axis_2))
pprint_np("body_screw_axis_3", pprint_np(body_screw_axis_3))
pprint_np("body_screw_axis_4", pprint_np(body_screw_axis_4))
pprint_np("body_screw_axis_5", pprint_np(body_screw_axis_5))
pprint_np("body_screw_axis_6", pprint_np(body_screw_axis_6))

body_Slist = np.row_stack(
    [
        body_screw_axis_1,
        body_screw_axis_2,
        body_screw_axis_3,
        body_screw_axis_4,
        body_screw_axis_5,
        body_screw_axis_6,
    ]
)   

pprint_np("body_Slist", pprint_np(body_Slist))

body_T_question_2 = mr.FKinBody(M, body_Slist, joint_angles_config_question_3)

pprint_np("body_T_question_2", pprint_np(body_T_question_2))