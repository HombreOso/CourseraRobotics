import numpy as np


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


def pprint_np(arr, label="", precision=8):
    """Print  label = <compact array>  on a single line."""
    print(f"{label} = {format_numpy_compact(arr, precision)}")
