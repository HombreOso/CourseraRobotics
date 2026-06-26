"""
plot_error_charts.py
--------------------
For each of the result folders (Overshoot, Best, NewTask) find every
*Xerr*.csv file, plot all six error-twist components against time, and
save a PNG with the same stem name into the same folder.

CSV format (no header): 6 comma-separated floats per row, one row per
timestep.  Timestep is 0.01 s (k=1, 10 ms integration step).

Usage
-----
    py plot_error_charts.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # no display needed — just save files
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).parent
RESULT_FOLDERS = ["Overshoot", "Best", "NewTask"]
DT           = 0.01            # seconds per row (k=1 → one row per 10 ms step)

COMPONENT_LABELS = [
    "error_1 (ωx)", "error_2 (ωy)", "error_3 (ωz)",
    "error_4 (vx)", "error_5 (vy)", "error_6 (vz)",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_xerr(path: Path) -> np.ndarray:
    """Load a headerless 6-column Xerr CSV into an (N, 6) float array."""
    rows = []
    with open(path, newline="") as f:
        for line in csv.reader(f):
            if line:
                rows.append([float(v) for v in line])
    return np.array(rows)


def plot_xerr(data: np.ndarray, title: str, out_path: Path) -> None:
    """
    Plot all 6 error-twist components against time and save to out_path.

    Parameters
    ----------
    data     : (N, 6) array
    title    : figure title (derived from filename)
    out_path : where to write the PNG
    """
    t = np.arange(len(data)) * DT

    fig, ax = plt.subplots(figsize=(10, 5))

    for col_idx, label in enumerate(COMPONENT_LABELS):
        ax.plot(t, data[:, col_idx], label=label, linewidth=1.4)

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("End-effector error twist", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xlim(left=0, right=t[-1])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved -> {out_path.relative_to(SCRIPT_DIR)}")


def make_overview(folder: Path, csv_paths: list[Path]) -> None:
    """
    Extra: one figure with one subplot per CSV, so you can compare all
    tests in the folder side by side.  Saved as  _overview.png .
    """
    n = len(csv_paths)
    if n < 2:
        return                 # no point for a single file

    cols = min(n, 2)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4),
                              squeeze=False)

    for idx, csv_path in enumerate(sorted(csv_paths)):
        data = load_xerr(csv_path)
        t    = np.arange(len(data)) * DT
        ax   = axes[idx // cols][idx % cols]

        for col_idx, label in enumerate(COMPONENT_LABELS):
            ax.plot(t, data[:, col_idx], label=label, linewidth=1.2)

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_title(csv_path.stem, fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Xerr", fontsize=9)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

    # hide any unused subplot panels
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(f"Error overview — {folder.name}", fontsize=13, y=1.01)
    fig.tight_layout()
    out = folder / "_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out.relative_to(SCRIPT_DIR)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    any_found = False

    for folder_name in RESULT_FOLDERS:
        folder = SCRIPT_DIR / folder_name

        if not folder.exists():
            print(f"[skip] folder not found: {folder_name}/")
            continue

        csv_files = sorted(folder.glob("*Xerr*.csv"))
        if not csv_files:
            # also accept any CSV with 6 columns as a fallback
            csv_files = [
                p for p in sorted(folder.glob("*.csv"))
                if _has_six_columns(p)
            ]

        if not csv_files:
            print(f"[skip] no Xerr CSV files in {folder_name}/")
            continue

        print(f"\n{folder_name}/  ({len(csv_files)} file(s))")
        for csv_path in csv_files:
            data = load_xerr(csv_path)
            title = f"{folder_name} — {csv_path.stem}"
            out   = csv_path.with_suffix(".png")
            plot_xerr(data, title, out)
            any_found = True

        make_overview(folder, csv_files)

    if not any_found:
        print("\nNo CSV files were processed.")
        print("Put *Xerr*.csv files into the Overshoot/, Best/, or NewTask/ folders")
        print("next to this script and run again.")


def _has_six_columns(path: Path) -> bool:
    """Quick check: does the first data row have exactly 6 columns?"""
    try:
        with open(path, newline="") as f:
            for line in csv.reader(f):
                if line:
                    return len(line) == 6
    except Exception:
        pass
    return False


if __name__ == "__main__":
    main()
