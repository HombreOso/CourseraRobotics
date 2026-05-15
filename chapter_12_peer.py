import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import modern_robotics as mr
from scipy.optimize import linprog
from dataclasses import dataclass


@dataclass
class Contact:
    x: float
    y: float
    direction: float  # contact normal angle in degrees


def read_applied_contacts(filepath: str = "applied_contacts.csv") -> list[Contact]:
    """Read applied contacts from a CSV file.

    The file may contain:
      - Comment lines starting with '#' (ignored)
      - A header line with columns: x, y, direction
      - Data rows with x (float), y (float), direction (float, degrees)

    Returns a list of Contact objects.
    """
    contacts: list[Contact] = []

    with open(filepath, newline="") as f:
        non_comment_lines = (line for line in f if not line.lstrip().startswith("#"))
        reader = csv.DictReader(non_comment_lines)
        for row in reader:
            contacts.append(Contact(
                x=float(row["x"]),
                y=float(row["y"]),
                direction=float(row["direction"]),
            ))

    return contacts


def compute_contact_wrench(contact: Contact) -> np.ndarray:
    """Compute the contact wrench (planar) for a given (planar) contact.

    The contact wrench is the force and torque applied by the contact.

    The contact wrench F is calculated as 
    F = [contact.x * sin(contact.direction) - contact.y * cos(contact.direction), 
    cos(contact.direction), sin(contact.direction)]

    Args:
        contact: The contact to compute the wrench for.

    Returns:
        The planar contact wrench.
    """
    phi = np.radians(contact.direction)
    return np.array([
        contact.x * np.sin(phi) - contact.y * np.cos(phi),
        np.cos(phi),
        np.sin(phi),
    ])




def plot_feasible_cor_regions(
    contacts: list[Contact],
    grid_range: float = 3.0,
    grid_resolution: int = 500,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the feasible center-of-rotation (COR) regions for a set of contacts.

    For a planar rigid body rotating about a COR at (cx, cy) with angular
    velocity ω, the velocity at contact point (x, y) along the inward contact
    normal n = (cos φ, sin φ) must be non-negative (no penetration):

        n · v_contact = ω · [cx·sin φ − cy·cos φ − (x·sin φ − y·cos φ)] ≥ 0

    Dividing by ω > 0 or ω < 0 gives two families of half-plane constraints on
    (cx, cy).  Their intersections are the feasible COR regions:
      - Blue  : CCW rotation (ω > 0) is feasible
      - Red   : CW  rotation (ω < 0) is feasible

    If both regions are empty the contacts achieve form closure.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    lin = np.linspace(-grid_range, grid_range, grid_resolution)
    cx, cy = np.meshgrid(lin, lin)

    feasible_ccw = np.ones(cx.shape, dtype=bool)
    feasible_cw  = np.ones(cx.shape, dtype=bool)

    for contact in contacts:
        phi = np.radians(contact.direction)
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)

        # Per-contact signed quantity in COR space
        lhs = cx * sin_phi - cy * cos_phi
        rhs = contact.y * cos_phi - contact.x * sin_phi

        feasible_ccw &= lhs >= rhs   # ω > 0  →  lhs ≥ rhs
        feasible_cw  &= lhs <= rhs   # ω < 0  →  lhs ≤ rhs

    for mask, color in [(feasible_ccw, "steelblue"), (feasible_cw, "tomato")]:
        if mask.any():
            ax.contourf(cx, cy, mask.astype(float), levels=[0.5, 1.5],
                        colors=[color], alpha=0.4)

    arrow_len = grid_range * 0.15
    for contact in contacts:
        phi = np.radians(contact.direction)
        ax.plot(contact.x, contact.y, "ko", markersize=7, zorder=5)
        ax.annotate(
            "",
            xy=(contact.x + arrow_len * np.cos(phi),
                contact.y + arrow_len * np.sin(phi)),
            xytext=(contact.x, contact.y),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            zorder=5,
        )

    ax.axhline(0, color="k", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.6, linestyle="--")
    ax.set_xlim(-grid_range, grid_range)
    ax.set_ylim(-grid_range, grid_range)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Feasible Center-of-Rotation Regions")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=[
        mpatches.Patch(facecolor="steelblue", alpha=0.4, label="CCW rotation (ω > 0)"),
        mpatches.Patch(facecolor="tomato",    alpha=0.4, label="CW  rotation (ω < 0)"),
    ], loc="upper right")

    return ax


if __name__ == "__main__":
    contacts = read_applied_contacts()
    f = np.full(len(contacts), 1.0)
    b = np.full(len(contacts), -1.0)
    A = np.eye(len(contacts)) * (-1)
    beq = np.zeros(3)
    contact_wrenches = np.array([compute_contact_wrench(contact) for contact in contacts]).transpose()

    print("Contact wrenches shape: ", contact_wrenches.shape)
    Aeq = contact_wrenches
    result = linprog(f, A, b, Aeq, beq)

    if result.success and result.x is not None:
        print("Linear combinations are: ", result.x)
    else:
        print("There is no form closure")

    plot_feasible_cor_regions(contacts)
    plt.tight_layout()
    plt.show()