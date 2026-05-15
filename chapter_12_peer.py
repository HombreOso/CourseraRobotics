import csv
from datetime import datetime
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
    """Plot the feasible center-of-rotation (CoR) regions for a set of contacts.

    Each contact normal arrow divides the plane into two half-planes:
      - CORs to the LEFT  of the arrow → CCW rotation (ω > 0) is feasible
      - CORs to the RIGHT of the arrow → CW  rotation (ω < 0) is feasible

    Formally, for a CoR (cx, cy) and contact at (x, y) with inward normal
    angle φ, the velocity along the normal is:

        n · v = ω · [(x·sin φ − y·cos φ) − (cx·sin φ − cy·cos φ)] ≥ 0

    Symbol legend:
      n          Unit contact normal vector (cos φ, sin φ) — points inward into
                 the object at the contact point.
      v          Velocity of the contact point on the rigid body.
      ·          Dot product (scalar projection).
      ω          Angular velocity of the rigid body (scalar; positive = CCW,
                 negative = CW).
      x, y       Coordinates of the contact point.
      φ          Angle of the contact normal vector in radians (contact.direction
                 converted from degrees).
      cx, cy     Coordinates of the candidate center of rotation (CoR).
      x·sin φ − y·cos φ
                 Projection of the contact point onto the direction perpendicular
                 to the normal; equivalently, the signed distance of the contact
                 point from the line through the origin in direction φ.
      cx·sin φ − cy·cos φ
                 Same projection evaluated at the CoR.
      (…) − (…) Signed distance between the CoR and the contact point along the
                 perpendicular-to-normal direction — determines left vs. right.
      ≥ 0        Non-penetration condition: the contact point must not move into
                 the surface; it may move along it or away from it.

    The velocity of the contact point along the inward normal equals
    ω times the signed distance from the CoR to the contact normal line — and
    that must be non-negative for a valid (non-penetrating) motion.

    Setting s = cx·sin φ − cy·cos φ  and  s0 = x·sin φ − y·cos φ:
      ω > 0 (CCW)  →  s ≤ s0   (CoR left  of arrow)
      ω < 0 (CW)   →  s ≥ s0   (CoR right of arrow)

    The feasible CoR region is the intersection of all per-contact half-planes.
    If either intersection is empty there is no feasible motion in that sense;
    if both are empty the contacts achieve form closure.
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

        # s  = cx·sinφ − cy·cosφ  (signed distance of CoR from normal line)
        # s0 = x·sinφ  − y·cosφ  (same quantity evaluated at the contact point)
        s  = cx * sin_phi - cy * cos_phi
        s0 = contact.x * sin_phi - contact.y * cos_phi

        feasible_ccw &= s <= s0   # ω > 0: CoR must be LEFT  of arrow
        feasible_cw  &= s >= s0   # ω < 0: CoR must be RIGHT of arrow

    for mask, color in [(feasible_ccw, "tomato"), (feasible_cw, "steelblue")]:
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
        mpatches.Patch(facecolor="tomato", alpha=0.4, label="CCW rotation (ω > 0)"),
        mpatches.Patch(facecolor="steelblue",    alpha=0.4, label="CW  rotation (ω < 0)"),
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

    ax = plot_feasible_cor_regions(contacts)
    fig = ax.figure
    plt.tight_layout()
    plt.show()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Chapter_12_Plot_CoR_{timestamp}.png"
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Plot saved as {filename}")