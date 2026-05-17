import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import modern_robotics as mr
from scipy.optimize import linprog
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Contact:
    # Position of the contact point in the object's 2-D coordinate frame.
    x: float
    y: float
    # Angle (degrees) of the inward contact normal measured from the positive
    # x-axis, counter-clockwise.  "Inward" means the normal points into the
    # object body, which is the direction a frictionless finger can push.
    direction: float


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

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
        # Strip comment lines before the CSV reader sees them so that '#'
        # lines can appear anywhere — even above the header row.
        non_comment_lines = (line for line in f if not line.lstrip().startswith("#"))
        # DictReader uses the first non-comment line as the column-name header,
        # mapping each subsequent row to a dict keyed by those names.
        reader = csv.DictReader(non_comment_lines)
        for row in reader:
            contacts.append(Contact(
                x=float(row["x"]),
                y=float(row["y"]),
                direction=float(row["direction"]),
            ))

    return contacts


# ---------------------------------------------------------------------------
# Wrench computation
# ---------------------------------------------------------------------------

def compute_contact_wrench(contact: Contact) -> np.ndarray:
    """Compute the planar contact wrench for a single frictionless contact.

    A planar wrench is a 3-vector [m, fx, fy] that encodes both the force
    a contact exerts on the object and the resulting moment about the origin:

        fx = cos φ          (x-component of the unit contact normal)
        fy = sin φ          (y-component of the unit contact normal)
        m  = x·sin φ − y·cos φ   (moment arm: cross product r × n in 2-D)

    where φ is the inward contact normal angle and (x, y) is the contact
    position.  The moment arm m = r × n equals the perpendicular distance
    from the origin to the line of action of the contact force (with sign).

    Frictionless contacts can only push (normal force ≥ 0), so each wrench
    represents one ray of the contact wrench cone.  Form closure requires
    that positive combinations of these wrenches span all of wrench space.

    Args:
        contact: Contact point with position (x, y) and normal angle φ.

    Returns:
        1-D ndarray of shape (3,): the planar wrench [m, fx, fy].
    """
    # Convert the stored degrees to radians for numpy trig functions.
    phi = np.radians(contact.direction)
    return np.array([
        contact.x * np.sin(phi) - contact.y * np.cos(phi),   # moment m
        np.cos(phi),                                           # force fx
        np.sin(phi),                                           # force fy
    ])


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

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

    # Build a 2-D grid of candidate CoR positions covering the area of interest.
    # Each grid point (cx[i,j], cy[i,j]) is a potential center of rotation.
    lin = np.linspace(-grid_range, grid_range, grid_resolution)
    cx, cy = np.meshgrid(lin, lin)

    # Start with every grid point marked as feasible for both rotation senses;
    # each contact will AND-mask out the half-plane that violates its constraint.
    feasible_ccw = np.ones(cx.shape, dtype=bool)   # CCW: ω > 0
    feasible_cw  = np.ones(cx.shape, dtype=bool)   # CW:  ω < 0

    for contact in contacts:
        phi = np.radians(contact.direction)
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)

        # s  is the signed projection of every grid CoR onto the
        #    perpendicular-to-normal direction (one scalar per grid cell).
        # s0 is the same projection for the actual contact point (a scalar).
        # The sign of (s − s0) tells us whether a grid CoR lies to the left
        # or right of the contact normal line passing through (x, y).
        s  = cx * sin_phi - cy * cos_phi
        s0 = contact.x * sin_phi - contact.y * cos_phi

        # Accumulate the feasible half-plane for this contact:
        #   CCW rotation needs CoR to the LEFT  of the arrow  →  s ≤ s0
        #   CW  rotation needs CoR to the RIGHT of the arrow  →  s ≥ s0
        feasible_ccw &= s <= s0
        feasible_cw  &= s >= s0

    # Render each feasible region as a filled contour.  contourf with
    # levels=[0.5, 1.5] highlights exactly the True (1.0) cells of the
    # boolean mask cast to float.
    for mask, color in [(feasible_ccw, "tomato"), (feasible_cw, "steelblue")]:
        if mask.any():   # skip if no feasible CoR exists for this sense
            ax.contourf(cx, cy, mask.astype(float), levels=[0.5, 1.5],
                        colors=[color], alpha=0.4)

    # Draw each contact point and its inward normal arrow on the same axes
    # so the viewer can relate the shaded regions to the contacts directly.
    arrow_len = grid_range * 0.15
    for contact in contacts:
        phi = np.radians(contact.direction)
        ax.plot(contact.x, contact.y, "ko", markersize=7, zorder=5)
        ax.annotate(
            "",
            xy=(contact.x + arrow_len * np.cos(phi),   # arrow tip
                contact.y + arrow_len * np.sin(phi)),
            xytext=(contact.x, contact.y),              # arrow tail
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            zorder=5,
        )

    # Reference axes (dashed lines through the origin).
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
        mpatches.Patch(facecolor="tomato",     alpha=0.4, label="CCW rotation (ω > 0)"),
        mpatches.Patch(facecolor="steelblue",  alpha=0.4, label="CW  rotation (ω < 0)"),
    ], loc="upper right")

    return ax


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load the contact configuration from the CSV file.
    contacts = read_applied_contacts()

    # -----------------------------------------------------------------------
    # Form-closure test via linear programming (Chapter 12, Modern Robotics)
    #
    # Form closure holds when the origin of wrench space (zero net wrench) can
    # be expressed as a strictly positive combination of the contact wrenches:
    #
    #     Σ kᵢ · Fᵢ = 0,   with kᵢ > 0 for all i
    #
    # We relax "strictly positive" to kᵢ ≥ 1 (equivalent after normalisation)
    # and solve with linprog:
    #
    #   minimise   Σ kᵢ          (objective: sum of coefficients)
    #   subject to −kᵢ ≤ −1     (each coefficient ≥ 1, written as −k ≤ −1)
    #              Aeq · k = 0   (wrenches balance to zero)
    #
    # If a feasible solution exists, form closure is achieved.
    # -----------------------------------------------------------------------

    # Objective: minimise the sum of all coefficients kᵢ (linear, all ones).
    f = np.full(len(contacts), 1.0)

    # Inequality constraint −kᵢ ≤ −1, i.e. kᵢ ≥ 1 for every contact.
    # Written in matrix form:  A · k ≤ b  with A = −I, b = −1.
    b = np.full(len(contacts), -1.0)
    A = np.eye(len(contacts)) * (-1)

    # Equality constraint: the weighted sum of all contact wrenches must be zero.
    # Each column of Aeq is one contact wrench; Aeq has shape (3, n_contacts).
    beq = np.zeros(3)
    contact_wrenches = np.array(
        [compute_contact_wrench(c) for c in contacts]
    ).transpose()          # shape: (3, n_contacts)

    print("Contact wrenches shape: ", contact_wrenches.shape)

    Aeq = contact_wrenches
    result = linprog(f, A, b, Aeq, beq)

    # linprog sets result.success = True and populates result.x when a
    # feasible solution is found; otherwise the problem is infeasible and
    # result.x is None — meaning no positive combination balances the wrenches,
    # so form closure does not hold.
    if result.success and result.x is not None:
        print("Linear combinations are: ", result.x)
    else:
        print("There is no form closure")

    # -----------------------------------------------------------------------
    # Visualise the feasible center-of-rotation regions and save the figure.
    # -----------------------------------------------------------------------

    ax = plot_feasible_cor_regions(contacts)
    fig = ax.figure
    plt.tight_layout()
    plt.show()   # blocks here until the user closes the window

    # After the window is closed, save a timestamped PNG so each run produces
    # a unique, traceable output file.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Chapter_12_Plot_CoR_{timestamp}.png"
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Plot saved as {filename}")
