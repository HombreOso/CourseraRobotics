import csv
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import modern_robotics as mr
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Body:
    body_id: int    # integer label; 0 is reserved for the stationary ground
    x_com: float    # x-coordinate of the center of mass [mm]
    y_com: float    # y-coordinate of the center of mass [mm]
    mass: float     # total mass [kg]


@dataclass
class ContactDescription:
    body_A: int       # first  body at this contact (0 = ground)
    body_B: int       # second body at this contact (0 = ground)
    x: float          # x-coordinate of the contact point [mm]
    y: float          # y-coordinate of the contact point [mm]
    normal_deg: float # angle of the inward contact normal INTO body_A [deg, CCW from +x]
    mu: float         # coefficient of friction at this contact [-]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_bodies(filepath: str = "bodies_static_mass_properties.csv") -> list[Body]:
    """Read rigid-body mass properties from a CSV file.

    The file may contain comment lines starting with '#' anywhere, followed
    by a header row and data rows with columns:
        body_id  – integer body label (ground = 0, not listed)
        x_com    – x-coordinate of the center of mass [mm]
        y_com    – y-coordinate of the center of mass [mm]
        mass     – total mass [kg]

    Returns a list of Body objects sorted by body_id.
    """
    bodies: list[Body] = []

    with open(filepath, newline="") as f:
        # Exclude comment lines so DictReader sees only the header and data.
        non_comment_lines = (line for line in f if not line.lstrip().startswith("#"))
        reader = csv.DictReader(non_comment_lines)
        for row in reader:
            bodies.append(Body(
                body_id=int(row["body_id"]),
                x_com=float(row["x_com"]),
                y_com=float(row["y_com"]),
                mass=float(row["mass"]),
            ))

    # Ensure consistent ordering regardless of file order.
    bodies.sort(key=lambda b: b.body_id)
    return bodies


def read_contacts(filepath: str = "contacts_description.csv") -> list[ContactDescription]:
    """Read contact descriptions from a CSV file.

    The file may contain comment lines starting with '#' anywhere, followed
    by a header row and data rows with columns:
        body_A     – first  body at this contact (0 = ground)
        body_B     – second body at this contact (0 = ground)
        x          – x-coordinate of the contact point [mm]
        y          – y-coordinate of the contact point [mm]
        normal_deg – angle of the inward contact normal INTO body_A [deg, CCW from +x]
        mu         – coefficient of friction [-]

    The contact normal convention: normal_deg is the direction the contact
    surface pushes INTO body_A.  The equal-and-opposite force on body_B
    is at normal_deg + 180°.

    Returns a list of ContactDescription objects in file order.
    """
    contacts: list[ContactDescription] = []

    with open(filepath, newline="") as f:
        non_comment_lines = (line for line in f if not line.lstrip().startswith("#"))
        reader = csv.DictReader(non_comment_lines)
        for row in reader:
            contacts.append(ContactDescription(
                body_A=int(row["body_A"]),
                body_B=int(row["body_B"]),
                x=float(row["x"]),
                y=float(row["y"]),
                normal_deg=float(row["normal_deg"]),
                mu=float(row["mu"]),
            ))

    return contacts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bodies   = read_bodies()
    contacts = read_contacts()

    print(f"Loaded {len(bodies)} body/bodies:")
    for b in bodies:
        print(f"  Body {b.body_id}: CoM=({b.x_com}, {b.y_com}) mm, mass={b.mass} kg")

    print(f"\nLoaded {len(contacts)} contact(s):")
    for c in contacts:
        print(
            f"  Body {c.body_A} ↔ Body {c.body_B}  "
            f"at ({c.x}, {c.y}) mm  "
            f"normal={c.normal_deg}°  μ={c.mu}"
        )
