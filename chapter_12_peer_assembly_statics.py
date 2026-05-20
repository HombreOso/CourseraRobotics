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

    @dataclass
    class FrictionCone:
        normal: np.ndarray
        friction: np.ndarray


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

def compute_planar_friction_cones_from_contact_list(contacts: list[ContactDescription]) -> list[FrictionCone]:
    """Compute the planar friction cones for a list of contacts.
    """
    return np.array([compute_planar_friction_cone_from_contact(c) for c in contacts])

def compute_planar_friction_cone_from_contact(contact: ContactDescription) -> FrictionCone:
    """Compute the planar friction cone for a single contact.
    """
    phi = np.radians(contact.normal_deg)
    return FrictionCone(
        normal=np.array([np.cos(phi), np.sin(phi)]),
        friction=np.array([-np.sin(phi), np.cos(phi)]),
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    import traceback

    # ------------------------------------------------------------------
    # Logging setup — one handler writes to the console, another to a
    # timestamped .txt file so every run produces a unique, traceable log.
    # The timestamp uses the moment the script starts (not when each line
    # is written), so the filename matches the execution you are reviewing.
    # ------------------------------------------------------------------
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename  = f"chapter_12_assembly_statics_{run_timestamp}.txt"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),                       # console
            logging.FileHandler(log_filename, encoding="utf-8"),  # txt file
        ],
    )
    log = logging.getLogger(__name__)
    log.info("Run started — log file: %s", log_filename)

    try:
        bodies   = read_bodies()
        contacts = read_contacts()

        log.info("Loaded %d body/bodies:", len(bodies))
        for b in bodies:
            log.info("  Body %d: CoM=(%s, %s) mm, mass=%s kg",
                     b.body_id, b.x_com, b.y_com, b.mass)

        log.info("Loaded %d contact(s):", len(contacts))
        for c in contacts:
            log.info("  Body %d <-> Body %d  at (%s, %s) mm  normal=%s deg  mu=%s",
                     c.body_A, c.body_B, c.x, c.y, c.normal_deg, c.mu)

        log.info("Run completed successfully.")

    except Exception:
        # Log the full traceback so the txt file contains enough detail to
        # diagnose the failure without needing to re-run the script.
        log.error("Run failed with an unhandled exception:\n%s", traceback.format_exc())
        raise   # re-raise so the process exits with a non-zero return code

