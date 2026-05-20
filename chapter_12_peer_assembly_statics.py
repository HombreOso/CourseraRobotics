import csv
import math
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
class Force:
    magnitude: float
    direction_degrees: float

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
    friction: list[Force]
    x_contact: float
    y_contact: float
    body_A_id: int
    body_B_id: int


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
    return np.array([compute_planar_friction_cone_from_contact(c, c.mu) for c in contacts])

def compute_planar_friction_cone_from_contact(contact: ContactDescription, friction_coefficient: float) -> FrictionCone:
    """Compute the planar friction cone for a single contact.
    """
    phi = np.radians(contact.normal_deg)  # angle of the inward contact normal INTO body_A [deg, CCW from +x]
    alpha = np.arctan(friction_coefficient) # angle of the friction force [rad] relative to the normal force
    normal_force = np.array([np.cos(phi), np.sin(phi)])
    friction_force_1_absolute_value = friction_force_2_absolute_value = np.linalg.norm(normal_force)*math.sqrt(1+friction_coefficient**2)
    friction_force_1 = Force(magnitude=friction_force_1_absolute_value, direction_degrees=np.degrees(phi+alpha))
    friction_force_2 = Force(magnitude=friction_force_2_absolute_value, direction_degrees=np.degrees(phi-alpha))
    return FrictionCone(normal=normal_force, friction=[friction_force_1, friction_force_2], x_contact=contact.x, y_contact=contact.y, body_A_id=contact.body_A, body_B_id=contact.body_B)


def compute_friction_cone_contact_wrench_pair_from_friction_cone(friction_cone: FrictionCone) -> tuple[np.ndarray, np.ndarray]:
    
    """Compute the planar contact wrench pair for a single friction cone.

    Args:
        friction_cone: Friction cone with normal and friction forces.
        x_contact: x-coordinate of the contact point [mm]
        y_contact: y-coordinate of the contact point [mm]
    Returns:
        tuple of 2-D ndarrays of shape (3,): the planar wrench pair [m, fx, fy].
    """
    x_contact = friction_cone.x_contact
    y_contact = friction_cone.y_contact
    force_1 = friction_cone.friction[0]
    force_2 = friction_cone.friction[1]
    angle_1 = np.radians(force_1.direction_degrees)
    angle_2 = np.radians(force_2.direction_degrees)

    wrench_1 = np.array([x_contact * np.sin(angle_1) - y_contact * np.cos(angle_1), 
    force_1.magnitude * np.cos(angle_1), 
    force_1.magnitude * np.sin(angle_1)])
    wrench_2 = np.array([x_contact * np.sin(angle_2) - y_contact * np.cos(angle_2), 
    force_2.magnitude * np.cos(angle_2), 
    force_2.magnitude * np.sin(angle_2)])
    return (wrench_1, wrench_2)
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

        # ---------------------------------------------------------------------------
        # Compute the planar friction cones for the contacts
        # ---------------------------------------------------------------------------
        friction_cones = compute_planar_friction_cones_from_contact_list(contacts)
        log.info("Computed %d friction cones:", len(friction_cones))
        for fc in friction_cones:
            log.info("  Normal: %s, Friction: %s", fc.normal, fc.friction)
        # ---------------------------------------------------------------------------
        # Prepare linear programming problem
        # ---------------------------------------------------------------------------
        # Earth Gravity forces
        gravity_forces_list = np.array([[0, 0, -bodies[b.body_id-1].mass * 9.81] for b in bodies])

        # ---------------------------------------------------------------------------

        # Objective: minimise the sum of all coefficients kᵢ (linear, all ones).
        f = np.full(len(contacts*2), 1.0)
        # Inequality constraint −kᵢ ≤ −1, i.e. kᵢ ≥ 1 for every contact.
        # Written in matrix form:  A · k ≤ b  with A = −I, b = −1.
        b = np.full(len(contacts*2), -1.0)
        for body in bodies:
            current_body_id = body.body_id
            current_body_contact_list = [c for c in contacts if c.body_A == current_body_id or c.body_B == current_body_id]
            current_body_friction_cones = [fc for fc in friction_cones if fc.body_A_id == current_body_id or fc.body_B_id == current_body_id]
        current_body_friction_cone_wrench_pairs = [
            # The wrench is defined with the normal pointing INTO body_A.
            # If the current body is body_B it experiences the equal-and-opposite
            # reaction (Newton's 3rd law), so the entire wrench flips sign.
            compute_friction_cone_contact_wrench_pair_from_friction_cone(fc) * (-1 if fc.body_B_id == current_body_id else 1)
            for fc in current_body_friction_cones
            if fc.body_A_id == current_body_id or fc.body_B_id == current_body_id
        ]
        print("current_body_friction_cone_wrench_pairs: ", current_body_friction_cone_wrench_pairs)
        print("current_body_contact_list: ", current_body_contact_list)
        print("current_body_friction_cones: ", current_body_friction_cones)
        print("current_body_id: ", current_body_id)
        log.info("current_body_friction_cone_wrench_pairs: %s", current_body_friction_cone_wrench_pairs)
        log.info("current_body_contact_list: %s", current_body_contact_list)
        log.info("current_body_friction_cones: %s", current_body_friction_cones)
        log.info("current_body_id: %s", current_body_id)
        print("Shape of current_body_friction_cone_wrench_pairs: ", np.shape(current_body_friction_cone_wrench_pairs))
        print("Shape of current_body_contact_list: ", np.shape(current_body_contact_list))
        print("Shape of current_body_friction_cones: ", np.shape(current_body_friction_cones))
        print("Shape of current_body_id: ", np.shape(current_body_id))
        print("Shape of gravity_forces_list: ", np.shape(gravity_forces_list))
        print("Shape of f: ", np.shape(f))
        print("Shape of b: ", np.shape(b))
        print("Shape of A: ", np.shape(A))
        print("Shape of contact_wrenches: ", np.shape(contact_wrenches))


    except Exception:
        # Log the full traceback so the txt file contains enough detail to
        # diagnose the failure without needing to re-run the script.
        log.error("Run failed with an unhandled exception:\n%s", traceback.format_exc())
        raise   # re-raise so the process exits with a non-zero return code

