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

def read_bodies(filepath: str = "bodies_static_mass_properties.csv", test_equilibrium: bool = False) -> list[Body]:
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

    if test_equilibrium:
        filepath = "bodies_static_mass_properties_equilibrium.csv"

    with open(filepath, newline="") as f:
        # Exclude comment lines (starting with '#') and blank/whitespace-only
        # lines so DictReader always sees the header as its very first line.
        non_comment_lines = (
            line for line in f
            if line.strip() and not line.lstrip().startswith("#")
        )
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


def read_contacts(filepath: str = "contacts_description.csv", test_equilibrium: bool = False) -> list[ContactDescription]:
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

    if test_equilibrium:
        filepath = "contacts_description_equilibrium.csv"

    with open(filepath, newline="") as f:
        non_comment_lines = (
            line for line in f
            if line.strip() and not line.lstrip().startswith("#")
        )
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


def compute_friction_cone_contact_wrench_pair_from_friction_cone(
    friction_cone: FrictionCone, force_sign: int) -> tuple[np.ndarray, np.ndarray]:
    
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

    # Planar wrench  F = [m_z, f_x, f_y]   with   m_z = x·f_y − y·f_x.
    # Force vector along this friction-cone edge is magnitude·(cos θ, sin θ),
    # so the moment about the origin is magnitude·(x·sin θ − y·cos θ).
    wrench_1 = np.array([
        force_1.magnitude * (x_contact * np.sin(angle_1) - y_contact * np.cos(angle_1)),
        force_1.magnitude * np.cos(angle_1),
        force_1.magnitude * np.sin(angle_1),
    ]) * force_sign
    wrench_2 = np.array([
        force_2.magnitude * (x_contact * np.sin(angle_2) - y_contact * np.cos(angle_2)),
        force_2.magnitude * np.cos(angle_2),
        force_2.magnitude * np.sin(angle_2),
    ]) * force_sign
    return (wrench_1, wrench_2)
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    import traceback

    test_equilibrium = True

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
        bodies   = read_bodies(test_equilibrium=test_equilibrium)
        contacts = read_contacts(test_equilibrium=test_equilibrium)

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
        
        equality_constraints = []
        resulting_body_k_arrays_dict = {}
        total_success = True
        for body in bodies:
            current_body_id = body.body_id
            # Gravity force on the body is (0, −m·g) acting at its CoM.  The
            # corresponding planar wrench about the world origin is
            #   m_z = x_com·f_y − y_com·f_x = −m·g·x_com
            #   f_x = 0
            #   f_y = −m·g
            current_body_gravity_wrench = np.array([
                -body.mass * 9.81 * body.x_com,
                0.0,
                -body.mass * 9.81,
            ])
            current_body_contact_list = [c for c in contacts if c.body_A == current_body_id or c.body_B == current_body_id]
            current_body_friction_cones = [fc for fc in friction_cones if fc.body_A_id == current_body_id or fc.body_B_id == current_body_id]
            current_body_wrenches = []
            for fc in current_body_friction_cones:
                forces_sign = -1 if fc.body_B_id == current_body_id else 1
                friction_cone_wrench_pair = compute_friction_cone_contact_wrench_pair_from_friction_cone(fc, forces_sign)
                
                current_body_wrenches += friction_cone_wrench_pair
            
            current_body_beq = (-1) * current_body_gravity_wrench
            current_body_Aeq = np.array(current_body_wrenches).transpose()
            current_body_f = np.full(len(current_body_wrenches), 1.0)
            print("current body id: ", current_body_id)
            print("current body beq: ", current_body_beq)
            print("current body Aeq: ", current_body_Aeq)
            log.info("current body id: %d", current_body_id)
            log.info("current body beq: %s", current_body_beq)
            log.info("current body Aeq: %s", current_body_Aeq)
           
            current_body_linprog_result = linprog(
                c=current_body_f,  
                A_eq=current_body_Aeq, 
                b_eq=current_body_beq,
                method="highs-ds")
            current_body_k_array = current_body_linprog_result.x
            resulting_body_k_arrays_dict[current_body_id] = {
                "current_body_k_array": current_body_k_array,   
            }
        total_success = True
        for body in bodies:
            current_body_id = body.body_id
            current_body_k_array = resulting_body_k_arrays_dict[current_body_id]
            print("current body id: ", current_body_id)
            print("current body k array: ", current_body_k_array)
            log.info("current body id: %d", current_body_id)
            log.info("current body k array: %s", current_body_k_array)
            if current_body_k_array["current_body_k_array"] is not None:
                print("current body k array is not None -> Equilibrium is achieved")
                log.info("current body k array is not None -> Equilibrium is achieved")
            else:
                print("current body k array is None -> Equilibrium is not achieved")
                log.info("current body k array is None -> Equilibrium is not achieved")
            total_success = total_success and (current_body_k_array["current_body_k_array"] is not None)
        
        print("\n\n\n############## RESULT OF THE EQUILIBRIUM CHECK ################\n\n\n")
        log.info("\n\n\n############## RESULT OF THE EQUILIBRIUM CHECK ################\n\n\n")
        if total_success:
            print("Total success: Equilibrium is achieved")
            log.info("Total success: Equilibrium is achieved")
        else:
            print("Total failure: Equilibrium is not achieved")
            log.info("Total failure: Equilibrium is not achieved")

    except Exception:
        # Log the full traceback so the txt file contains enough detail to
        # diagnose the failure without needing to re-run the script.
        log.error("Run failed with an unhandled exception:\n%s", traceback.format_exc())
        raise   # re-raise so the process exits with a non-zero return code

