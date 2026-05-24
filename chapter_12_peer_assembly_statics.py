"""Assembly statics checker — Chapter 12 peer assignment.

For each rigid body in a planar assembly this script asks: can the contact
forces at its interfaces balance gravity?  It answers the question by casting
it as a linear program (LP):

    find  k ≥ 0   such that   A_eq @ k = b_eq

where each column of A_eq is one edge-wrench of a friction cone at a contact,
and b_eq is the negated gravity wrench on the body.  A feasible solution means
static equilibrium is achievable within the friction constraints; no feasible
solution means the body would slide or tip.
"""

import csv
import math
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Force:
    """A planar force defined by its magnitude and absolute direction.

    Used to represent one boundary edge of a linearized 2-D friction cone.
    The two edges bracket the set of admissible contact forces for a given
    normal direction and friction coefficient.
    """
    magnitude: float          # [N]  (or any consistent force unit)
    direction_degrees: float  # [deg]  CCW from the +x axis


@dataclass
class Body:
    body_id: int    # integer label; 0 is reserved for the stationary ground
    x_com: float    # x-coordinate of the center of mass [mm]
    y_com: float    # y-coordinate of the center of mass [mm]
    mass: float     # total mass [kg]


@dataclass
class ContactDescription:
    """Raw geometric + friction data for one contact pair read from CSV.

    Convention: normal_deg is the direction the contact surface pushes INTO
    body_A (i.e. the inward normal for body_A).  The reaction on body_B is
    at normal_deg + 180°.
    """
    body_A: int       # first  body at this contact (0 = ground)
    body_B: int       # second body at this contact (0 = ground)
    x: float          # x-coordinate of the contact point [mm]
    y: float          # y-coordinate of the contact point [mm]
    normal_deg: float # angle of the inward contact normal INTO body_A [deg, CCW from +x]
    mu: float         # coefficient of friction at this contact [-]


@dataclass
class FrictionCone:
    """Precomputed planar friction cone for one contact.

    `normal` is the unit normal vector pushed into body_A.
    `friction` holds exactly two Force objects — the two boundary edges of
    the linearized 2-D cone.  Edge 1 is rotated +alpha from the normal
    (CCW) and edge 2 is rotated -alpha (CW), where alpha = arctan(mu).
    Any admissible contact force on body_A is a non-negative combination
    of these two edges (and the normal itself, which lies between them).
    """
    normal: np.ndarray     # unit normal vector (2,) pointing into body_A
    friction: list[Force]  # [edge_CCW, edge_CW]
    x_contact: float       # contact point x [mm]
    y_contact: float       # contact point y [mm]
    body_A_id: int
    body_B_id: int


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_bodies(filepath: str = "bodies_static_mass_properties.csv",
                test_equilibrium: bool = False,
                use_assembly_from_book: bool = False) -> list[Body]:
    """Read rigid-body mass properties from a CSV file.

    File-selection precedence (first match wins):
      1. use_assembly_from_book=True  → bodies_static_masses_customized.csv
      2. test_equilibrium=True        → bodies_static_mass_properties_equilibrium.csv
      3. default                      → bodies_static_mass_properties.csv

    The file may contain comment lines starting with '#' anywhere, followed
    by a header row and data rows with columns:
        body_id  – integer body label (ground = 0, not listed)
        x_com    – x-coordinate of the center of mass [mm]
        y_com    – y-coordinate of the center of mass [mm]
        mass     – total mass [kg]

    Returns a list of Body objects sorted by body_id.
    """
    bodies: list[Body] = []

    # use_assembly_from_book overrides test_equilibrium so that the customised
    # scene from the textbook can be tested independently of the equilibrium check.
    if use_assembly_from_book:
        filepath = "bodies_static_masses_customized.csv"
    elif test_equilibrium:
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


def read_contacts(filepath: str = "contacts_description.csv",
                  test_equilibrium: bool = False,
                  use_assembly_from_book: bool = False) -> list[ContactDescription]:
    """Read contact descriptions from a CSV file.

    File-selection precedence (first match wins):
      1. use_assembly_from_book=True  → contacts_description_customized.csv
      2. test_equilibrium=True        → contacts_description_equilibrium.csv
      3. default                      → contacts_description.csv

    The file may contain comment lines starting with '#' anywhere, followed
    by a header row and data rows with columns:
        body_A     – first  body at this contact (0 = ground)
        body_B     – second body at this contact (0 = ground)
        x          – x-coordinate of the contact point [mm]
        y          – y-coordinate of the contact point [mm]
        normal_deg – angle of the inward contact normal INTO body_A [deg, CCW from +x]
        mu         – coefficient of friction [-]

    Returns a list of ContactDescription objects in file order.
    """
    contacts: list[ContactDescription] = []

    # Same precedence logic as read_bodies — customised scene takes priority.
    if use_assembly_from_book:
        filepath = "contacts_description_customized.csv"
    elif test_equilibrium:
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


# ---------------------------------------------------------------------------
# Friction-cone geometry
# ---------------------------------------------------------------------------

def compute_planar_friction_cones_from_contact_list(contacts: list[ContactDescription]) -> list[FrictionCone]:
    """Vectorised wrapper: build a FrictionCone for every contact in the list."""
    return np.array([compute_planar_friction_cone_from_contact(c, c.mu) for c in contacts])


def compute_planar_friction_cone_from_contact(contact: ContactDescription,
                                              friction_coefficient: float) -> FrictionCone:
    """Build the planar friction cone for a single contact.

    The friction angle alpha = arctan(mu) is the maximum angle the resultant
    contact force may deviate from the contact normal while remaining inside
    the friction cone.  The two boundary edges of the linearised 2-D cone are
    the normal direction rotated by ±alpha.

    Edge magnitudes are set to sqrt(1 + mu²) so that the component along the
    normal direction equals 1 (unit normal force), which keeps the LP columns
    well-scaled and independent of mu.
    """
    phi   = np.radians(contact.normal_deg)  # normal angle [rad] into body_A
    alpha = np.arctan(friction_coefficient) # friction half-angle [rad]

    normal_force = np.array([np.cos(phi), np.sin(phi)])  # unit normal into body_A

    # Both edges have the same magnitude: the hypotenuse of the (1, mu) right
    # triangle, which keeps the normal component equal to 1 for both edges.
    edge_magnitude = np.linalg.norm(normal_force) * math.sqrt(1 + friction_coefficient**2)

    friction_force_1 = Force(magnitude=edge_magnitude, direction_degrees=np.degrees(phi + alpha))  # CCW edge
    friction_force_2 = Force(magnitude=edge_magnitude, direction_degrees=np.degrees(phi - alpha))  # CW  edge

    return FrictionCone(
        normal=normal_force,
        friction=[friction_force_1, friction_force_2],
        x_contact=contact.x,
        y_contact=contact.y,
        body_A_id=contact.body_A,
        body_B_id=contact.body_B,
    )


# ---------------------------------------------------------------------------
# Wrench computation
# ---------------------------------------------------------------------------

def compute_friction_cone_contact_wrench_pair_from_friction_cone(
        friction_cone: FrictionCone, force_sign: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert both friction-cone edges to planar wrenches about the world origin.

    A planar wrench is F = [m_z, f_x, f_y] with m_z = x·f_y − y·f_x.

    Args:
        friction_cone: precomputed friction cone for the contact.
        force_sign:    +1 when computing the wrench on body_A (force pushes INTO body_A),
                       -1 when computing the wrench on body_B (Newton's 3rd law reversal).

    Returns:
        (wrench_1, wrench_2) — one 3-vector per cone edge, ready to be stacked
        as columns of the LP equality matrix A_eq.
    """
    x_contact = friction_cone.x_contact
    y_contact = friction_cone.y_contact
    force_1   = friction_cone.friction[0]  # CCW cone edge
    force_2   = friction_cone.friction[1]  # CW  cone edge
    angle_1   = np.radians(force_1.direction_degrees)
    angle_2   = np.radians(force_2.direction_degrees)

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

    # ------------------------------------------------------------------
    # Scene / data-source flags
    #
    #   use_assembly_from_book = True   →  load the customized scene CSVs
    #                                      (bodies_static_masses_customized.csv
    #                                       contacts_description_customized.csv)
    #   use_assembly_from_book = False  →  fall back to the standard CSVs,
    #                                      further selected by test_equilibrium:
    #       test_equilibrium   = True   →  *_equilibrium.csv (known-good scene)
    #       test_equilibrium   = False  →  base CSV files (assignment scene)
    #
    # use_assembly_from_book takes precedence over test_equilibrium.
    # ------------------------------------------------------------------
    test_equilibrium       = True
    use_assembly_from_book = True

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
            logging.StreamHandler(),                              # console
            logging.FileHandler(log_filename, encoding="utf-8"), # txt file
        ],
    )
    log = logging.getLogger(__name__)
    log.info("Run started — log file: %s", log_filename)

    try:
        # Load bodies and contacts from the CSV files selected by the flags above.
        bodies   = read_bodies(test_equilibrium=test_equilibrium, use_assembly_from_book=use_assembly_from_book)
        contacts = read_contacts(test_equilibrium=test_equilibrium, use_assembly_from_book=use_assembly_from_book)

        log.info("Loaded %d body/bodies:", len(bodies))
        for b in bodies:
            log.info("  Body %d: CoM=(%s, %s) mm, mass=%s kg",
                     b.body_id, b.x_com, b.y_com, b.mass)

        log.info("Loaded %d contact(s):", len(contacts))
        for c in contacts:
            log.info("  Body %d <-> Body %d  at (%s, %s) mm  normal=%s deg  mu=%s",
                     c.body_A, c.body_B, c.x, c.y, c.normal_deg, c.mu)

        log.info("Run completed successfully.")

        # ------------------------------------------------------------------
        # Precompute friction cones for every contact.
        # Each cone carries two edge-wrenches that will become LP columns.
        # ------------------------------------------------------------------
        friction_cones = compute_planar_friction_cones_from_contact_list(contacts)
        log.info("Computed %d friction cones:", len(friction_cones))
        for fc in friction_cones:
            log.info("  Normal: %s, Friction: %s", fc.normal, fc.friction)

        # ------------------------------------------------------------------
        # Gravity wrench list (unused directly below but useful for debugging).
        # Each entry is [0, 0, -m·g] in world coordinates.
        # ------------------------------------------------------------------
        gravity_forces_list = np.array([[0, 0, -bodies[b.body_id-1].mass * 9.81] for b in bodies])

        # ------------------------------------------------------------------
        # Per-body LP: check whether contact forces can balance gravity.
        #
        # For each body we solve:
        #   find  k ≥ 0   such that   A_eq @ k = b_eq
        #
        # where:
        #   A_eq  — (3 × 2·n_contacts) matrix; each column is one friction-cone
        #            edge wrench [m_z, f_x, f_y] acting on this body.
        #   b_eq  — (3,) negated gravity wrench; the RHS the contact forces must match.
        #   k     — non-negative scale factors for each cone edge.
        #
        # A feasible k (linprog returns x ≠ None) means static equilibrium is
        # achievable.  An infeasible LP (x is None) means no combination of
        # admissible contact forces can cancel gravity → body cannot be in equilibrium.
        # ------------------------------------------------------------------
        resulting_body_k_arrays_dict = {}
        for body in bodies:
            current_body_id = body.body_id

            # Gravity wrench on this body about the world origin:
            #   [m_z, f_x, f_y] = [-m·g·x_com,  0,  -m·g]
            # because gravity acts at the CoM with force (0, -m·g), so
            #   m_z = x_com·(-m·g) − y_com·0 = -m·g·x_com
            current_body_gravity_wrench = np.array([
                -body.mass * 9.81 * body.x_com,
                0.0,
                -body.mass * 9.81,
            ])

            # Collect all friction cones that touch this body (from either side).
            current_body_friction_cones = [
                fc for fc in friction_cones
                if fc.body_A_id == current_body_id or fc.body_B_id == current_body_id
            ]

            # Build the list of edge-wrenches (columns of A_eq) for this body.
            current_body_wrenches = []
            for fc in current_body_friction_cones:
                # If this body is body_B the contact normal points AWAY from it,
                # so the force on it is the Newton's-3rd-law reaction: flip sign.
                forces_sign = -1 if fc.body_B_id == current_body_id else 1
                wrench_pair = compute_friction_cone_contact_wrench_pair_from_friction_cone(fc, forces_sign)
                current_body_wrenches += wrench_pair  # adds both cone-edge wrenches

            # b_eq = -gravity_wrench because the LP enforces
            #   sum_of_contact_wrenches + gravity_wrench = 0
            # →  A_eq @ k = -gravity_wrench
            current_body_beq = (-1) * current_body_gravity_wrench

            # A_eq shape: (3, 2·n_cones) — each column is one cone-edge wrench.
            current_body_Aeq = np.array(current_body_wrenches).transpose()

            # Minimise the sum of all k_i (feasibility LP; any feasible point suffices).
            current_body_f = np.full(len(current_body_wrenches), 1.0)

            print("current body id: ", current_body_id)
            print("current body beq: ", current_body_beq)
            print("current body Aeq: ", current_body_Aeq)
            log.info("current body id: %d", current_body_id)
            log.info("current body beq: %s", current_body_beq)
            log.info("current body Aeq: %s", current_body_Aeq)

            # highs-ds is the dual simplex solver inside HiGHS; reliable for
            # small equality-only LPs like this one.
            current_body_linprog_result = linprog(
                c=current_body_f,
                A_eq=current_body_Aeq,
                b_eq=current_body_beq,
                method="highs-ds",
            )

            # linprog returns x=None when the problem is infeasible (no valid k exists).
            current_body_k_array = current_body_linprog_result.x
            resulting_body_k_arrays_dict[current_body_id] = {
                "current_body_k_array": current_body_k_array,
            }

        # ------------------------------------------------------------------
        # Report results: a body is in equilibrium iff its LP was feasible.
        # total_success is True only if EVERY body is in equilibrium.
        # ------------------------------------------------------------------
        total_success = True
        for body in bodies:
            current_body_id    = body.body_id
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
