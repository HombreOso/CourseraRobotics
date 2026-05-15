import csv
import numpy as np
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
    return np.array([contact.x * np.sin(contact.direction) - contact.y * np.cos(contact.direction), np.cos(contact.direction), np.sin(contact.direction)])




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