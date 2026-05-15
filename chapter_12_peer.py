import csv
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
