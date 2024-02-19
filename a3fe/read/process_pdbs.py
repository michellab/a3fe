"""Some helpful functionality for processing PDB files."""

__all__ = ["remove_alt_states"]

import os as _os


def remove_alt_states(pdb_file: str) -> None:
    """Remove alternate states from a PDB file, keeping
    only the A state.

    Parameters
    ----------
    pdb_file : str
        The path to the PDB file.

    Returns
    -------
    None
    """
    # Read the PDB file
    with open(pdb_file, "r") as f:
        lines = f.readlines()

    # Remove alternate states
    new_lines = []
    print("Removing alternate states and ANISOU records...")
    for line in lines:
        # If the line doesn't start with ATOM or HETATM, add it
        if (
            not line.startswith("ATOM")
            and not line.startswith("HETATM")
            and not line.startswith("ANISOU")
        ):
            new_lines.append(line)
            continue
        # If the line starts with ANISOU, discard it
        if line.startswith("ANISOU"):
            continue
        # If the line starts with ATOM or HETATM, check the state
        state_char = line[16]
        if state_char == " ":
            new_lines.append(line)
        elif state_char == "A":
            new_line = line[:16] + " " + line[17:]
            print(f"Rewriting: \n{line}{new_line}")
            new_lines.append(new_line)
        else:
            print(f"Removing: \n{line}")

    # Save the file with _single_state appended to the name
    save_dir = _os.path.dirname(pdb_file)
    save_name = _os.path.basename(pdb_file).split(".")[0] + "_single_state.pdb"
    with open(_os.path.join(save_dir, save_name), "w") as f:
        f.writelines(new_lines)
