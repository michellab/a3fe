"""Functionality to manipulate SOMD simfiles"""

def read_simfile_option(simfile: str, option: str) -> str:
    """Read an option from a SOMD simfile.

    Parameters
    ----------
    simfile : str
        The path to the simfile.
    option : str
        The option to read.
    Returns
    -------
    value : str
        The value of the option.
    """
    with open(simfile, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.split("=")[0].strip() == option:
            value = line.split("=")[1].strip()
            return value
    raise ValueError(f"Option {option} not found in simfile {simfile}")

def write_simfile_option(simfile: str, option: str, value: str) -> None:
    """Write an option to a SOMD simfile.

    Parameters
    ----------
    simfile : str
        The path to the simfile.
    option : str
        The option to write.
    value : str
        The value to write.
    Returns
    -------
    None
    """
    # Read the simfile and check if the option is already present
    with open(simfile, 'r') as f:
        lines = f.readlines()
    option_line_idx = None
    for i, line in enumerate(lines):
        if line.split("=")[0].strip() == option:
                option_line_idx = i
                break

    # If the option is not present, append it to the end of the file
    if option_line_idx is None:
        lines.append(f"{option} = {value}\n")
    # Otherwise, replace the line with the new value
    else:
        lines[option_line_idx] = f"{option} = {value}\n"

    # Write the updated simfile
    with open(simfile, 'w') as f:
        f.writelines(lines)
