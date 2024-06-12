"""Functionality to read experimental free energies from a supplied csv file.
   This must have the columns: calc_base_dir, name, exp_dg, exp_er"""

import os as _os

import pandas as _pd


def read_exp_dgs(dgs_file: str, base_dir: str) -> _pd.DataFrame:
    """Read the experimental free energy changes into a pandas dataframe

    Parameters
    ----------
    dgs_file : str
        Path to the experimental free energy changes file.

    base_dir: str
        The base directory to interpret the relative paths in the dgs file
        with respect to.

    Returns
    -------
    _pd.DataFrame
        Dataframe containing the experimental free energy changes
    """
    required_columns = ["calc_base_dir", "exp_dg", "exp_er", "calc_cor"]
    exp_df = _pd.read_csv(dgs_file, index_col=1)  # Use the names in the index col

    # Check that we have the required columns
    if list(exp_df.columns) != required_columns:
        raise ValueError(
            f"The experimental values file must have the columns {required_columns} but has the columns {exp_df.columns}"
        )

    # Convert the paths to absolute paths relative to the dgs file
    exp_df["calc_base_dir"] = exp_df["calc_base_dir"].apply(
        lambda x: _os.path.abspath(_os.path.join(base_dir, x))
    )

    return exp_df
