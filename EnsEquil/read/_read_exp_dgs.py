"""Functionality to read experimental free energies from a supplied csv file.
   This must have the columns: calc_base_dir, name, exp_dg, exp_er"""

import os as _os
import pandas as _pd


def read_exp_dgs(dgs_file: str) -> _pd.DataFrame:
    """Read the experimental free energy changes into a pandas dataframe

    Parameters
    ----------
    dgs_file : str
        Path to the experimental free energy changes file.

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

    # Convert the paths to absolute paths
    exp_df["calc_base_dir"] = exp_df["calc_base_dir"].apply(_os.path.abspath)

    return exp_df
