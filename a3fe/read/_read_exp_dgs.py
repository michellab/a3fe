"""Functionality to read experimental free energies from a supplied csv file.
This must have the columns: calc_base_dir, name, exp_dg, exp_er"""

import os as _os

from typing import Optional as _Optional

import pandas as _pd
import numpy as _np


def read_exp_dgs(
    dgs_file: _Optional[str] = None, base_dir: _Optional[str] = None
) -> _pd.DataFrame:
    """
    Read the experimental free energy changes into a pandas dataframe. If
    the dgs file is not supplied, return an empty dataframe.

    Parameters
    ----------
    dgs_file : str, optional, default=None
        Path to the experimental free energy changes file.

    base_dir: str, optional, default=None
        The base directory to interpret the relative paths in the dgs file
        with respect to.

    Returns
    -------
    _pd.DataFrame
        Dataframe containing the experimental free energy changes
    """
    required_columns = ["calc_base_dir", "exp_dg", "exp_er", "calc_cor"]

    # Make sure that base_dir is supplied if dgs_file is supplied
    if dgs_file is not None and base_dir is None:
        raise ValueError("If dgs_file is supplied, base_dir must also be supplied")

    # If the dgs file is not supplied, create an empty dataframe
    if dgs_file is None:
        results_df = _pd.DataFrame(columns=required_columns)

    else:
        # Read the dgs file
        results_df = _pd.read_csv(
            dgs_file, index_col=1
        )  # Use the names in the index col

        # Check that we have the required columns
        if list(results_df.columns) != required_columns:
            raise ValueError(
                f"The experimental values file must have the columns {required_columns} but has the columns {results_df.columns}"
            )

        # Convert the paths to absolute paths relative to the dgs file
        results_df["calc_base_dir"] = results_df["calc_base_dir"].apply(
            lambda x: _os.path.abspath(_os.path.join(base_dir, x))
        )

    # Add colums for calculated free energies
    results_df["calc_dg"] = _np.nan
    results_df["calc_er"] = _np.nan

    return results_df
