from pathlib import Path

import pandas as pd


def assign_named_calibrant_to_spot(
    data: pd.DataFrame,
    calibrant_spots: dict,
    calibrant_info: pd.DataFrame,
    concentration_col: str,
    dropna: bool = True,
) -> pd.DataFrame:
    """Assigns calibrant solutions to spots.

    Designed for urea: the standard
    solutions are stable. Should not be used with
    :param data: Pandas DataFrame containing measurement data
    :param calibrant_spots: dict with format {spot: calibrant_name}
    i.e.,
    {'P2-E2': 'urea solution 2'}
    :param calibrant_info: Pandas DataFrame containing calibrant info.
    :param concentration_col: Name of the concentration column from the
    calibrant info dataframe that you want to use.
    :param dropna: bool, default True. If True, drops samples that are not
    calibrants (Recommended).
    :return:
    """
    for spot, calibrant in calibrant_spots.items():
        concentration = calibrant_info.loc[
            calibrant_info["name"] == calibrant, concentration_col
        ]
        data.loc[data["spot"] == spot, concentration_col] = concentration
    if dropna:
        data = data.dropna(axis="index", subset=[concentration_col])
    return data


def get_calibrant_info(std_path: Path) -> pd.DataFrame:
    """Gets calibrant information from a .csv or .xslx file.

    Requires calibrant to have already been processed.
    :param std_path:
    :return:
    """
    if std_path.suffix == ".csv":
        data = pd.read_csv(std_path)
    elif std_path.suffix == ".xlsx":
        data = pd.read_excel(std_path)
    return data


def calibrate_by_dilution(
    data: pd.DataFrame, calibrant_info_path, dropna: bool = True
) -> pd.DataFrame:

    return
