from ib_calibration_curves.fits import load_model
from pathlib import Path
import pandas as pd
import numpy as np


def set_bounds_warning(
    data,
    bounds,
    x_column_name="area",
    warning_col_name="outside_range_warning",
):
    x = data[x_column_name]
    warning_col = x.copy()
    warning_col.name = warning_col_name

    for i in range(x.shape[0]):
        if x.iloc[i] < bounds[0]:
            warning_col.iloc[i] = "low"
        elif x.iloc[i] > bounds[1]:
            warning_col.iloc[i] = "high"
        else:
            warning_col.iloc[i] = np.nan
    return warning_col


def apply_model(
    file_path: Path,
    y,
    dy,
    model,
    bounds,
    model_path: Path,
    save_to_path: Path,
    x_column_name: str = "area",
    y_column_name: str = "concentration",
    dy_column_name: str = None,
    bounds_warning_column_name: str = "outside_range_warning",
):
    """
    file_path: pathlib.Path
    The path to the data file returned by the instrument.
    y: function.
        The function that will transform the instrument signal
        to concentration.
    dy: function.
        The function that will transform the instrument signal
        to uncertainty in concentration.
    model: sm.OLS
        The statsmodels fitting result.
    x_column_name: str, default "area"
        The name of the column in the file that contains the x values.
    y_column_name: str, default "concentration"
        The name of the column in the file where you wish to store the
        concentration values.
    dy_column_name: str, default None
        The name of the column in the file where you wish to store the
        uncertainty in concentration values.
    model_path: Path, default None
        The path to the model. Should not include a suffix.
    save_to_path: bool, default True
         The path to where you wish to save the data.
         Options exist to overwrite the original data file are possible if
         save_to_path is the same as file_path."
    """

    if dy_column_name is None:
        dy_column_name = "d_" + y_column_name

    df = pd.read_excel(file_path)
    x = df[x_column_name]
    df[y_column_name] = y(x)
    df[dy_column_name] = dy(x)

    df["bounds_warning_column_name"] = set_bounds_warning(
        df,
        bounds,
        x_column_name=x_column_name,
        warning_col_name=bounds_warning_column_name,
    )
    # save fit description column
    fit_description = y_column_name + " fit done by"
    df[fit_description] = str(model_path)
    df.loc[1:, fit_description] = np.nan

    if save_to_path == file_path:
        pass
        print("Proceeding will overwrite original data spreadsheet.\n")
        print("\nReply y to proceed.")
        response = input("Anything else will stop spreadsheet creation.\n")
        if response == "y":
            pass
        else:
            return None
    df.to_excel(save_to_path, index=False)
    return df


def main():
    p = Path("/Users/ianbillinge/Documents/kimlab/projects")
    p = p / "vuv/xanthydrol/fits/2025_06_17_low"

    y, dy, model, bounds = load_model(p)
    file_path = Path("/Users/ianbillinge/Documents/kimlab/projects/")
    file_path = file_path / "vuv/xanthydrol/2025-06-17/20250617.xlsx"
    df = pd.read_excel(file_path)
    df = apply_model(
        file_path, y, dy, model, model_path=p, x_column_name="Area"
    )
    print(df)

    return


if __name__ == "__main__":
    main()
