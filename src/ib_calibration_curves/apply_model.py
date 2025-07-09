from ib_calibration_curves.fits import load_model
from pathlib import Path
import pandas as pd
import numpy as np


def apply_model(
    file_path: Path,
    y,
    dy,
    model,
    model_path: Path,
    save_to_path: Path,
    x_column_name: str = "area",
    y_column_name: str = "concentration",
    dy_column_name: str = None,
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
            return
    df.to_excel(save_to_path, index=False)
    return df


def main():
    p = Path("/Users/ianbillinge/Documents/kimlab/projects")
    p = p / "vuv/xanthydrol/fits/2025_06_17_low"

    y, dy, model = load_model(p)
    file_path = "/Users/ianbillinge/Documents/kimlab/projects/"
    file_path = file_path / "vuv/xanthydrol/2025-06-17/20250617.xlsx"
    df = pd.read_excel(file_path)
    print(df)
    df = apply_model(
        file_path, y, dy, model, model_path=p, x_column_name="Area"
    )
    print(df)

    return


if __name__ == "__main__":
    main()
