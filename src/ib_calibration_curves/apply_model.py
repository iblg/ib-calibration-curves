from ib_calibration_curves.fits import load_model
from pathlib import Path
import pandas as pd
import numpy as np


def apply_model(
    file_path,
    y,
    dy,
    model,
    x_column_name: str = "area",
    y_column_name: str = "concentration",
    dy_column_name: str = None,
    model_path: Path = None,
    overwrite_file: bool = True,
):

    if dy_column_name is None:
        dy_column_name = "d_" + y_column_name

    df = pd.read_excel(file_path)
    x = df[x_column_name]
    df[y_column_name] = y(x)
    df[dy_column_name] = dy(x)
    fit_description = y_column_name + " fit done by"
    df[fit_description] = str(model_path)
    df.loc[1:, fit_description] = np.nan

    if overwrite_file:
        pass
        print("Proceeding will overwrite spreadsheet.\n")
        print("\nReply y to proceed.")
        response = input("Anything else will stop spreadsheet creation.\n")
        if response == "y":
            pass
        else:
            return
    df.to_excel(file_path, index=False)
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
