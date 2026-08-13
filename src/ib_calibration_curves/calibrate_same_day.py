# should load an excel file describing the calibrants
# should perform concentration calculation
# should fit and save model in indicated directory

from pathlib import Path
import pandas as pd
from ib_calibration_curves.dilution import (
    calculate_dilution,
    calculate_dilution_factor,
)
from ib_calibration_curves.fits import linearfit, save_model


def check_calibrant_cols(
    data: pd.DataFrame,
    # x_col: str,
    # y_col: str,
    std_col: str,
    index_col: str,
    analyte_col: str,
    diluent_cols: str,
):
    # if x_col not in data.columns:
    #     print(f'x column {x_col} not found in data')
    #     return False
    # if y_col not in data.columns:
    #     print(f'y column {y_col} not found in data')
    #     return False
    if index_col not in data.columns:
        print(f"index column {index_col} not found in data")
        return False
    if std_col not in data.columns:
        print(f"stanard column {std_col} not found in data")
        return False
    if analyte_col not in data.columns:
        print(f"analyte column {analyte_col} not found in data")
        return False
    for c in diluent_cols:
        if c not in data.columns:
            print(f"diluent column {c} not found in data")
            return False

    return True


def load_data(p):
    suffix = p.suffix.lower()
    if suffix == ".csv":
        data = pd.read_csv(p)
    elif suffix == ".tsv":
        data = pd.read_csv(p, sep="\t")
    elif suffix == ".xlsx":
        data = pd.read_excel(p)
    return data


def save_data(data, p, idx_flag: bool = False):
    suffix = p.suffix.lower()

    if suffix == ".csv":
        data.to_csv(p, index=idx_flag)
    elif suffix == ".tsv":
        data.to_csv(p, sep="\t", index=idx_flag)
    elif suffix == ".xlsx":
        data.to_excel(p, index=idx_flag)
    return data


def calibrate_same_day(
    calibrant_info_path: Path,
    data_path: Path,
    std_col: str,
    analyte_col: str,
    diluent_cols: list[str],
    index_col: str,
    x_col: str,
    y_col: str,
    model_save_path: Path,
    method: str = "linear",
    y_unit: str = "",
    verbose: bool = False,
    save_calibrant_data: bool = True,
):
    cal_data = load_data(calibrant_info_path)

    if check_calibrant_cols(
        cal_data, std_col, index_col, analyte_col, diluent_cols
    ):
        pass
    else:
        return None

    if verbose:
        print(cal_data)

    cal_data = calculate_dilution_factor(
        cal_data, analyte_column=analyte_col, diluent_columns=diluent_cols
    )
    cal_data = calculate_dilution(
        cal_data, analyte="h2o2", column_base_string="c_std_"
    )

    if verbose:
        print(cal_data)

    def merge_areas_into_calibrant(cal_data):
        meas_data = load_data(data_path)
        # cal_data = cal_data.merge(meas_data, how='inner', on=index_col)
        print("Trying to merge.")
        print("Left is")
        print(cal_data)
        print("Right is")
        print(meas_data)
        cal_data.merge(meas_data[x_col], how="inner", on=index_col)

        return cal_data

    cal_data = merge_areas_into_calibrant(cal_data)

    if save_calibrant_data:
        save_data(cal_data, calibrant_info_path)

    if method == "linear":
        model = linearfit(
            data_path,
            x=x_col,
            y=y_col,
        )
    else:
        print(
            f"Fitting method {model} not available. Only linear is supported."
        )

    save_model(model_save_path, model, y_unit=y_unit)

    return
