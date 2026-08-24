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
from ib_calibration_curves.plot_fitting_data import plot_results


def check_calibrant_cols(
    data: pd.DataFrame,
    # x_col: str,
    # y_col: str,
    c_std_col: str,
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
    if c_std_col not in data.columns:
        print(f"c_stanard column {c_std_col} not found in data")
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
    analyte: str,
    c_std_col: str,
    amt_std_col: str,
    amt_diluent_cols: list[str],
    index_col: str,
    x_col: str,
    y_col: str,
    model_save_path: Path,
    method: str = "linear",
    y_unit: str = "none",
    verbose: bool = False,
    save_calibrant_data: bool = True,
    plot_results_flag: bool = True,
):
    """

    :param calibrant_info_path: Path or Path-like
    Path to the sheet showing calibrant info.
    :param data_path:

    :param analyte: str.
    Name of the analyte, i.e. 'urea' or 'h2o2'. Calibrant info sheet should
    include a column c_std_{analyte} telling the concentration of the analyte
    in the standard solution that was diluted to make the series.

    :param std_col:
    :param amt_analyte_col: str.
    Name of the column in calibrant sheet showing how much the standard was
    diluted. Examples might include "m_std" or "V_std".

    :param diluent_cols: list[str]
    List of names of columns in calibrant sheet showing how much other stuff
    was added to the calibration series. Examples might be
    ["V_diluent_A", "V_diluent_B"] or ["m_water"].
    :param index_col:
    :param x_col:
    :param y_col:
    :param model_save_path:
    :param method:
    :param y_unit:
    :param verbose:
    :param save_calibrant_data:
    :param plot_results_flag:
    :return:
    """
    cal_data = load_data(calibrant_info_path)

    if check_calibrant_cols(
        cal_data, c_std_col, index_col, amt_std_col, amt_diluent_cols
    ):
        pass
    else:
        return None

    if verbose:
        print("We have opened the calibrant info file. It looks like this:")
        print(cal_data)

    cal_data = calculate_dilution_factor(
        cal_data, analyte_column=amt_std_col, diluent_columns=amt_diluent_cols
    )
    cal_data = calculate_dilution(
        cal_data, analyte=analyte, column_base_string="c_std_"
    )

    if verbose:
        print("Dilutions have been calculated. Cal data is now:")
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None
        ):
            print(cal_data)

    def merge_areas_into_calibrant(cal_data):
        flattened = load_data(data_path)
        # cal_data = cal_data.merge(flattened, how="inner", on=index_col)
        cal_data[x_col] = cal_data[index_col].map(
            flattened.set_index(index_col)[x_col]
        )

        return cal_data

    cal_data = merge_areas_into_calibrant(cal_data)

    if save_calibrant_data:
        save_data(cal_data, calibrant_info_path)

    if verbose:
        print("We are just about to fit:")
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None
        ):
            print(cal_data)

    if method == "linear":
        model = linearfit(
            calibrant_info_path,
            x=x_col,
            y=y_col,
        )
    else:
        print(
            f"Fitting method {method} not available. Only linear is supported."
        )

    save_model(model_save_path, model, y_unit=y_unit)

    if plot_results_flag:
        plot_results(
            model,
            show_flag=True,
            save_to_path=model_save_path,
            x_label="signal",
            y_label=f"{y_col} ({y_unit})",
        )
    return
