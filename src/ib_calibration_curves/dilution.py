import pandas as pd
import numpy as np


def calculate_dilution_factor(
    data: pd.DataFrame,
    analyte_column: str = "amt_standard",
    diluent_columns: list[str] = ["amt_water"],
    infinite_dilution_number=1e16,
) -> pd.DataFrame:
    """Calculates dilution factors. Note: for the case of infinite dilution,
    i.e. zero analyte,

    :param data: pd.DataFrame The data to apply the dilution onto.
        Typically contains masses or volumes.
    :param analyte_column: str The name of the column of the thing to be
        diluted.
    :param diluent_columns: [str] The names of the columns of all other
        components of the mixture
    :param infinite_dilution_number: float, default 1e16 The value to
        assign to infinite dilution (i.e. no analyte added).
    :return:
    """
    mixture_columns = diluent_columns
    mixture_columns.append(analyte_column)

    total_amount = data[mixture_columns].to_numpy()
    total_amount = np.sum(total_amount, axis=1)
    total_amount = pd.Series(total_amount, name="total_amount")
    data["total_amount"] = total_amount
    # data['total_amount'] = data.sum(axis='columns', skipna=True)

    dilution_factor = data["total_amount"] / data[analyte_column]
    dilution_factor = np.where(
        dilution_factor == np.inf, infinite_dilution_number, dilution_factor
    )

    data["dilution_factor"] = dilution_factor
    return data


def calculate_dilution(
    data: pd.DataFrame, ion: str, column_base_string="standard_concentration_"
):
    ion = ion.lower()
    data["sample_concentration_" + ion] = (
        data[column_base_string + ion] / data["dilution_factor"]
    )
    return data
