import pandas as pd
import numpy as np


def calculate_dilution_factor(data):
    dilution_factor = (data["amt_standard"] + data["amt_water"]) / data[
        "amt_standard"
    ]
    dilution_factor = np.where(dilution_factor == np.inf, 1.0, dilution_factor)

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
