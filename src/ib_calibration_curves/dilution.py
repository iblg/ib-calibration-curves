import pandas as pd


def calculate_dilution_factor(data):
    data["dilution_factor"] = (
        data["amt_standard"] + data["amt_water"]
    ) / data["amt_standard"]
    return data


def calculate_dilution(
    data: pd.DataFrame, ion: str, column_base_string="standard_concentration_"
):
    ion = ion.lower()
    data["sample_concentration_" + ion] = (
        data[column_base_string + ion] / data["dilution_factor"]
    )
    return data
