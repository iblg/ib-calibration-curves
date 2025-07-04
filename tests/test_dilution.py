import pytest
import pandas as pd
from pathlib import Path
from ib_calibration_curves.dilution import (
    calculate_dilution_factor,
    calculate_dilution,
)

test_files = [
    "dilution_data.csv",
    "dilution_data_no_standard_added.csv",
    "dilution_data_no_water_added.csv",
]
test_files = ["dilution_test_files/" + file for file in test_files]


@pytest.mark.parametrize("file", test_files)
def test_calculate_dilution_factor(data_files, file):
    # Taylor page ~195

    # get data
    test_data_path = Path(data_files) / file
    df = pd.read_csv(test_data_path, sep=",")

    # get expected
    expected = df["dilution_factor"].iloc[0]

    # calculate result
    modified_df = calculate_dilution_factor(df)
    result = modified_df["dilution_factor"].iloc[0]
    assert result == expected


@pytest.mark.parametrize("file", test_files)
def test_calculate_dilution(data_files, file):
    test_data_path = Path(data_files) / file
    df = pd.read_csv(test_data_path, sep=",")
    df = calculate_dilution_factor(df)
    ion_name = "foo"
    result_df = calculate_dilution(df, ion_name)
    result = result_df["sample_concentration_" + ion_name].iloc[0]
    expected = df["sample_concentration_" + ion_name].iloc[0]
    assert result == pytest.approx(expected)
