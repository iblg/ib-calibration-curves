import pytest
import pandas as pd
from pathlib import Path
from ib_calibration_curves.dilution import (
    calculate_dilution_factor,
    calculate_dilution,
)


def test_calculate_dilution_factor(data_files):
    # Taylor page ~195
    test_data_path = Path(data_files) / "dilution_data.csv"
    df = pd.read_csv(test_data_path, sep=",")
    expected = df["dilution_factor"].iloc[0]
    modified_df = calculate_dilution_factor(df)
    result = modified_df["dilution_factor"].iloc[0]

    assert result == pytest.approx(expected)
    # assert expected == pytest.approx(result, abs=1e-1)


def test_calculate_dilution(data_files):
    test_data_path = Path(data_files) / "dilution_data.csv"
    df = pd.read_csv(test_data_path, sep=",")
    df = calculate_dilution_factor(df)
    ion_name = "foo"
    result_df = calculate_dilution(df, ion_name)
    result = result_df["sample_concentration_" + ion_name].iloc[0]
    expected = df["sample_concentration_" + ion_name].iloc[0]
    assert result == pytest.approx(expected)
