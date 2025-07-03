import pytest
from ib_calibration_curves.urea import (
    linearfit,
    exponentialfit,
    powerlawfit,
)
from pathlib import Path


def test_exponential_fit(data_files):
    # Taylor page ~195
    p = Path(data_files) / "exponential_fit_test_data.xlsx"
    y, dy, model = exponentialfit(p, x="x", y="y")
    result = model.params.iloc[1]
    expected = 0.089
    assert expected == pytest.approx(result, abs=1e-1)


def test_linear_fit(data_files):
    # Taylor page ~185
    p = Path(data_files) / "linear_fit_test_data.xlsx"
    y, dy, model = linearfit(p, x="x", y="y")
    result = model.params.iloc[0]
    expected = 39.0
    assert expected == pytest.approx(result, abs=1e-1)


def test_power_law_fit(data_files):
    p = Path(data_files) / "power_law_fit_test_data.xlsx"
    y, dy, model = powerlawfit(p, x="x", y="y")
    result = model.params.iloc[1]
    expected = 2.787
    assert expected == pytest.approx(result, abs=1e-3)
