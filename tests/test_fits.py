import pytest
from ib_calibration_curves.hplc_urea import linearfit, exponentialfit
from pathlib import Path


def test_exponential_fit():
    p = Path() / "tests" / "linear_fit_test_data.csv"
    y, dy, model = exponentialfit(p, x="x", y="y")
    result = model.params[0]
    expected = 11.93
    assert expected == pytest.approx(result, abs=1e-2)


def test_linear_fit():
    p = (
        Path() / "tests" / "linear_fit_test_data.csv"
    )  # the function linearfit takes a Path as the only argument
    print(p)
    y, dy, model = linearfit(p, x="x", y="y")
    result = model.params[0]
    expected = 6.9
    assert expected == pytest.approx(result, abs=1e-1)
