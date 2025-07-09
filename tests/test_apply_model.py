import pandas as pd
from pathlib import Path
from ib_calibration_curves.apply_model import apply_model, set_bounds_warning
from ib_calibration_curves.fits import load_model
from pandas.testing import assert_series_equal


def test_apply_model_loads_dataframe(monkeypatch, data_files):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    data_dir = Path(data_files)
    model_path = data_dir / "example_linear_model"
    data_path = data_files / "example_data.xlsx"
    y, dy, model, bounds = load_model(model_path)
    df = apply_model(
        data_path,
        y,
        dy,
        model,
        bounds,
        model_path,
        data_path,
    )
    assert isinstance(df, pd.DataFrame)


def test_apply_model_outside_range(data_files):
    """If a point is outside the range used to make the calibration curve, it
    should save in CSV with a warning."""
    data_path = data_files / "example_data.xlsx"
    data = pd.read_excel(data_path)

    bounds = (1090.0, 1250.0)
    result = set_bounds_warning(data, bounds)
    expected = data["outside_range_warning"]

    assert_series_equal(result, expected)
