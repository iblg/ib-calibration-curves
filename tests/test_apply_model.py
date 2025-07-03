import pandas as pd
from pathlib import Path
from ib_calibration_curves.apply_model import apply_model
from ib_calibration_curves.fits import load_model

# model_p = Path() / "tests" / "test_apply_model_files" / "test"
# file_p = Path() / "tests" / "test_apply_model_files" / "example_data.xlsx"


def test_apply_model_loads_dataframe(monkeypatch, data_files):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    data_dir = Path(data_files)
    model_p = data_dir / "test"
    file_p = data_files / "example_data.xlsx"
    y, dy, model = load_model(model_p)
    df = apply_model(file_p, y, dy, model)
    assert isinstance(df, pd.DataFrame)


def test_loads_x_bounds():
    # x_bounds_path = model_p.with_suffix('.logfile')
    pass


def test_raises_problem_if_outside_range():
    """If a point is outside the range used to make the calibration curve, it
    should save in csv with a warning."""
    pass
