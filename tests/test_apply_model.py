import pandas as pd
from pathlib import Path
from ib_calibration_curves.apply_model import apply_model
from ib_calibration_curves.hplc_urea import load_model


def test_apply_model_loads_dataframe():
    model_p = Path() / "tests" / "test_apply_model_files" / "test"
    file_p = Path() / "tests" / "test_apply_model_files" / "example_data.xlsx"
    y, dy, model = load_model(model_p)
    df = apply_model(file_p, y, dy, model)

    assert isinstance(df, pd.DataFrame)
