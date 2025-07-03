import pytest
from ib_calibration_curves.generate_hplc_urea_spreadsheet import (
    generate_measurement_spreadsheet,
)
from pathlib import Path


def test_complains_if_file_dir_does_not_exist():
    p = Path() / "tests" / "bogus_folder"
    if p.exists():
        p.rmdir()
    p = p.parent
    file_name = "bogus_file_name"
    dims = ["dummy", "dims"]
    common_dims = {"dummy": "common_dims"}
    generate_measurement_spreadsheet(p, file_name, dims, common_dims)
    assert pytest.raises(FileNotFoundError)
