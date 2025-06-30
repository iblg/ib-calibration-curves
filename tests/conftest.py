from pathlib import Path
import shutil
import pytest


@pytest.fixture()
def data_files(tmp_path):
    data_files = Path(tmp_path) / "test_data"
    # data_files.mkdir(parents=True, exist_ok=True)
    source_data = Path(__file__).parent / "test_apply_model_files"
    shutil.copytree(source_data, data_files)
    yield data_files
