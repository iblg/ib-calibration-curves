import pytest
from ib_calibration_curves.fits import load_model, save_model

bounds = (0, 100000)


#
# @pytest.fixture(autouse=True)
# def cleanup_test_file(data):
#     """ Fixture to ensure the test parent directory
#     file is removed before and after each test."""
#     if p2.exists():
#         p2.unlink()
#     yield  # This yields control to the test function


# def test_checks_parent_directory_exists(data_files):
#     p2 = data_files / 'test1'
#     y, dy, model = load_model(p)
#     # there should be nothing at p2, if cleanup_test_file is working
#     save_model(p2, y, dy, model, bounds)
#     assert pytest.raises(FileNotFoundError)


@pytest.mark.parametrize(
    "input, expected",
    [
        # C1 file already exists. expect FileExistsError
        (
            "model-dir/existing-file",
            (
                FileExistsError,
                "File: existing-file already exists. Please "
                "rerun with a different name.",
            ),
        ),
    ],
)
def test_save_model_bad(tmp_path, data_files, input, expected):
    """Should raise FileExistsError."""
    model = data_files / "test"
    model_function, model_func_uncertainty, stats_model = load_model(model)
    target_dir = tmp_path / "model-dir"
    target_dir.mkdir(parents=True, exist_ok=True)

    target_file = target_dir / "existing-file.y"
    target_file.touch()
    assert target_file.exists()
    with pytest.raises(expected[0], match=expected[1]):
        save_model(
            target_file,
            model_function,
            model_func_uncertainty,
            stats_model,
            bounds,
        )

    target_file2 = target_dir / "existing-file.dy"
    target_file2.touch()
    assert target_file.exists()
    with pytest.raises(expected[0], match=expected[1]):
        save_model(
            target_file2,
            model_function,
            model_func_uncertainty,
            stats_model,
            bounds,
        )

    target_file3 = target_dir / "existing-file.model"
    target_file3.touch()
    assert target_file.exists()
    with pytest.raises(expected[0], match=expected[1]):
        save_model(
            target_file3,
            model_function,
            model_func_uncertainty,
            stats_model,
            bounds,
        )

    # target_dir.rmdir("model-dir")
    # target_dir.exists
    # mystring = 'A directory in path to no-existing-dir
    # cannot be found.  Please rerun with a valid path.'
    # with pytest.raises(FileNotFoundError, match=mystring:
    #     save_model(target_dir, model_function,
    #     model_func_uncertainty, stats_model, bounds)


def test_checks_overwrite_willingness():
    pass


def test_saves_logfile(data_files):
    "The model should save the x bounds." ""
    p = data_files / "test_apply_model_fits"
    logfile_path = p.with_suffix(".logfile")
    y, dy, model = load_model(p)
    save_model(p, y, dy, model, bounds)
    assert logfile_path.is_file()


def test_saves_ybounds():
    "The model should save the y bounds." ""
    pass


# def test_model_summary_converts_to_string(model):
#     x = model.summary()
#     x = str(x)
#     assert isinstance(x, str)


def test_model_log_file_has_bounds():
    pass
