import pytest
from ib_calibration_curves.generate_hplc_urea_spreadsheet import (
    generate_measurement_spreadsheet,
)
from pathlib import Path


def test_complains_if_file_dir_does_not_exist():
    p = Path() / "tests" / "bogus_folder"
    file_name = "bogus_file_name"
    dims = ["dummy", "dims"]
    common_dims = {"dummy": "common_dims"}
    generate_measurement_spreadsheet(p, file_name, dims, common_dims)
    assert pytest.raises(NotADirectoryError)


# def test_dot_product_2D_list():
#     a = [1, 2]
#     b = [3, 4]
#     expected = 11.0
#     actual = functions.dot_product(a, b)
#     assert actual == expected
#
#
# def test_dot_product_3D_list():
#     a = [1, 2, 3]
#     b = [4, 5, 6]
#     expected = 32.0
#     actual = functions.dot_product(a, b)
#     assert actual == expected
#
#
# @pytest.mark.parametrize(
#     "a, b, expected",
#     [
#         # Test whether the dot product function works with 2D and 3D vectors
#         # C1: lists, expect correct float output
#         ([1, 2], [3, 4], 11.0),
#         ([1, 2, 3], [4, 5, 6], 32.0),
#         # C2: tuples, expect correct float output
#         ((1, 2), (3, 4), 11.0),
#         ((1, 2, 3), (4, 5, 6), 32.0),
#         # C3: numpy arrays, expect correct float output
#         (np.array([1, 2]), np.array([3, 4]), 11.0),
#         (np.array([1, 2, 3]), np.array([4, 5, 6]), 32.0),
#     ],
# )
# def test_dot_product(a, b, expected):
#     actual = functions.dot_product(a, b)
#     assert actual == expected
