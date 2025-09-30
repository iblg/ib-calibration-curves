from ib_calibration_curves.fits import load_model
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_results(
    fit: dict,
    show_flag: bool = False,
    save_to_path: Path = None,
    x_label: str = None,
    y_label: str = None,
):
    """
    fit: dict
    Contains:
        'y': python function
        'dy': python function
        'results': statsmodels FittingResult object
        'model': statsmodels Model object
        'x_range': tuple
        'y_range': tuple


    model_bounds: tuple
    Lower and upper bounds of model parameter. If model_bounds is None, no
    model interpolation interval is shown. If a tuple is passed, a model
    window is shown.

    show_flag: bool, default False
    If False, no plot is shown. If True, a plot is shown.

    save_to_path: str, default None
    If None, no plot is saved. If str or Path is passed, plot is saved at
    location specified.

    x_label: str, default None
    y_label: str, default None
    """
    y_function = fit["y"]
    dy_function = fit["dy"]
    model_bounds = fit["x_range"]
    model = fit["model"]
    fig, ax = plt.subplots()

    if model_bounds is None:
        pass
    else:
        # ax.axvspan(model_bounds[0], model_bounds[1], alpha=0.2, color="Grey")
        ax.axvline(model_bounds[0], color="k", linestyle="--")
        ax.axvline(model_bounds[1], color="k", linestyle="--")

    x_data = model.exog[:, -1]  # assumes linear model with intercept
    y_data = model.endog
    ax.plot(x_data, y_data, "o", label="Training data")

    x_data.sort()
    ax.plot(
        x_data,
        y_function(x_data),
        "-",
        color="black",
        label="Model prediction",
    )
    ax.fill_between(
        x_data,
        y_function(x_data) - dy_function(x_data),
        y_function(x_data) + dy_function(x_data),
        color="black",
        alpha=0.5,
        edgecolor=None,
        label=r"Standard error region",
    )
    ax.legend(loc="best")
    str_kwargs = {"ha": "right", "va": "bottom", "transform": ax.transAxes}

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    ax.text(
        0.95, 0.05, "Std err: {:1.6f}".format(dy_function([0])), **str_kwargs
    )
    if save_to_path is None:
        pass
    else:
        plt.savefig(save_to_path)

    if show_flag:
        plt.show()
    return


def main():
    path_to_file = Path(__file__).parent
    path_to_model = (
        path_to_file.parent.parent / "tests" / "test_apply_model_files"
    )
    path_to_model = path_to_model / "example_linear_model.model"

    y, dy, model, bounds = load_model(path_to_model)
    x_data = np.random.uniform(low=bounds[0], high=bounds[1], size=100)

    plot_results(x_data, y, dy, bounds, show_flag=True)
    return


if __name__ == "__main__":
    main()
