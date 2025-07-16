from fits import load_model
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_results(
    x_data,
    y_function,
    dy_function,
    model_bounds,
    show_flag=False,
    save_to_path=None,
):
    fig, ax = plt.subplots()

    ax.axvspan(model_bounds[0], model_bounds[1], alpha=0.2, color="Grey")
    ax.errorbar(x_data, y_function(x_data), yerr=dy_function(x_data), fmt="o")
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
