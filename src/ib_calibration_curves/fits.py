import pandas as pd
import dill
from pathlib import Path
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt


def identity(x):
    return x


def log10(x):
    return np.log10(x)


def filter_data(
    path_in: Path,
    x_column: str,
    y_column: str,
    x_bounds,
    x_transformation,
    y_transformation,
    add_X_constant: bool,
):
    """
    path_in: Path
    Path to the data file

    x_column: str
    Name of the x column in the data file

    y_column: str
    Name of the y column in the data file

    x_bounds: tuple
    Bounds over which to filter the x values

    x_transformation: function
    Transformation function for x values. Should be identity for linear fits.

    y_transformation: function
    Transformation function for y values. Should be identity for linear fits.

    add_X_constant: bool
    If True, fit will include y-intercept. If False, fit will be restricted
    to pass through the origin.
    """
    if path_in.suffix == ".csv":
        df = pd.read_csv(path_in)
    elif path_in.suffix == ".xlsx":
        df = pd.read_excel(path_in)
    else:
        print("Wrong data type")
        return None

    if x_bounds is None:
        pass
    else:
        print("Filtering over {}, {}".format(x_bounds[0], x_bounds[1]))
        # df = (
        #     df.where(df[x] > x_range[0])
        #     .where(df[x] < x_range[1])
        #     .dropna(how="all")
        # )
        df = df.loc[df[x_column] >= x_bounds[0]]
        df = df.loc[df[x_column] <= x_bounds[1]]
        # mask = df[x].notna()
        # df = df[mask]

    y_column = y_transformation(df[y_column])
    print(df.columns)
    X = x_transformation(df[x_column])
    if add_X_constant:
        X = sm.add_constant(X)
    return df, X, y_column


def get_power_law_y_function(a0, a1):
    def y(x):
        return 10**a0 * x**a1

    return y


def get_power_law_dy_function(y_func, dlogy):
    def dy(x):
        y = y_func(x)
        uncertainty = np.abs(y * np.log(10.0)) * dlogy
        return uncertainty

    return dy


def powerlawfit(
    p: Path,
    x="area",
    y="concentration",
    add_X_constant=True,
    x_range=None,
    y_transformation=np.log,
    x_transformation=np.log,
):
    df, X, y = filter_data(
        p, x, y, x_range, x_transformation, y_transformation, add_X_constant
    )

    model = sm.OLS(y, X)
    res = model.fit()

    y_func = get_power_law_y_function(res.params.iloc[0], res.params.iloc[1])
    dlogy = np.sqrt(res.mse_resid)  # get the uncertainty on the fit
    # percent_err = 100 * dlogy * np.log(10.0)

    dy_func = get_power_law_dy_function(y_func, dlogy)
    return y_func, dy_func, res


def get_exponential_y_function(a0, a1):
    def y(x):
        return a0 * (np.e ** (a1 * x))

    return y


def get_exponential_dy_function(y_func, dlogy):
    def dy(x):
        y = y_func(x)
        uncertainty = y * dlogy
        return uncertainty

    return dy


def exponentialfit(
    p: Path,
    x="area",
    y="concentration",
    add_X_constant=True,
    x_range=None,
    y_transformation=np.log,
    x_transformation=identity,
):
    """Log base e fit."""
    df, X, y = filter_data(
        p, x, y, x_range, x_transformation, y_transformation, add_X_constant
    )

    model = sm.OLS(y, X)
    res = model.fit()

    y_func = get_exponential_y_function(res.params.iloc[0], res.params.iloc[1])
    dlogy = np.sqrt(res.mse_resid)
    dy_func = get_exponential_dy_function(y_func, dlogy)
    return y_func, dy_func, res


def get_linear_y_function(a0, a1):
    def y(x):
        return a0 + a1 * x

    return y


def get_linear_dy_function(dy_val):
    def dy(x):
        return dy_val

    return dy


def linearfit(
    path_to_data_sheet: Path,
    x="area",
    y="concentration",
    add_X_constant=True,
    x_range=None,
    y_transformation=identity,
    x_transformation=identity,
):
    """
    Fit a linear function onto data.
    Arguments:
        path_to_data_sheet: pathlib.Path
            The path to the data that you wish to fit
        x: str, default "area"
            The column name corresponding to the quantity measured
            by the instrument
        y: str, default "concentration"
            The column name corresponding to the concentration
        add_X_constant: bool, default True
            If True, allow the fit to have a y-intercept parameter.
            If False, restrict the fit to pass through the origin.
        x_range: (float, float), default None
            The range over which to restrict the fit.
        y_transformation: function, default identity
            The function to use to modify the data.
            In linear fitting, the transformation should
            be the identity.
        x_transformation: function, default identity
            The function to use to modify the data.
            In linear fitting, the transformation should be
            the identity.

    Returns:
        y_func, a python function (the function allowing you
        to put in your areas and get concentrations)

        dy_func, the function returning the fitting uncertainty on the data

        res, the statsmodels FittingResults object with more statistical
        details about the fit.

        model, Statsmodels model object

        bounds: tuple, default None

    """
    df, X, y = filter_data(
        path_to_data_sheet,
        x,
        y,
        x_range,
        x_transformation,
        y_transformation,
        add_X_constant,
    )
    model = sm.OLS(y, X)
    results = model.fit()

    y_func = get_linear_y_function(
        results.params.iloc[0], results.params.iloc[1]
    )
    dy = np.sqrt(results.mse_resid)
    dy_func = get_linear_dy_function(dy)
    if x_range is None:
        x_range = (df[x].min(), df[x].max())
    else:
        pass
    y_range = y_func(np.array(x_range))
    fit = {
        "y": y_func,
        "dy": dy_func,
        "results": results,
        "model": model,
        "x_range": x_range,
        "y_range": y_range,
    }
    return fit


def save_model(
    path_out,
    fit: dict,
    y_unit: str = None,
):
    """Saves the model using dill.

    path_out: pathlib.Path. The path at which you want to save the model.

    fit: dict,
    with keys
        "y": y_func,
        "dy": dy_func,
        "results": statsmodels.FittingResults object,
        "model": statsmodels object,
        "x_range": tuple,
        "y_range": tuple,
    }
    """
    func_path = path_out.with_suffix(".y")
    err_path = path_out.with_suffix(".dy")
    mod_path = path_out.with_suffix(".model")
    info_path = path_out.with_suffix(".txt")

    def save(p, obj):
        with open(p, "wb") as outfile:
            dill.dump(obj, outfile)
        return

    save(func_path, fit["y"])
    save(err_path, fit["dy"])
    save(mod_path, fit["results"])

    def save_summary(p, fit, y_unit):
        summ = fit["results"].summary()
        summ = [str(summ)]
        summ.append(50 * "=")
        summ.append(
            "x-range: ({:1.6f}, {:1.6f})".format(
                fit["x_range"][0], fit["x_range"][1]
            )
        )
        summ.append(
            "y-range: ({:1.6f}, {:1.6f})".format(
                fit["y_range"][0], fit["y_range"][1]
            )
        )
        summ.append("Standard error: {:2.5f}".format(fit["dy"]([0])))
        summ.append("y unit: {}".format(y_unit))

        with open(p, "w") as outfile:
            outfile.write("\n".join(summ))

    save_summary(info_path, fit, y_unit)
    return


def load_model(path_in):
    """Loads model using dill.

    path_in: pathlib.Path

    Returns:
        y: function
        dy: function
        model: sm.OLS.results
        bounds: tuple
    """
    func_path = path_in.with_suffix(".y")
    err_path = path_in.with_suffix(".dy")
    mod_path = path_in.with_suffix(".model")
    bounds_path = path_in.with_suffix(".bounds")

    def load(p):
        with open(p, "rb") as infile:
            obj = dill.load(infile)
        return obj

    y = load(func_path)
    dy = load(err_path)
    model = load(mod_path)
    bounds = load(bounds_path)
    return y, dy, model, bounds


def main():
    p = Path("/Users/ianbillinge/Documents/kimlab/projects/vuv/xanthydrol/")
    infile_path = p / "20241211 HPLC Urea-Xan.xlsx"

    # fit models
    powerlaw_y, powerlaw_dy, powerlaw_model = powerlawfit(infile_path)
    exp_y, exp_dy, exponential_model = exponentialfit(infile_path)
    lin_y, lin_dy, linear_model = linearfit(infile_path)
    print(linear_model.summary())

    lowbounds = (0, 200)
    highbounds = (100, 100000)
    lin_low_y, lin_low_dy, linear_low_model = linearfit(
        infile_path, x_range=lowbounds
    )
    lin_high_y, lin_high_dy, linear_high_model = linearfit(
        infile_path, x_range=highbounds
    )

    path_out_lin_low = p / "fits" / "2025_06_17_low"
    save_model(
        path_out_lin_low,
        lin_low_y,
        lin_low_dy,
        linear_low_model,
        bounds=lowbounds,
    )

    path_out_lin_high = p / "fits" / "2025_06_17_high"
    save_model(
        path_out_lin_high,
        lin_high_y,
        lin_high_dy,
        linear_high_model,
        bounds=highbounds,
    )

    xx = "area"
    yy = "concentration"
    df, x, y = filter_data(
        infile_path,
        x_column=xx,
        y_column=yy,
        x_bounds=None,
        x_transformation=identity,
        y_transformation=identity,
        add_X_constant=True,
    )
    df_log, x_log, y_log = filter_data(
        infile_path,
        x_column=xx,
        y_column=yy,
        x_bounds=None,
        x_transformation=log10,
        y_transformation=log10,
        add_X_constant=True,
    )

    def plot_results():
        fig, ax = plt.subplots(nrows=2)

        # Plot
        x = df[xx]
        y = df[yy]
        x1 = np.linspace(x.min(), x.max(), 30)
        print(x1, lin_y(x1), lin_dy(x1))
        ax[0].plot(df_log[xx], df_log[yy], "o", label="real data")
        ax[0].errorbar(x, lin_y(x), yerr=lin_dy(x), label="linear model")
        ax[0].errorbar(
            x, lin_low_y(x), yerr=lin_low_dy(x), label="linear low model"
        )
        ax[0].errorbar(
            x, lin_high_y(x), yerr=lin_high_dy(x), label="linear high model"
        )

        ax[1].plot(x, y, "o", label="real data")
        ax[1].errorbar(
            x1,
            powerlaw_y(x1),
            yerr=powerlaw_dy(x1),
            label="power law prediction",
        )
        ax[1].errorbar(
            x1, exp_y(x1), yerr=exp_dy(x1), label="exponential prediction"
        )
        [axis.legend() for axis in ax]

        plt.show()

    plot_results()

    return


if __name__ == "__main__":
    main()
