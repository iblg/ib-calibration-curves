import pandas as pd
import numpy as np
from pathlib import Path


def find_row_with_substring(data: pd.DataFrame, substring: str) -> pd.Index:
    """Returns an index with True for all rows containing a substring (any
    column). Case-insensitive.

    :param data: pandas.DataFrame. The DataFrame to be examined.
    :param substring: str. The substring to search for.
    :return: idx: pandas.Index. The Pandas Index object showing all rows
        containing that substring.
    """
    idx = data.apply(
        lambda row: row.astype(str)
        .str.contains(substring.lower(), case=False)
        .any(),
        axis=1,
    )
    return idx


def find_measurement_spot(df: pd.DataFrame) -> str:
    """Finds the measurement spot from the HPLC associated with the sheet of
    the spreadsheet.

    Measurement spot = P1-A1 through to P2-F11.
    :param df: pandas.DataFrame. The DataFrame to be inspected.
    :return: spot: str. The spot in the HPLC autosampler.
    """
    spot_str = r"P\d+-[A-Z]\d+"

    row, col = find_row_and_col_with_substring(df, spot_str)

    spot = df.iloc[row, col]
    spot = spot.values.item()
    return spot


def find_row_and_col_with_substring(df: pd.DataFrame, substring: str):
    """Returns the index of the row and column (as integers) of all cells in
    the DataFrame that contain a substring.

    :param df: pandas.DataFrame. The DataFrame to be examined.
    :param substring: str. The substring to search for.
    :return: row, col: (int, int)
    """
    mask = df.apply(
        lambda col: col.astype(str).str.contains(
            substring, case=False, na=False
        )
    )
    row, col = np.where(mask)
    return row, col


def read_one_page(
    data: dict[pd.DataFrame],
    page_name: str,
    print_flag: bool = False,
    detector="DAD1",
) -> pd.DataFrame:
    """
    :param data: dict[pandas.DataFrame]. The data to be processed. data should
    be a dict where the keys are the Sheet name exported by the HPLC instrument
     (one per sample). In my reports, this is "Page 1" through
     "Page N+1" for N samples. "Page 1" does not contain any sample information
     and only has information about the method.
    :param page_name: str. The name of the page that you want to read
    :param print_flag: bool, default False. If True, some printed information
    is displayed.
    :return: pandas DataFrame. The DataFrame containing the data.
    """
    if print_flag:
        print(f"reading {page_name}")

    df = data[page_name]

    spot = find_measurement_spot(df)

    df = df.dropna(axis="columns", how="all")
    df = df.dropna(axis="rows", how="all")
    df = df.reset_index(drop=True)

    sum_str = "Sum"
    sum_idx = find_row_with_substring(df, sum_str)
    sum_rows = df.loc[sum_idx].index

    default_columns = ["RT [min]", "Width [min]", "Area", "Height", "Area%"]
    empty_dataframe = pd.DataFrame(np.nan, index=[0], columns=default_columns)
    if sum_rows.empty:
        print(f"No peaks found for {page_name}")
        # data = {"spot": spot, "DAD": None, "FLD": None}
        data = {"spot": spot, "DAD": empty_dataframe, "FLD": empty_dataframe}
        return data

    def process_detector(det):
        idx = find_row_with_substring(df, det)
        rows = df.loc[idx].index
        if "DAD" in det:
            sum_row_idx = 0
        elif "FLD" in det:
            sum_row_idx = 1
        else:
            print('Unknown detector. Should be "DAD1" or "FLD".')
        if len(sum_rows) > 1:
            data = df.iloc[rows[0] + 2 : sum_rows[sum_row_idx]]
        elif len(sum_rows) == 1:
            data = df.iloc[rows[0] + 2 : sum_rows[0]]
        data = data.dropna(axis="columns", how="all")
        data = pd.DataFrame(data)
        cols = {
            "Unnamed: 2": "RT [min]",
            "Unnamed: 3": "Width [min]",
            "Unnamed: 7": "Area",
            "Unnamed: 8": "Height",
            "Unnamed: 10": "Area%",
        }
        data = data.rename(columns=cols)
        return data

    if "DAD" in detector:
        DAD_data = process_detector(detector)
    else:
        DAD_data = None

    if "FLD" in detector:
        FLD_data = process_detector(detector)
    else:
        FLD_data = None

    if print_flag:
        print(f"FLD data: \n{FLD_data}")
        print(f"DAD data: \n{DAD_data}")

    data = {"spot": spot, "DAD": DAD_data, "FLD": FLD_data}
    return data


def flatten_detector_peak_into_array(
    data: dict, peak_RTs: dict[float], detector: str
) -> pd.DataFrame:
    """Flatten peaks of interest into an array. This only processes either the
    DAD or FLD data, as specfied in argument detector.

    :param data: dict. A dict containing three items, witk keys: 'spot':
        the sample spot in the HPLC autosampler 'DAD': a
        pandas.DataFrame containing the DAD data 'FLD': a
        pandas.DataFrame containing the FLD data str (the autosampler
        spot), followed by
    :param peak_RTs: dict[dict[tuple[float,float]]]. A dict containing
        the retention time bounds for both FLD and DAD signals. For
        instance, if the DAD detector has a peak you care about that
        typically shows up at 1.5 min, and the FLD detector has two
        peaks that show up at 10.1 and 12.6 mins, you could have
        peak_RTs = { 'FLD': { 'peak_A_name': (1.4, 1.6) }, 'DAD': {
        'peak_B_name': (10.0, 10.5), 'peak_C_name': (12.0, 13.0) } }
        This would find any peaks that the machine reports between 1.4
        and 1.6 mins for the FLD. And so on.
    :param detector: str. 'FLD' or 'DAD'. The detector to process.
    :return: pandas DataFrame. The DataFrame contains the values for the
        peaks specified in peak_RTs, sample-by-sample.
    """
    try:
        peak_times = peak_RTs[detector]
    except KeyError:
        print(f"No peaks found for {detector}.")
        return pd.DataFrame()

    spots = [i["spot"] for i in data]
    data = [i[detector] for i in data]

    # print(f'Data: {data}')
    # print(f'Peak times: {peak_times}')
    # print(f'Detector: {detector}')

    def get_area_single_peak(data, name, time):
        # print(name, time)
        # print('\n\n\n\n\n')
        # [print(i['Area'].dtype) for i in data]

        areas = [
            d["Area"]
            .where(d["RT [min]"] < time[1])
            .where(d["RT [min]"] > time[0])
            for d in data
        ]
        # [print(f'\n{area}') for area in areas]
        areas = [d.dropna(axis="rows", how="all") for d in areas]
        areas = [
            float(d.item()) if d.shape[0] > 0 else float(0) for d in areas
        ]
        areas = {f"area_{detector}_{name}": areas}

        # print(area)
        # [print(f'{spot}: {a}') for spot, a in zip(spots, area)]
        return areas

    areas = []
    for name, time in peak_times.items():
        areas.append(get_area_single_peak(data, name, time))

    areas.append({"spot": spots})

    # print(areas)
    areas = [pd.DataFrame(area) for area in areas]
    areas = pd.concat(areas, axis="columns")
    return areas


def flatten_peaks_into_array(
    data: tuple[str, pd.DataFrame, pd.DataFrame], peak_RTs: dict[float]
) -> pd.DataFrame:
    """Flatten peaks of interest into an array. This only processes either the
    DAD or FLD data, as specfied in argument detector.

    :param data: dict. A dict containing three items, witk keys: 'spot':
        the sample spot in the HPLC autosampler 'DAD': a
        pandas.DataFrame containing the DAD data 'FLD': a
        pandas.DataFrame containing the FLD data str (the autosampler
        spot), followed by
    :param peak_RTs: dict[dict[tuple[float,float]]]. A dict containing
        the retention time bounds for both FLD and DAD signals. For
        instance, if the DAD detector has a peak you care about that
        typically shows up at 1.5 min, and the FLD detector has two
        peaks that show up at 10.1 and 12.6 mins, you could have
        peak_RTs = { 'FLD': { 'peak_A_name': (1.4, 1.6) }, 'DAD': {
        'peak_B_name': (10.0, 10.5), 'peak_C_name': (12.0, 13.0) } }
        This would find any peaks that the machine reports between 1.4
        and 1.6 mins for the FLD. And so on.
    :param detector: str. 'FLD' or 'DAD'. The detector to process.
    :return: pandas DataFrame. The DataFrame contains the values for the
        peaks specified in peak_RTs, sample-by-sample.
    """
    DAD_data = flatten_detector_peak_into_array(data, peak_RTs, detector="DAD")
    FLD_data = flatten_detector_peak_into_array(data, peak_RTs, detector="FLD")
    all_data = pd.concat([DAD_data, FLD_data], axis="columns")
    all_data = all_data.loc[:, ~all_data.columns.duplicated()].copy()
    return all_data


def read_long_format_ywic_export(
    path_to_data: Path,
    path_to_processed_data: Path,
    peak_RTs: dict,
    detector: str | list[str] = "DAD1",
    path_to_flattened_data: Path = None,
    print_flag: bool = False,
) -> pd.DataFrame:
    """Automatically read a long-format ywic export and save to file.

    :param
    path_to_data: Path or str indicating where to read data in from. :param
    path_to_processed_data: Path or str indicating where to write data to.
    Recommended to have this be separate from path_to_data :param peak_RTs:
    dict defining the peak retention times of interest for both the DAD and FLD
    detectors.

    :return pd.DataFrame: Your data!

    NOTE: On my computer, I ran into an error reading the exported data from
    the HPLC computer. To overcome this error, I:
        1. Downloaded the file onto my hard drive.
        2. Opened the file.
        3. Saved-as the file. You can use a new filename or just overwrite the
        original one. This re-formats the file to a modern standard Excel
        format.
        4. Used the new filepath (in my case, the original one) in my code.
    """

    data = pd.read_excel(path_to_data, sheet_name=None)
    data.pop("Page 1")  # drop the unneeded first page, which doesn't contain
    # sample data
    data = [
        read_one_page(
            data, page_name, detector=detector, print_flag=print_flag
        )
        for page_name in data.keys()
    ]
    if path_to_flattened_data is not None:
        data = flatten_peaks_into_array(data, peak_RTs)
        data.to_csv(path_to_flattened_data, index=False)
    return data
