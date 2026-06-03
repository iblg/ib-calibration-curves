import pandas as pd
import numpy as np


def find_row_with_substring(data: pd.DataFrame, substring: str):
    idx = data.apply(
        lambda row: row.astype(str)
        .str.contains(substring.lower(), case=False)
        .any(),
        axis=1,
    )
    return idx


def read_one_page(
    data: dict[pd.DataFrame], page_name: str, print_flag: bool = False
):
    if print_flag:
        print(f"reading {page_name}")
    df = data[page_name]

    def find_measurement_spot():
        """Use regex to find the measurement spot (P1-A1 to P2-F11) :return:"""
        spot_str = r"P\d+-[A-Z]\d+"

        mask = df.apply(
            lambda col: col.astype(str).str.contains(
                spot_str, case=False, na=False
            )
        )

        idx, col = np.where(mask)

        spot = df.iloc[idx, col]
        return spot

    msi = find_measurement_spot()
    print(f"Meas spot index = {msi}")
    df = df.dropna(axis="columns", how="all")
    df = df.dropna(axis="rows", how="all")
    df = df.reset_index(drop=True)
    sum_str = "sum"
    sum_idx = find_row_with_substring(df, sum_str)
    sum_rows = df.loc[sum_idx].index

    DAD_str = "DAD1"
    DAD_idx = find_row_with_substring(df, DAD_str)
    DAD_rows = df.loc[DAD_idx].index

    FLD_str = "FLD"
    FLD_idx = find_row_with_substring(df, FLD_str)
    FLD_rows = df.loc[FLD_idx].index

    DAD_data = df.iloc[DAD_rows[0] + 2 : sum_rows[0]]
    DAD_data = DAD_data.dropna(axis="columns", how="all")
    DAD_data = pd.DataFrame(DAD_data)
    cols = {
        "Unnamed: 2": "RT [min]",
        "Unnamed: 3": "Width [min]",
        "Unnamed: 7": "Area",
        "Unnamed: 8": "Height",
        "Unnamed: 10": "Area%",
    }
    DAD_data = DAD_data.rename(columns=cols)

    FLD_data = df.iloc[FLD_rows[0] + 2 : sum_rows[1]]
    FLD_data = FLD_data.dropna(axis="columns", how="all")

    FLD_data = FLD_data.rename(columns=cols)
    # with(pd.option_context('display.max_columns', None)):
    #     print(FLD_data)
    if print_flag:
        print(f"FLD data: \n{FLD_data}")
        print(f"DAD data: \n{DAD_data}")
    return DAD_data, FLD_data


def flatten_peaks_into_array(data, peak_RTs: dict[float]):
    return


def read_long_format_ywic_export(
    path_to_data, path_to_model, peak_RTs: dict[float]
):
    # to open, you need to first download the file and then save as.
    # You can save
    # in place to overwrite the initial.
    data = pd.read_excel(path_to_data, sheet_name=None)
    data.pop("Page 1")
    data = [
        read_one_page(data, page_name, print_flag=False)
        for page_name in data.keys()
    ]
    data = flatten_peaks_into_array(data, peak_RTs)

    return data
