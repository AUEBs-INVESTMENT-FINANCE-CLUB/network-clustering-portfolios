from typing import Tuple, List
import numpy as np
import pandas as pd


def load_ftse100_data(csv_file: str = "ftse_stock_prices.csv") -> Tuple[pd.DataFrame, List[str]]:
    try:
        df = pd.read_csv(csv_file, keep_default_na=True)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Please ensure the data file is present.")
        return pd.DataFrame(), []

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df.set_index("Date", inplace=True)

    if ".FTSE" in df.columns and "FTSE100" not in df.columns:
        df = df.rename(columns={".FTSE": "FTSE100"})

    components = [c for c in df.columns if c != "FTSE100"]
    return df, components


def preprocess_returns(
    data: pd.DataFrame,
    components: List[str],
    min_data_availability: float = 0.9
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available_components = [c for c in components if c in data.columns]
    if len(available_components) == 0:
        raise ValueError("No components found in data.columns")

    prices = data[available_components].copy()
    prices = prices.sort_index()

    prices = prices.dropna(how="all")
    if len(prices) == 0:
        raise ValueError("Price panel is empty after dropping all-NaN rows.")

    start_date = prices.index[0]

    col_coverage = prices.notna().mean(axis=0)
    keep = col_coverage >= float(min_data_availability)

    keep &= prices.loc[start_date].notna()

    kept_cols = keep[keep].index.tolist()
    if len(kept_cols) == 0:
        raise ValueError(
            "No stocks remain after column filtering. "
            "Lower MIN_DATA_AVAILABILITY or check if prices exist on the first in-sample date."
        )

    prices = prices[kept_cols].copy()

    prices = prices.ffill()

    prices = prices.dropna(axis=1, how="any")
    if prices.shape[1] == 0:
        raise ValueError(
            "All stocks removed after forward-fill cleanup. "
            "This usually means leading NaNs at the in-sample start date."
        )

    logret = np.log(prices).diff().dropna(how="any")

    correlation = logret.corr()
    return logret, correlation, prices
