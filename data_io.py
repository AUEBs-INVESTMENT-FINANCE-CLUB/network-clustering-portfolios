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
    prices = prices.dropna(how="all")

    non_null_count = prices.notna().sum(axis=1)
    total_stocks = len(available_components)
    available_pct = non_null_count / total_stocks
    valid_dates = available_pct >= min_data_availability

    prices = prices.loc[valid_dates].copy()
    if len(prices) == 0:
        raise ValueError(f"No dates have at least {min_data_availability*100:.0f}% data availability")

    prices = prices.sort_index().ffill()

    all_nan_cols = prices.columns[prices.isna().all(axis=0)].tolist()
    if all_nan_cols:
        prices = prices.drop(columns=all_nan_cols)

    logret = np.log(prices).diff().dropna(how="all")
    logret = logret.dropna(how="any")

    correlation = logret.corr()
    return logret, correlation, prices
