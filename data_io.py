from typing import Tuple, List
import numpy as np
import pandas as pd


def load_bloomberg_data(csv_file: str = "bloomberg_prices.csv") -> Tuple[pd.DataFrame, List[str]]:
    try:
        df = pd.read_csv(csv_file, keep_default_na=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {csv_file}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df.set_index("Date", inplace=True)
    df = df.sort_index()

    index_col = None
    if "FTSE Index" in df.columns:
        index_col = "FTSE Index"
    if index_col:
        components = [c for c in df.columns if c != index_col]
    else:
        components = list(df.columns)
    return df, components


def preprocess_returns(
    data: pd.DataFrame,
    components: List[str],
    in_sample_start: str,
    in_sample_end: str,
    min_data_availability: float = 0.9,
    min_price: float = 0.01
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available_components = [c for c in components if c in data.columns]
    if len(available_components) == 0:
        raise ValueError("No components found in data.columns")

    prices_full = data[available_components].copy()
    in_sample_mask = (prices_full.index >= in_sample_start) & (prices_full.index <= in_sample_end)
    prices_in_sample = prices_full.loc[in_sample_mask].copy()

    if len(prices_in_sample) == 0:
        raise ValueError(f"No data in in-sample period {in_sample_start} to {in_sample_end}")

    start_date = prices_in_sample.index[0]
    has_start_price = prices_in_sample.loc[start_date].notna()
    in_sample_coverage = prices_in_sample.notna().mean(axis=0)
    meets_coverage = in_sample_coverage >= float(min_data_availability)
    avg_price = prices_in_sample.mean(axis=0)
    above_min_price = avg_price >= min_price
    keep_mask = has_start_price & meets_coverage & above_min_price
    kept_cols = keep_mask[keep_mask].index.tolist()

    if len(kept_cols) == 0:
        raise ValueError(
            "No stocks remain after filtering. "
            f"Filters: start_price={has_start_price.sum()}, "
            f"coverage>={min_data_availability}: {meets_coverage.sum()}, "
            f"price>={min_price}: {above_min_price.sum()}"
        )

    prices = prices_in_sample[kept_cols].copy()
    returns = np.log(prices / prices.shift(1))
    returns = returns.dropna(how="all")
    if returns.isna().any().any():
        returns = returns.dropna(axis=0, how="any")
        returns = returns.dropna(axis=1, how="any")
    if returns.shape[1] == 0:
        raise ValueError("All stocks removed after return computation")
    correlation = returns.corr()
    prices_full_filtered = prices_full[kept_cols].copy()
    return returns, correlation, prices, prices_full_filtered

