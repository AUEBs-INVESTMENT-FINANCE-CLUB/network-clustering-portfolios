from typing import List
import numpy as np
import pandas as pd


def compute_weighted_portfolio(prices: pd.DataFrame, weights: pd.Series, name: str = "portfolio") -> pd.Series:
    if prices is None or len(prices) == 0 or weights is None or len(weights) == 0:
        return pd.Series(0.0, index=prices.index if prices is not None else None, name=name)

    w = weights.astype(float).copy()
    s = float(w.sum())
    if s == 0.0:
        return pd.Series(0.0, index=prices.index, name=name)

    X = prices.reindex(columns=w.index).copy()
    X = X.ffill()

    first = X.iloc[0].copy()
    if first.isna().any():
        miss = first.index[first.isna()]
        X.loc[X.index[0], miss] = 1.0
        X = X.ffill()

    first = X.iloc[0]
    normed = X.divide(first, axis=1)

    portfolio = normed.mul(w, axis=1).sum(axis=1)
    portfolio.name = name
    return portfolio


def compute_equal_weighted_portfolio(prices: pd.DataFrame, selected_stocks: List[str], name: str = "portfolio") -> pd.Series:
    selected = [s for s in selected_stocks if s in prices.columns]
    if len(selected) == 0:
        return pd.Series(0.0, index=prices.index, name=name)
    w = pd.Series(1.0 / len(selected), index=selected, dtype=float)
    return compute_weighted_portfolio(prices, w, name=name)


def validate_portfolio_weights(weights: pd.Series, name: str = "portfolio", tolerance: float = 1e-4) -> None:
    if weights.sum() == 0:
        print(f"Warning: {name} weights sum to zero.")
        return
    assert abs(weights.sum() - 1.0) < tolerance, f"{name}: weights sum to {weights.sum():.8f}, expected 1.0"
    assert (weights >= -1e-6).all(), f"{name}: contains negative weights"
