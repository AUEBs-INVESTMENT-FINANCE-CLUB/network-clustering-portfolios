from typing import List
import numpy as np
import pandas as pd


def compute_weighted_portfolio(prices: pd.DataFrame, weights: pd.Series, name: str = "portfolio") -> pd.Series:
    if prices is None or len(prices) == 0 or weights is None or len(weights) == 0:
        return pd.Series(np.nan, index=prices.index if prices is not None else None, name=name)

    w = weights.astype(float).copy()
    w = w.reindex([c for c in w.index if c in prices.columns]).fillna(0.0)
    if abs(w.sum()) < 1e-10:
        return pd.Series(np.nan, index=prices.index, name=name)
    w = w / float(w.sum())

    X = prices.reindex(columns=w.index).sort_index().copy()
    X = X.ffill()

    ok = X.notna().all(axis=1)
    if not bool(ok.any()):
        return pd.Series(np.nan, index=prices.index, name=name)

    start = ok[ok].index[0]
    X2 = X.loc[start:].copy()

    first = X2.iloc[0]
    normed = X2.divide(first, axis=1)
    portfolio = normed.mul(w, axis=1).sum(axis=1)
    portfolio = portfolio.reindex(prices.index)
    portfolio.name = name
    return portfolio


def validate_portfolio_weights(weights: pd.Series, name: str = "portfolio", tolerance: float = 1e-4) -> None:
    if abs(weights.sum()) < 1e-10:
        print(f"Warning: {name} weights sum to zero.")
        return
    assert abs(float(weights.sum()) - 1.0) < tolerance, f"{name}: weights sum to {weights.sum():.8f}, expected 1.0"
    assert (weights >= -1e-6).all(), f"{name}: contains negative weights"