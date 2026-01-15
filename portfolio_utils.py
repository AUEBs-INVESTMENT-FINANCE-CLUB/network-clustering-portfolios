from typing import List

import numpy as np
import pandas as pd


def compute_weighted_portfolio(prices: pd.DataFrame, weights: pd.Series, name: str = "portfolio") -> pd.Series:
    common = [c for c in weights.index if c in prices.columns]
    if not common:
        return pd.Series(0.0, index=prices.index, name=name)

    w = weights.loc[common].astype(float).values
    if w.sum() != 0:
        w = w / w.sum()

    normed = prices[common] / prices[common].iloc[0]
    portfolio = (normed * w).sum(axis=1)
    portfolio.name = name
    return portfolio


def compute_equal_weighted_portfolio(prices: pd.DataFrame, selected_stocks: List[str], name: str = "portfolio") -> pd.Series:
    selected = [s for s in selected_stocks if s in prices.columns]
    if not selected:
        return pd.Series(0.0, index=prices.index, name=name)

    normed = prices[selected] / prices[selected].iloc[0]
    portfolio = normed.mean(axis=1)
    portfolio.name = name
    return portfolio


def validate_portfolio_weights(weights: pd.Series, name: str = "portfolio", tolerance: float = 1e-4) -> None:
    if weights.sum() == 0:
        print(f"Warning: {name} weights sum to zero.")
        return
    assert abs(weights.sum() - 1.0) < tolerance, f"{name}: weights sum to {weights.sum():.8f}, expected 1.0"
    assert (weights >= -1e-6).all(), f"{name}: contains negative weights"
