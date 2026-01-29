from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def diversification_ratio(weights: pd.Series, asset_returns: pd.DataFrame) -> float:
    """
    Diversification ratio: DR = (w' sigma) / sqrt(w' Sigma w),
    where sigma is the vector of asset volatilities and Sigma is the covariance matrix.
    """
    if asset_returns is None or asset_returns.empty:
        return np.nan

    w = weights.reindex(asset_returns.columns).fillna(0.0).astype(float)
    s = float(w.sum())
    if s == 0.0:
        return np.nan
    w = w / s

    cov = asset_returns.cov().values
    vol = np.sqrt(np.maximum(np.diag(cov), 0.0))
    port_var = float(w.values @ cov @ w.values)
    if not np.isfinite(port_var) or port_var <= 0:
        return np.nan
    port_vol = np.sqrt(port_var)

    weighted_avg_vol = float(w.values @ vol)
    if not np.isfinite(weighted_avg_vol) or weighted_avg_vol <= 0:
        return np.nan
    dr = weighted_avg_vol / port_vol
    return float(dr)


def compute_performance_metrics(returns: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    returns = returns.dropna()
    if len(returns) == 0 or float(returns.std(ddof=1)) == 0.0:
        return {"mean_return": np.nan, "volatility": np.nan, "sharpe_ratio": np.nan}

    mean_return = float(returns.mean()) * periods_per_year
    volatility = float(returns.std(ddof=1)) * np.sqrt(periods_per_year)
    sharpe_ratio = mean_return / volatility if volatility > 0 else np.nan
    return {"mean_return": mean_return, "volatility": volatility, "sharpe_ratio": sharpe_ratio}


def max_drawdown_from_values(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) == 0:
        return np.nan
    peak = values.cummax()
    dd = values / peak - 1.0
    return float(dd.min())


def metrics_table_from_values(
    portfolio_values: pd.DataFrame,
    portfolios: List[str],
    bench_name: str = "FTSE100",
    periods_per_year: int = 252,
    asset_returns: Optional[pd.DataFrame] = None,
    weights_map: Optional[Dict[str, pd.Series]] = None,
) -> pd.DataFrame:
    rows = []

    for p in portfolios:
        if p not in portfolio_values.columns:
            continue

        s = portfolio_values[p].dropna()
        if len(s) < 3:
            row = {"Portfolio": p, "MeanRet_%": np.nan, "Vol_%": np.nan, "Sharpe": np.nan, "MaxDD_%": np.nan}
            if asset_returns is not None and weights_map is not None:
                row["DivRatio"] = np.nan
            rows.append(row)
            continue

        r = s.pct_change().dropna()
        m = compute_performance_metrics(r, periods_per_year=periods_per_year)
        dd = max_drawdown_from_values(s)

        div_ratio = np.nan
        if asset_returns is not None and weights_map is not None and p in weights_map:
            div_ratio = diversification_ratio(weights_map[p], asset_returns)

        row = {
            "Portfolio": p,
            "MeanRet_%": 100.0 * m["mean_return"] if np.isfinite(m["mean_return"]) else np.nan,
            "Vol_%": 100.0 * m["volatility"] if np.isfinite(m["volatility"]) else np.nan,
            "Sharpe": m["sharpe_ratio"],
            "MaxDD_%": 100.0 * dd if np.isfinite(dd) else np.nan,
        }
        if asset_returns is not None and weights_map is not None:
            row["DivRatio"] = div_ratio if np.isfinite(div_ratio) else np.nan
        rows.append(row)

    base_cols = ["MeanRet_%", "Vol_%", "Sharpe", "MaxDD_%"]
    if asset_returns is not None and weights_map is not None:
        base_cols = base_cols + ["DivRatio"]
    if not rows:
        return pd.DataFrame(columns=base_cols).astype(float)

    return pd.DataFrame(rows).set_index("Portfolio")
