from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd


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
    periods_per_year: int = 252
) -> pd.DataFrame:
    rows = []

    for p in portfolios:
        if p not in portfolio_values.columns:
            continue

        s = portfolio_values[p].dropna()
        if len(s) < 3:
            rows.append(
                {"Portfolio": p, "MeanRet_%": np.nan, "Vol_%": np.nan, "Sharpe": np.nan, "MaxDD_%": np.nan}
            )
            continue

        r = s.pct_change().dropna()
        m = compute_performance_metrics(r, periods_per_year=periods_per_year)
        dd = max_drawdown_from_values(s)

        rows.append(
            {
                "Portfolio": p,
                "MeanRet_%": 100.0 * m["mean_return"] if np.isfinite(m["mean_return"]) else np.nan,
                "Vol_%": 100.0 * m["volatility"] if np.isfinite(m["volatility"]) else np.nan,
                "Sharpe": m["sharpe_ratio"],
                "MaxDD_%": 100.0 * dd if np.isfinite(dd) else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["MeanRet_%", "Vol_%", "Sharpe", "MaxDD_%"]).astype(float)

    return pd.DataFrame(rows).set_index("Portfolio")
