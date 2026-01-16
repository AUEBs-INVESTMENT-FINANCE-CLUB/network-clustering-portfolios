from typing import Dict, List
import numpy as np
import pandas as pd


def compute_performance_metrics(returns: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    returns = returns.dropna()
    if len(returns) == 0 or float(returns.std()) == 0.0:
        return {"mean_return": 0.0, "volatility": 0.0, "sharpe_ratio": 0.0}
    mean_return = float(returns.mean()) * periods_per_year
    volatility = float(returns.std()) * np.sqrt(periods_per_year)
    sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0
    return {"mean_return": mean_return, "volatility": volatility, "sharpe_ratio": sharpe_ratio}


def max_drawdown_from_values(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) == 0:
        return 0.0
    equity = values / float(values.iloc[0])
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def tracking_error_simple(port_ret: pd.Series, bench_ret: pd.Series, periods_per_year: int = 252) -> float:
    port_ret = port_ret.dropna()
    bench_ret = bench_ret.dropna()
    common_idx = port_ret.index.intersection(bench_ret.index)
    if len(common_idx) == 0:
        return 0.0
    diff = port_ret.loc[common_idx] - bench_ret.loc[common_idx]
    return float(diff.std() * np.sqrt(periods_per_year))


def metrics_table_from_values(portfolio_values: pd.DataFrame, portfolios: List[str], bench_name: str = "FTSE100") -> pd.DataFrame:
    rows = []
    bench_ret = portfolio_values[bench_name].pct_change().dropna() if bench_name in portfolio_values.columns else None

    for p in portfolios:
        if p not in portfolio_values.columns:
            continue
        s = portfolio_values[p].dropna()
        if len(s) < 2:
            continue

        r = s.pct_change().dropna()
        m = compute_performance_metrics(r)
        dd = max_drawdown_from_values(s)
        te = tracking_error_simple(r, bench_ret) if (bench_ret is not None and p != bench_name) else 0.0

        rows.append(
            {
                "Portfolio": p,
                "MeanRet_%": 100.0 * m["mean_return"],
                "Vol_%": 100.0 * m["volatility"],
                "Sharpe": m["sharpe_ratio"],
                "MaxDD_%": 100.0 * dd,
                "TrackErr_%": 100.0 * te,
            }
        )

    return pd.DataFrame(rows).set_index("Portfolio") if rows else pd.DataFrame(
        columns=["MeanRet_%", "Vol_%", "Sharpe", "MaxDD_%", "TrackErr_%"]
    )
