from typing import Dict, Tuple, List, Optional
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import OUTPUT_DIR
from metrics import compute_performance_metrics


def plotly_correlation_heatmap(
    correlation: pd.DataFrame,
    title: str = "Correlation Heatmap",
    show_values: bool = False,
    width: int = 950,
    height: int = 900
) -> go.Figure:
    corr = correlation.copy()
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
    labels = list(corr.index)

    text = None
    if show_values:
        text = np.round(corr.values, 2).astype(str)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=text,
            texttemplate="%{text}" if show_values else None,
            hovertemplate="X: %{x}<br>Y: %{y}<br>Corr: %{z:.3f}<extra></extra>",
            colorbar=dict(title="Corr"),
        )
    )

    fig.update_layout(
        title=title,
        width=width,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=120, r=40, t=70, b=120),
        xaxis=dict(tickangle=45, automargin=True),
        yaxis=dict(automargin=True),
    )
    return fig


def save_plotly(fig: go.Figure, filename: str) -> None:
    path = OUTPUT_DIR / filename
    fig.write_html(str(path), include_plotlyjs="cdn")


def save_plotly_png(fig: go.Figure, filename: str, scale: float = 2.0) -> None:
    path = OUTPUT_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(path), scale=scale)
    except Exception as e:
        warnings.warn(
            f"PNG export skipped for {filename}: {e}. Try: pip install -U kaleido plotly",
            UserWarning,
            stacklevel=2,
        )


def plot_cumulative_returns(
    portfolio_data: pd.DataFrame,
    portfolios: List[str],
    style_map: Dict[str, Tuple[str, str, str]],
    title: Optional[str] = None,
) -> go.Figure:
    fig = go.Figure()
    dash_map = {
        "-": "solid", "--": "dash", "-.": "dashdot", ":": "dot",
        "solid": "solid", "dash": "dash", "dashdot": "dashdot", "dot": "dot"
    }

    for portfolio in portfolios:
        if portfolio not in portfolio_data.columns:
            continue

        s = portfolio_data[portfolio].dropna()
        if len(s) == 0:
            continue

        pct_from_start = 100.0 * (s / s.iloc[0] - 1.0)
        color, linestyle, label = style_map.get(portfolio, ("#1f77b4", "-", portfolio))
        dash = dash_map.get(linestyle, "solid")

        fig.add_trace(
            go.Scatter(
                x=pct_from_start.index,
                y=pct_from_start.values,
                mode="lines",
                name=label,
                line=dict(width=2, dash=dash, color=color),
                hovertemplate="%{x|%Y-%m-%d}<br><b>%{y:.2f}%</b><extra>" + label + "</extra>",
            )
        )

    fig.update_layout(
        title=(title or None),
        width=1100,
        height=520,
        margin=dict(l=45, r=20, t=25, b=45),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)", title="Date"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)", title="Cumulative Return (%)"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
            font=dict(size=12),
        ),
        hovermode="x unified",
    )
    return fig


def plot_mean_variance_scatter(
    portfolio_data: pd.DataFrame,
    portfolios: List[str],
    label_map: Dict[str, str],
    color_map: Dict[str, str],
    title: str,
) -> go.Figure:
    rows = []
    for portfolio in portfolios:
        if portfolio not in portfolio_data.columns:
            continue

        values = portfolio_data[portfolio].dropna()
        if len(values) < 2:
            continue

        r = values.pct_change().dropna()
        if len(r) == 0:
            continue

        metrics = compute_performance_metrics(r)
        rows.append((portfolio, metrics["volatility"] * 100, metrics["mean_return"] * 100, metrics["sharpe_ratio"]))

    if not rows:
        fig = go.Figure()
        fig.update_layout(title=title, width=900, height=520)
        return fig

    dfp = pd.DataFrame(rows, columns=["portfolio", "vol", "ret", "sharpe"])
    dfp["label"] = dfp["portfolio"].map(lambda x: label_map.get(x, x))
    dfp["color"] = dfp["portfolio"].map(lambda x: color_map.get(x, "#1f77b4"))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dfp["vol"],
            y=dfp["ret"],
            mode="markers+text",
            text=dfp["label"],
            textposition="top center",
            marker=dict(size=14, color=dfp["color"], line=dict(width=1, color="rgba(0,0,0,0.5)"), opacity=0.9),
            customdata=np.stack([dfp["sharpe"].values, dfp["portfolio"].values], axis=1),
            hovertemplate="<b>%{text}</b><br>Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<br>Sharpe: %{customdata[0]:.3f}<extra></extra>",
            showlegend=False,
        )
    )

    xmin, xmax = dfp["vol"].min(), dfp["vol"].max()
    ymin, ymax = dfp["ret"].min(), dfp["ret"].max()
    xpad = (xmax - xmin) * 0.15 if xmax > xmin else 1.0
    ypad = (ymax - ymin) * 0.15 if ymax > ymin else 1.0

    fig.update_layout(
        title=title,
        width=900,
        height=520,
        margin=dict(l=50, r=20, t=60, b=50),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(title="Realized Volatility (%)", range=[xmin - xpad, xmax + xpad], showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(title="Realized Return (%)", range=[ymin - ypad, ymax + ypad], showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    )
    return fig
