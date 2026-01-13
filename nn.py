import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from typing import Tuple, List, Dict
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)

IN_SAMPLE_START = "2015"
IN_SAMPLE_END = "2017"
OUT_SAMPLE_START = "2018"
OUT_SAMPLE_END = "2025"

MIN_DATA_AVAILABILITY = 0.90
NETWORK_THRESHOLD_QUANTILE = 0.80
HIER_MAX_CLUSTERS = 10
HERC_MAX_DEPTH = 10
PERIODS_PER_YEAR = 252

KMEANS_K = 4
KMEANS_ELBOW_MIN_K = 2
KMEANS_ELBOW_MAX_K = 12
KMEANS_SCALE_FEATURES = True
KMEANS_REMOVE_OUTLIERS = True
KMEANS_OUTLIER_Z = 3.5
KMEANS_RANDOM_STATE = 42


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

    data_subset = data[available_components].copy()

    # drop rows that are fully empty across all stocks (your ",,,," rows)
    data_subset = data_subset.dropna(how="all")

    non_null_count = data_subset.notna().sum(axis=1)
    total_stocks = len(available_components)
    available_pct = non_null_count / total_stocks

    valid_dates = available_pct >= min_data_availability
    data_filtered = data_subset.loc[valid_dates].copy()
    if len(data_filtered) == 0:
        raise ValueError(f"No dates have at least {min_data_availability*100:.0f}% data availability")

    # forward-fill remaining gaps (only within valid dates)
    data_filled = data_filtered.sort_index().ffill()

    # remove stocks that are still completely NaN after filtering/fill
    all_nan_cols = data_filled.columns[data_filled.isna().all(axis=0)].tolist()
    if all_nan_cols:
        data_filled = data_filled.drop(columns=all_nan_cols)

    pctreturn = data_filled.pct_change().dropna(how="all")

    # IMPORTANT: this drops ANY date that has ANY missing return across remaining stocks.
    # Keeping it (your original behavior) gives a clean full panel for cov/corr methods.
    pctreturn = pctreturn.dropna(how="any")

    correlation = pctreturn.corr()
    return pctreturn, correlation, data_filled


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


def validate_portfolio_value(portfolio_value: pd.Series, name: str = "portfolio", tolerance: float = 0.01) -> None:
    if len(portfolio_value) == 0:
        return
    first_value = portfolio_value.iloc[0]
    assert abs(first_value - 1.0) < tolerance, f"{name}: portfolio starts at {first_value:.6f}, expected ~1.0"


def plotly_network(
    graph: nx.Graph,
    layout: dict,
    title: str = "Network",
    selected: list | None = None,
    node_score: dict | None = None,
    show_labels: bool = False,
    width: int = 1000,
    height: int = 700,
    selected_color: str = "#EC2049",
    default_color: str = "#D3D3D3",
    color_mode: str = "score",
):
    if selected is None:
        selected = []
    if node_score is None:
        node_score = {}

    nodes = list(graph.nodes())
    if len(nodes) == 0:
        fig = go.Figure()
        fig.update_layout(title=title, width=width, height=height)
        return fig

    edge_x, edge_y = [], []
    for u, v, d in graph.edges(data=True):
        x0, y0 = layout.get(u, (0, 0))
        x1, y1 = layout.get(v, (0, 0))
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="rgba(160,160,160,0.35)"),
        hoverinfo="skip",
        showlegend=False,
    )

    x = [layout.get(n, (0, 0))[0] for n in nodes]
    y = [layout.get(n, (0, 0))[1] for n in nodes]

    degrees = dict(graph.degree())
    deg_vals = np.array([degrees.get(n, 0) for n in nodes], dtype=float)
    score_vals = np.array([float(node_score.get(n, 0.0)) for n in nodes], dtype=float)

    score_norm = score_vals / score_vals.max() if score_vals.max() > 0 else score_vals

    size = 10 + 20 * (deg_vals / (deg_vals.max() if deg_vals.max() > 0 else 1.0)) + 25 * score_norm
    size = np.clip(size, 10, 45)

    is_selected = np.array([n in selected for n in nodes])

    if color_mode == "selected":
        color_vals = np.where(is_selected, selected_color, default_color)
        colorscale = None
        showscale = False
        colorbar = None
    else:
        color_vals = score_vals if score_vals.max() > 0 else deg_vals
        colorscale = "Plasma"
        showscale = True
        colorbar = dict(title="Score")

    line_width = np.where(is_selected, 2.5, 1.0)
    line_color = np.where(is_selected, "rgba(0,0,0,0.85)", "rgba(0,0,0,0.35)")

    hovertext = [
        f"<b>{n}</b><br>degree: {degrees.get(n, 0)}<br>score: {float(node_score.get(n, 0.0)):.4f}"
        for n in nodes
    ]

    node_trace = go.Scatter(
        x=x,
        y=y,
        mode="markers+text" if show_labels else "markers",
        text=nodes if show_labels else None,
        textposition="top center",
        hovertext=hovertext,
        hoverinfo="text",
        marker=dict(
            size=size,
            color=color_vals,
            colorscale=colorscale,
            showscale=showscale,
            colorbar=colorbar,
            line=dict(width=line_width, color=line_color),
            opacity=0.92,
        ),
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def plot_cumulative_returns(
    portfolio_data: pd.DataFrame,
    portfolios: List[str],
    style_map: Dict[str, Tuple[str, str, str]],
    title: str,
) -> None:
    fig = go.Figure()

    dash_map = {
        "-": "solid", "--": "dash", "-.": "dashdot", ":": "dot",
        "solid": "solid", "dash": "dash", "dashdot": "dashdot", "dot": "dot"
    }

    for portfolio in portfolios:
        if portfolio in portfolio_data.columns:
            s = portfolio_data[portfolio].dropna()
            if len(s) == 0:
                continue
            normalized = s / s.iloc[0]
            color, linestyle, label = style_map.get(portfolio, ("#1f77b4", "-", portfolio))
            dash = dash_map.get(linestyle, "solid")

            fig.add_trace(
                go.Scatter(
                    x=normalized.index,
                    y=normalized.values,
                    mode="lines",
                    name=label,
                    line=dict(width=2, dash=dash, color=color),
                    hovertemplate="%{x|%Y-%m-%d}<br><b>%{y:.4f}</b><extra>" + label + "</extra>",
                )
            )

    fig.update_layout(
        title=title,
        width=1100,
        height=520,
        margin=dict(l=40, r=20, t=60, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)", title="Date"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)", title="Normalized Value"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.show()


def plot_mean_variance_scatter(
    portfolio_data: pd.DataFrame,
    portfolios: List[str],
    label_map: Dict[str, str],
    color_map: Dict[str, str],
    title: str,
) -> None:
    rows = []
    for portfolio in portfolios:
        if portfolio in portfolio_data.columns:
            returns = portfolio_data[portfolio].pct_change().dropna()
            if len(returns) == 0:
                continue
            metrics = compute_performance_metrics(returns)
            rows.append((portfolio, metrics["volatility"] * 100, metrics["mean_return"] * 100, metrics["sharpe_ratio"]))

    if not rows:
        go.Figure().update_layout(title=title, width=900, height=520).show()
        return

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
    fig.show()


def compute_markowitz_weights(returns: pd.DataFrame, allow_short: bool = False) -> Tuple[pd.Series, pd.Series]:
    returns = returns.dropna(axis=1, how="all")
    mu = returns.mean()
    Sigma = returns.cov()
    Sigma_inv = np.linalg.pinv(Sigma.values)
    ones = np.ones(len(mu))

    w_minvar = Sigma_inv @ ones
    denom_minvar = ones @ w_minvar
    if denom_minvar != 0:
        w_minvar = w_minvar / denom_minvar

    w_maxsharpe = Sigma_inv @ mu.values
    denom_maxsharpe = w_maxsharpe.sum()
    if denom_maxsharpe != 0:
        w_maxsharpe = w_maxsharpe / denom_maxsharpe

    if not allow_short:
        w_minvar = np.maximum(w_minvar, 0.0)
        if w_minvar.sum() > 0:
            w_minvar = w_minvar / w_minvar.sum()

        w_maxsharpe = np.maximum(w_maxsharpe, 0.0)
        if w_maxsharpe.sum() > 0:
            w_maxsharpe = w_maxsharpe / w_maxsharpe.sum()

    w_minvar = pd.Series(w_minvar, index=mu.index, name="markowitz_minvar")
    w_maxsharpe = pd.Series(w_maxsharpe, index=mu.index, name="markowitz_maxsharpe")
    return w_minvar, w_maxsharpe


def create_graph(components: List[str], correlation: pd.DataFrame, threshold: float) -> Tuple[nx.Graph, Dict]:
    graph = nx.Graph()
    valid_components = [c for c in components if c in correlation.index and c in correlation.columns]
    graph.add_nodes_from(valid_components)

    for i in range(len(valid_components)):
        for j in range(i + 1, len(valid_components)):
            stock_i = valid_components[i]
            stock_j = valid_components[j]
            corr_val = correlation.loc[stock_i, stock_j]
            if corr_val > threshold:
                graph.add_edge(stock_i, stock_j, weight=corr_val)

    layout = nx.spring_layout(graph, seed=42) if len(graph.nodes) > 0 else {}
    return graph, layout


def create_full_correlation_graph(components: List[str], correlation: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    valid_components = [c for c in components if c in correlation.index and c in correlation.columns]
    graph.add_nodes_from(valid_components)

    for i in range(len(valid_components)):
        for j in range(i + 1, len(valid_components)):
            stock_i = valid_components[i]
            stock_j = valid_components[j]
            corr_val = correlation.loc[stock_i, stock_j]
            weight = max(corr_val, 0.0)
            if weight > 0:
                graph.add_edge(stock_i, stock_j, weight=weight)
    return graph


def degeneracy_ordering(graph: nx.Graph, components: List[str]) -> Tuple[List[str], List[str]]:
    isolated = [node for node in graph.nodes() if graph.degree(node) == 0]
    non_isolated = [node for node in graph.nodes() if graph.degree(node) > 0]

    if non_isolated:
        subgraph = graph.subgraph(non_isolated)
        degeneracy = dict(sorted(nx.core_number(subgraph).items(), key=lambda x: x[1]))
        independence = []
        for i in degeneracy:
            if not set(subgraph.neighbors(i)).intersection(set(independence)):
                independence.append(i)
    else:
        independence = []

    return isolated, independence


def eigenvector_centrality_weights(graph: nx.Graph, components: List[str], inverse: bool = False) -> pd.Series:
    if len(graph.nodes) == 0:
        return pd.Series(0.0, index=components, name="eigen_centrality")

    try:
        ec_dict = nx.eigenvector_centrality(graph, weight="weight", max_iter=2000)
    except Exception:
        ec_dict = nx.eigenvector_centrality_numpy(graph, weight="weight")

    ec = pd.Series(ec_dict, name="eigen_centrality").reindex(components).fillna(0.0)

    if inverse:
        ec = 1.0 / (ec.replace(0, np.nan))
        ec = ec.fillna(0.0)

    if ec.sum() != 0:
        ec = ec / ec.sum()
    return ec


def corr_distance_matrix(correlation: pd.DataFrame) -> np.ndarray:
    corr_values = correlation.values.copy()
    corr_values = np.clip(corr_values, -1.0, 1.0)
    np.fill_diagonal(corr_values, 1.0)
    distance = np.sqrt(np.maximum(0.0, 0.5 * (1 - corr_values)))
    np.fill_diagonal(distance, 0.0)
    return distance


def corr_distance_of_distance_matrix(correlation: pd.DataFrame) -> np.ndarray:
    D = corr_distance_matrix(correlation)
    n = D.shape[0]
    Dtilde = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(D[:, i] - D[:, j])
            Dtilde[i, j] = dist
            Dtilde[j, i] = dist
    np.fill_diagonal(Dtilde, 0.0)
    return Dtilde


def hierarchical_clusters_from_corr(correlation: pd.DataFrame, method: str = "average", max_clusters: int = 10) -> Tuple[pd.Series, np.ndarray]:
    corr = correlation.copy().fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)
    corr = corr.clip(-1.0, 1.0)

    dist = np.sqrt(np.maximum(0.0, 0.5 * (1 - corr.values)))
    dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=1.0)
    condensed = squareform(dist, checks=False)

    Z = linkage(condensed, method=method)
    labels = fcluster(Z, max_clusters, criterion="maxclust")
    labels = pd.Series(labels, index=correlation.index, name="cluster")
    return labels, Z


def cluster_equal_weights(labels: pd.Series) -> pd.Series:
    weights = pd.Series(0.0, index=labels.index)
    clusters = labels.unique()
    n_clusters = len(clusters)
    for c in clusters:
        members = labels.index[labels == c]
        if len(members) == 0:
            continue
        weights.loc[members] = 1.0 / (n_clusters * len(members))
    weights.name = "cluster_equal"
    return weights


def hierarchical_one_over_n_weights(correlation: pd.DataFrame, linkage_method: str = "ward", use_distance_of_distance: bool = False) -> pd.Series:
    corr = correlation.copy().fillna(0.0).clip(-1.0, 1.0)
    np.fill_diagonal(corr.values, 1.0)
    distance_matrix = corr_distance_of_distance_matrix(corr) if use_distance_of_distance else corr_distance_matrix(corr)
    condensed_dist = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed_dist, method=linkage_method)

    n_items = len(correlation.index)
    root_cluster_id = 2 * n_items - 2
    weights_dict = {i: 0.0 for i in range(n_items)}

    def assign_weight_recursive(cluster_id: int, weight: float):
        if cluster_id < n_items:
            weights_dict[cluster_id] = weight
        else:
            cluster_idx = cluster_id - n_items
            left_child = int(linkage_matrix[cluster_idx, 0])
            right_child = int(linkage_matrix[cluster_idx, 1])
            assign_weight_recursive(left_child, weight / 2.0)
            assign_weight_recursive(right_child, weight / 2.0)

    assign_weight_recursive(root_cluster_id, 1.0)
    weights = pd.Series([weights_dict[i] for i in range(n_items)], index=correlation.index)
    weights = weights / weights.sum()
    weights.name = "hier_1N"
    return weights


def compute_inverse_variance_portfolio(cov_matrix: np.ndarray) -> np.ndarray:
    variances = np.diag(cov_matrix)
    inv_variances = 1.0 / np.maximum(variances, 1e-10)
    weights = inv_variances / inv_variances.sum()
    return weights


def compute_cluster_variance(cov_matrix: np.ndarray) -> float:
    if cov_matrix.shape[0] == 1:
        return cov_matrix[0, 0]
    w = compute_inverse_variance_portfolio(cov_matrix)
    return float(w.T @ cov_matrix @ w)


def quasi_diagonalize(linkage_matrix: np.ndarray, n_items: int) -> List[int]:
    sorted_index = [int(linkage_matrix[-1, 0]), int(linkage_matrix[-1, 1])]
    while True:
        to_expand = []
        for i, idx in enumerate(sorted_index):
            if idx >= n_items:
                to_expand.append((i, idx - n_items))
        if not to_expand:
            break
        for pos, cluster_idx in reversed(to_expand):
            merged_1 = int(linkage_matrix[cluster_idx, 0])
            merged_2 = int(linkage_matrix[cluster_idx, 1])
            sorted_index[pos] = merged_1
            sorted_index.insert(pos + 1, merged_2)
    return sorted_index


def hrp_weights(covariance: pd.DataFrame, linkage_method: str = "single") -> pd.Series:
    std_dev = np.sqrt(np.diag(covariance.values))
    denom = np.outer(std_dev, std_dev)
    denom = np.where(denom == 0, np.nan, denom)
    correlation = covariance.values / denom
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    correlation_df = pd.DataFrame(correlation, index=covariance.index, columns=covariance.index)

    distance_matrix = corr_distance_of_distance_matrix(correlation_df)
    condensed_dist = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed_dist, method=linkage_method)

    n_items = len(covariance.index)
    sorted_index = quasi_diagonalize(linkage_matrix, n_items)
    sorted_items = [covariance.index[i] for i in sorted_index]
    cov_sorted = covariance.loc[sorted_items, sorted_items]

    weights = pd.Series(1.0, index=sorted_items)
    groups = [list(range(n_items))]

    while any(len(g) > 1 for g in groups):
        new_groups = []
        for group in groups:
            if len(group) <= 1:
                new_groups.append(group)
                continue

            mid = len(group) // 2
            group_1 = group[:mid]
            group_2 = group[mid:]

            items_1 = [sorted_items[i] for i in group_1]
            items_2 = [sorted_items[i] for i in group_2]

            cov_1 = cov_sorted.loc[items_1, items_1].values
            cov_2 = cov_sorted.loc[items_2, items_2].values

            var_1 = compute_cluster_variance(cov_1)
            var_2 = compute_cluster_variance(cov_2)

            alpha = var_2 / (var_1 + var_2) if (var_1 + var_2) > 0 else 0.5

            for i in group_1:
                weights.iloc[i] *= alpha
            for i in group_2:
                weights.iloc[i] *= (1 - alpha)

            new_groups.append(group_1)
            new_groups.append(group_2)
        groups = new_groups

    weights = weights / weights.sum()
    weights.name = "hrp"
    return weights.reindex(covariance.index).fillna(0.0)


def herc_weights(
    covariance: pd.DataFrame,
    max_depth: int | None = None,
    linkage_method: str = "ward",
    use_distance_of_distance: bool = True
) -> pd.Series:
    std_dev = np.sqrt(np.diag(covariance.values))
    denom = np.outer(std_dev, std_dev)
    denom = np.where(denom == 0, np.nan, denom)
    correlation = covariance.values / denom
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    correlation_df = pd.DataFrame(correlation, index=covariance.index, columns=covariance.index)

    distance_matrix = corr_distance_of_distance_matrix(correlation_df) if use_distance_of_distance else corr_distance_matrix(correlation_df)
    condensed_dist = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed_dist, method=linkage_method)

    n_items = len(covariance.index)
    sorted_index = quasi_diagonalize(linkage_matrix, n_items)
    sorted_items = [covariance.index[i] for i in sorted_index]
    cov_sorted = covariance.loc[sorted_items, sorted_items]

    weights = pd.Series(0.0, index=sorted_items)

    def split_group_recursive(group: List[int], cluster_weight: float, depth: int = 0):
        if len(group) == 1 or (max_depth is not None and depth >= max_depth):
            equal_w = cluster_weight / len(group)
            for i in group:
                weights.iloc[i] += equal_w
            return

        mid = len(group) // 2
        group_1 = group[:mid]
        group_2 = group[mid:]

        items_1 = [sorted_items[i] for i in group_1]
        items_2 = [sorted_items[i] for i in group_2]

        cov_1 = cov_sorted.loc[items_1, items_1].values
        cov_2 = cov_sorted.loc[items_2, items_2].values

        n1, n2 = len(group_1), len(group_2)
        w1_equal = np.ones(n1) / n1
        w2_equal = np.ones(n2) / n2

        var_1 = float(w1_equal.T @ cov_1 @ w1_equal)
        var_2 = float(w2_equal.T @ cov_2 @ w2_equal)

        total_var = var_1 + var_2
        if total_var > 0:
            w1 = var_2 / total_var
            w2 = var_1 / total_var
        else:
            w1 = w2 = 0.5

        split_group_recursive(group_1, cluster_weight * w1, depth + 1)
        split_group_recursive(group_2, cluster_weight * w2, depth + 1)

    split_group_recursive(list(range(n_items)), cluster_weight=1.0, depth=0)
    weights = weights / weights.sum()
    weights.name = "herc"
    return weights.reindex(covariance.index).fillna(0.0)


def compute_annualized_return_vol(stock_returns: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    mu = stock_returns.mean() * periods_per_year
    vol = stock_returns.std() * np.sqrt(periods_per_year)
    df = pd.DataFrame({"Return": mu, "Volatility": vol})
    df.index.name = "Ticker"
    return df


def remove_outliers_zscore(df: pd.DataFrame, cols: List[str], z: float = 3.5) -> Tuple[pd.DataFrame, List[str]]:
    x = df[cols].copy()
    zscores = (x - x.mean()) / (x.std(ddof=0).replace(0, np.nan))
    mask = (zscores.abs() <= z).all(axis=1)
    removed = df.index[~mask].tolist()
    return df.loc[mask].copy(), removed


def plot_kmeans_elbow(stats_df: pd.DataFrame, k_min: int, k_max: int, scale: bool = True) -> None:
    X = stats_df[["Return", "Volatility"]].values
    if scale:
        X = StandardScaler().fit_transform(X)

    ks = list(range(k_min, k_max + 1))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers", name="Inertia"))
    fig.update_layout(
        title="K-Means Elbow Curve (Return vs Volatility)",
        width=900,
        height=420,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(title="Number of clusters (k)", showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(title="Within-cluster SSE (Inertia)", showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    )
    fig.show()


def kmeans_cluster_retvol(stats_df: pd.DataFrame, k: int, scale: bool = True) -> Tuple[pd.Series, pd.DataFrame, KMeans]:
    X = stats_df[["Return", "Volatility"]].values
    if scale:
        X = StandardScaler().fit_transform(X)

    model = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=20)
    labels = model.fit_predict(X)

    out = stats_df.copy()
    out["Cluster"] = labels
    return pd.Series(labels, index=stats_df.index, name="kmeans_cluster"), out, model


def kmeans_cluster_equal_weights(labels: pd.Series, universe: List[str]) -> pd.Series:
    labels = labels.dropna()
    clusters = sorted(pd.unique(labels.values))
    k = len(clusters)

    w = pd.Series(0.0, index=universe, dtype=float)
    if k == 0:
        w.name = "kmeans_retvol"
        return w

    for c in clusters:
        members = labels.index[labels == c].tolist()
        if len(members) == 0:
            continue
        w.loc[members] = 1.0 / (k * len(members))

    if w.sum() > 0:
        w = w / w.sum()

    w.name = "kmeans_retvol"
    return w


def plot_kmeans_scatter(stats_with_clusters: pd.DataFrame, title: str = "K-Means Clusters: Return vs Volatility") -> None:
    dfp = stats_with_clusters.reset_index().rename(columns={"index": "Ticker"})
    fig = px.scatter(
        dfp,
        x="Return",
        y="Volatility",
        color="Cluster",
        hover_data=["Ticker"],
        title=title,
        width=950,
        height=520,
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)", title="Annualized Return"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)", title="Annualized Volatility"),
        legend_title_text="Cluster",
    )
    fig.show()


def compute_performance_metrics(returns: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    if len(returns) == 0 or returns.std() == 0:
        return {"mean_return": 0.0, "volatility": 0.0, "sharpe_ratio": 0.0}
    mean_return = returns.mean() * periods_per_year
    volatility = returns.std() * np.sqrt(periods_per_year)
    sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0
    return {"mean_return": mean_return, "volatility": volatility, "sharpe_ratio": sharpe_ratio}


def max_drawdown(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    cum = (1 + series).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def tracking_error(port_ret: pd.Series, bench_ret: pd.Series, periods_per_year: int = 252) -> float:
    common_idx = port_ret.index.intersection(bench_ret.index)
    if len(common_idx) == 0:
        return 0.0
    diff = port_ret.loc[common_idx] - bench_ret.loc[common_idx]
    return float(diff.std() * np.sqrt(periods_per_year))


def plot_linkage_dendrogram(Z: np.ndarray, labels: List[str], title: str, figsize: Tuple[int, int] = (18, 6)) -> None:
    plt.figure(figsize=figsize)
    dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=6
    )
    plt.title(title)
    plt.xlabel("Assets")
    plt.ylabel("Linkage distance")
    plt.tight_layout()
    plt.show()


def linkage_from_correlation(correlation: pd.DataFrame, linkage_method: str, use_distance_of_distance: bool) -> np.ndarray:
    corr = correlation.copy().fillna(0.0).clip(-1.0, 1.0)
    np.fill_diagonal(corr.values, 1.0)
    D = corr_distance_of_distance_matrix(corr) if use_distance_of_distance else corr_distance_matrix(corr)
    condensed = squareform(D, checks=False)
    return linkage(condensed, method=linkage_method)


if __name__ == "__main__":
    print("Starting Portfolio Analysis...")

    metadata, components = load_ftse100_data("ftse_stock_prices.csv")
    if metadata.empty:
        print("Stopping execution due to missing data.")
        raise SystemExit(1)

    print(f"Loaded data: {len(components)} stocks, {len(metadata)} trading days")

    in_sample_raw = metadata[IN_SAMPLE_START:IN_SAMPLE_END].copy()
    in_sample_returns, in_sample_correlation, in_sample_prices = preprocess_returns(
        in_sample_raw, components, min_data_availability=MIN_DATA_AVAILABILITY
    )

    # Plotly correlation heatmap (in-sample)
    fig_corr_is = plotly_correlation_heatmap(
        in_sample_correlation,
        title=f"FTSE100 Constituents Correlation Heatmap ({IN_SAMPLE_START}-{IN_SAMPLE_END})",
        show_values=False
    )
    fig_corr_is.show()

    in_sample_data = in_sample_prices.copy()
    in_sample_data["FTSE100"] = metadata.loc[in_sample_data.index, "FTSE100"].ffill()
    components = list(in_sample_returns.columns)
    print(f"Using {len(components)} stocks for in-sample analysis.")

    corr_vals = in_sample_correlation.values
    upper = corr_vals[np.triu_indices_from(corr_vals, k=1)]
    upper = upper[~np.isnan(upper)]
    threshold = np.quantile(upper, NETWORK_THRESHOLD_QUANTILE)
    print(f"Network threshold (q={NETWORK_THRESHOLD_QUANTILE}): {threshold:.3f}")

    in_sample_graph, in_sample_layout = create_graph(components, in_sample_correlation, threshold)
    in_sample_full_graph = create_full_correlation_graph(components, in_sample_correlation)

    minvar_w, maxsharpe_w = compute_markowitz_weights(in_sample_returns, allow_short=False)
    validate_portfolio_weights(minvar_w, "markowitz_minvar")
    validate_portfolio_weights(maxsharpe_w, "markowitz_maxsharpe")

    in_sample_data["markowitz_minvar"] = compute_weighted_portfolio(in_sample_data, minvar_w, "markowitz_minvar")
    in_sample_data["markowitz_maxsharpe"] = compute_weighted_portfolio(in_sample_data, maxsharpe_w, "markowitz_maxsharpe")

    isolated, independence = degeneracy_ordering(in_sample_graph, components)
    selected_stocks_deg = isolated + independence
    in_sample_data["degeneracy"] = compute_equal_weighted_portfolio(in_sample_data, selected_stocks_deg, "degeneracy")

    fig_deg = plotly_network(
        in_sample_graph,
        in_sample_layout,
        title="Degeneracy Network (yellow = selected stocks)",
        selected=selected_stocks_deg,
        node_score=None,
        show_labels=False,
        color_mode="selected",
        selected_color="#F9D423",
        default_color="#D3D3D3",
    )
    fig_deg.show()

    eigen_w_central = eigenvector_centrality_weights(in_sample_full_graph, components)
    validate_portfolio_weights(eigen_w_central, "eigen_central")
    in_sample_data["eigen_central"] = compute_weighted_portfolio(in_sample_data, eigen_w_central, "eigen_central")

    try:
        in_ec = nx.eigenvector_centrality(in_sample_graph, weight="weight", max_iter=2000)
    except Exception:
        in_ec = nx.eigenvector_centrality_numpy(in_sample_graph, weight="weight")

    fig_eigen = plotly_network(
        in_sample_graph,
        in_sample_layout,
        title="Eigenvector Centrality Network (color = centrality score)",
        selected=None,
        node_score=in_ec,
        show_labels=False,
        color_mode="score",
    )
    fig_eigen.show()

    cluster_labels, Z_cluster = hierarchical_clusters_from_corr(in_sample_correlation, method="average", max_clusters=HIER_MAX_CLUSTERS)
    cluster_w = cluster_equal_weights(cluster_labels)
    validate_portfolio_weights(cluster_w, "cluster_equal")
    in_sample_data["cluster_equal"] = compute_weighted_portfolio(in_sample_data, cluster_w, "cluster_equal")

    h1n_weights = hierarchical_one_over_n_weights(in_sample_correlation, linkage_method="ward", use_distance_of_distance=True)
    validate_portfolio_weights(h1n_weights, "hier_1N")
    in_sample_data["hier_1N"] = compute_weighted_portfolio(in_sample_data, h1n_weights, "hier_1N")

    Z_h1n = linkage_from_correlation(in_sample_correlation, linkage_method="ward", use_distance_of_distance=True)
    plot_linkage_dendrogram(Z_h1n, list(in_sample_correlation.index), "Dendrogram: Hierarchical 1/N (ward, distance-of-distance)")

    in_sample_cov = in_sample_returns.cov()

    std_dev = np.sqrt(np.diag(in_sample_cov.values))
    denom = np.outer(std_dev, std_dev)
    denom = np.where(denom == 0, np.nan, denom)
    corr_from_cov = in_sample_cov.values / denom
    corr_from_cov = np.nan_to_num(corr_from_cov, nan=0.0, posinf=0.0, neginf=0.0)
    corr_from_cov = np.clip(corr_from_cov, -1.0, 1.0)
    np.fill_diagonal(corr_from_cov, 1.0)
    corr_cov_df = pd.DataFrame(corr_from_cov, index=in_sample_cov.index, columns=in_sample_cov.index)

    hrp_weights_series = hrp_weights(in_sample_cov, linkage_method="single")
    validate_portfolio_weights(hrp_weights_series, "hrp")
    in_sample_data["hrp"] = compute_weighted_portfolio(in_sample_data, hrp_weights_series, "hrp")

    Z_hrp = linkage_from_correlation(corr_cov_df, linkage_method="single", use_distance_of_distance=True)
    plot_linkage_dendrogram(Z_hrp, list(corr_cov_df.index), "Dendrogram: HRP (single, distance-of-distance)")

    herc_w = herc_weights(in_sample_cov, max_depth=HERC_MAX_DEPTH, linkage_method="ward", use_distance_of_distance=True)
    validate_portfolio_weights(herc_w, "herc")
    in_sample_data["herc"] = compute_weighted_portfolio(in_sample_data, herc_w, "herc")

    Z_herc = linkage_from_correlation(corr_cov_df, linkage_method="ward", use_distance_of_distance=True)
    plot_linkage_dendrogram(Z_herc, list(corr_cov_df.index), "Dendrogram: HERC (ward, distance-of-distance)")

    stats = compute_annualized_return_vol(in_sample_returns, periods_per_year=PERIODS_PER_YEAR)

    if KMEANS_REMOVE_OUTLIERS:
        stats_clean, removed = remove_outliers_zscore(stats, ["Return", "Volatility"], z=KMEANS_OUTLIER_Z)
        if removed:
            print(f"KMeans: removed outliers (|z|>{KMEANS_OUTLIER_Z}): {removed}")
        stats_for_kmeans = stats_clean
    else:
        stats_for_kmeans = stats

    plot_kmeans_elbow(stats_for_kmeans, KMEANS_ELBOW_MIN_K, KMEANS_ELBOW_MAX_K, scale=KMEANS_SCALE_FEATURES)

    kmeans_labels, stats_with_clusters, kmeans_model = kmeans_cluster_retvol(
        stats_for_kmeans, k=KMEANS_K, scale=KMEANS_SCALE_FEATURES
    )

    plot_kmeans_scatter(stats_with_clusters, title=f"K-Means Clusters (k={KMEANS_K}): Annualized Return vs Volatility")

    kmeans_w = kmeans_cluster_equal_weights(kmeans_labels, components)
    validate_portfolio_weights(kmeans_w, "kmeans_retvol")
    in_sample_data["kmeans_retvol"] = compute_weighted_portfolio(in_sample_data, kmeans_w, "kmeans_retvol")

    portfolios = [
        "FTSE100",
        "markowitz_minvar",
        "markowitz_maxsharpe",
        "degeneracy",
        "eigen_central",
        "cluster_equal",
        "kmeans_retvol",
        "hier_1N",
        "hrp",
        "herc",
    ]

    style_map = {
        "FTSE100": ("grey", "-", "FTSE 100 Index"),
        "markowitz_minvar": ("blue", "-", "Markowitz Min-Var"),
        "markowitz_maxsharpe": ("purple", "--", "Markowitz Max-Sharpe"),
        "degeneracy": ("green", "-", "Degeneracy Portfolio"),
        "eigen_central": ("orange", "-", "Eigen Centrality Portfolio"),
        "cluster_equal": ("magenta", ":", "Cluster Equal-Weight"),
        "kmeans_retvol": ("black", "-", "K-Means (Ret/Vol) Cluster Portfolio"),
        "hier_1N": ("teal", "--", "Hierarchical 1/N"),
        "hrp": ("cyan", "-.", "HRP"),
        "herc": ("brown", ":", "HERC"),
    }

    label_map = {p: style_map.get(p, ("", "", p))[2] for p in portfolios}
    color_map = {k: v[0] for k, v in style_map.items()}

    print("\n" + "=" * 100)
    print(f"{'In-Sample Performance Metrics (2015-2017)':^100}")
    print("=" * 100)
    print(f"{'Portfolio':<25} {'Mean Ret':<12} {'Vol':<12} {'Sharpe':<12} {'Max DD':<12} {'Track Err':<12}")
    print("-" * 100)

    ftse_is_ret = in_sample_data["FTSE100"].pct_change().dropna()
    for portfolio in portfolios:
        if portfolio in in_sample_data.columns:
            r = in_sample_data[portfolio].pct_change().dropna()
            metrics = compute_performance_metrics(r)
            dd = max_drawdown(r)
            te = tracking_error(r, ftse_is_ret) if portfolio != "FTSE100" else 0.0
            print(
                f"{portfolio:<25} "
                f"{metrics['mean_return']*100:>10.2f}% "
                f"{metrics['volatility']*100:>10.2f}% "
                f"{metrics['sharpe_ratio']:>10.4f} "
                f"{dd*100:>10.2f}% "
                f"{te*100:>10.2f}%"
            )

    plot_cumulative_returns(in_sample_data, portfolios, style_map, "Cumulative Returns (2015-2017)")
    plot_mean_variance_scatter(in_sample_data, portfolios, label_map, color_map, "Mean-Variance Analysis (2015-2017)")

    out_sample_raw = metadata[OUT_SAMPLE_START:OUT_SAMPLE_END].copy()
    out_sample_returns, out_sample_correlation, out_sample_prices = preprocess_returns(
        out_sample_raw, components, min_data_availability=MIN_DATA_AVAILABILITY
    )

    # Plotly correlation heatmap (out-of-sample)
    fig_corr_os = plotly_correlation_heatmap(
        out_sample_correlation,
        title=f"FTSE100 Constituents Correlation Heatmap ({OUT_SAMPLE_START}-{OUT_SAMPLE_END})",
        show_values=False
    )
    fig_corr_os.show()

    out_sample_data = out_sample_prices.copy()
    out_sample_data["FTSE100"] = metadata.loc[out_sample_data.index, "FTSE100"].ffill()

    out_sample_data["markowitz_minvar"] = compute_weighted_portfolio(out_sample_data, minvar_w, "markowitz_minvar")
    out_sample_data["markowitz_maxsharpe"] = compute_weighted_portfolio(out_sample_data, maxsharpe_w, "markowitz_maxsharpe")
    out_sample_data["degeneracy"] = compute_equal_weighted_portfolio(out_sample_data, selected_stocks_deg, "degeneracy")
    out_sample_data["eigen_central"] = compute_weighted_portfolio(out_sample_data, eigen_w_central, "eigen_central")
    out_sample_data["cluster_equal"] = compute_weighted_portfolio(out_sample_data, cluster_w, "cluster_equal")
    out_sample_data["kmeans_retvol"] = compute_weighted_portfolio(out_sample_data, kmeans_w, "kmeans_retvol")
    out_sample_data["hier_1N"] = compute_weighted_portfolio(out_sample_data, h1n_weights, "hier_1N")
    out_sample_data["hrp"] = compute_weighted_portfolio(out_sample_data, hrp_weights_series, "hrp")
    out_sample_data["herc"] = compute_weighted_portfolio(out_sample_data, herc_w, "herc")

    print("\n" + "=" * 100)
    print(f"{'Out-of-Sample Performance Metrics (2018-2025)':^100}")
    print("=" * 100)
    print(f"{'Portfolio':<25} {'Mean Ret':<12} {'Vol':<12} {'Sharpe':<12} {'Max DD':<12} {'Track Err':<12}")
    print("-" * 100)

    ftse_os_ret = out_sample_data["FTSE100"].pct_change().dropna()
    for portfolio in portfolios:
        if portfolio in out_sample_data.columns:
            r = out_sample_data[portfolio].pct_change().dropna()
            metrics = compute_performance_metrics(r)
            dd = max_drawdown(r)
            te = tracking_error(r, ftse_os_ret) if portfolio != "FTSE100" else 0.0
            print(
                f"{portfolio:<25} "
                f"{metrics['mean_return']*100:>10.2f}% "
                f"{metrics['volatility']*100:>10.2f}% "
                f"{metrics['sharpe_ratio']:>10.4f} "
                f"{dd*100:>10.2f}% "
                f"{te*100:>10.2f}%"
            )

    plot_cumulative_returns(out_sample_data, portfolios, style_map, "Cumulative Returns (2018-2025)")
    plot_mean_variance_scatter(out_sample_data, portfolios, label_map, color_map, "Mean-Variance Analysis (2018-2025)")
