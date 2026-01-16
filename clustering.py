from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

from config import OUTPUT_DIR, KMEANS_RANDOM_STATE


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


def hierarchical_clusters_from_corr(
    correlation: pd.DataFrame, method: str = "average", max_clusters: int = 10
) -> Tuple[pd.Series, np.ndarray]:
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


def save_dendrogram(
    Z: np.ndarray, labels: List[str], title: str, filename: str, figsize: Tuple[int, int] = (18, 6)
) -> None:
    plt.figure(figsize=figsize)
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=6)
    plt.title(title)
    plt.xlabel("Assets")
    plt.ylabel("Linkage distance")
    plt.tight_layout()
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=200)
    plt.close()


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

        var_1 = compute_cluster_variance(cov_1)
        var_2 = compute_cluster_variance(cov_2)


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


def kmeans_cluster_retvol(stats_df: pd.DataFrame, k: int, scale: bool = True) -> Tuple[pd.Series, pd.DataFrame, KMeans, np.ndarray]:
    X_raw = stats_df[["Return", "Volatility"]].values
    X = X_raw.copy()
    if scale:
        X = StandardScaler().fit_transform(X)

    model = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=20)
    labels = model.fit_predict(X)

    out = stats_df.copy()
    out["Cluster"] = labels
    return pd.Series(labels, index=stats_df.index, name="kmeans_cluster"), out, model, X_raw


def build_kmeans_weights(
    stats_with_clusters: pd.DataFrame,
    labels: pd.Series,
    universe: List[str],
    top_clusters: int = 2,
    min_cluster_size: int = 4
) -> Tuple[pd.Series, pd.DataFrame]:
    df = stats_with_clusters.copy()
    g = df.groupby("Cluster")[["Return", "Volatility"]].mean()
    g["Score"] = g["Return"] / g["Volatility"].replace(0, np.nan)
    g = g.replace([np.inf, -np.inf], np.nan).dropna()
    sizes = df.groupby("Cluster").size()
    g["Size"] = g.index.map(lambda c: int(sizes.loc[c]) if c in sizes.index else 0)
    g = g[g["Size"] >= min_cluster_size].copy()

    w = pd.Series(0.0, index=universe, dtype=float)
    if len(g) == 0:
        w.name = "kmeans"
        return w, g

    kpick = int(min(top_clusters, len(g)))
    picked = g.sort_values("Score", ascending=False).head(kpick).copy()

    scores = picked["Score"].clip(lower=0.0)
    if scores.sum() <= 0:
        scores = pd.Series(1.0, index=picked.index)

    cluster_alloc = scores / scores.sum()

    for c in picked.index:
        members = labels.index[labels == c].tolist()
        members = [m for m in members if m in w.index]
        if len(members) == 0:
            continue
        w.loc[members] += float(cluster_alloc.loc[c]) * (1.0 / len(members))

    if w.sum() > 0:
        w = w / w.sum()
    w.name = "kmeans"
    return w, picked


def plot_kmeans_scatter(stats_with_clusters: pd.DataFrame, picked_clusters: List[int], title: str):

    dfp = stats_with_clusters.reset_index().rename(columns={"index": "Ticker"})
    dfp["Cluster"] = dfp["Cluster"].astype(int)

    fig = px.scatter(
        dfp,
        x="Return",
        y="Volatility",
        color="Cluster",
        hover_data=["Ticker"],
        title=title,
        width=980,
        height=540,
    )
    if picked_clusters is not None and len(picked_clusters) > 0:
        picked_set = set(int(c) for c in picked_clusters)
        df_pick = dfp[dfp["Cluster"].isin(picked_set)].copy()

        if len(df_pick) > 0:
            fig.add_scatter(
                x=df_pick["Return"],
                y=df_pick["Volatility"],
                mode="markers",
                marker=dict(
                    symbol="circle-open",
                    size=14,
                    line=dict(width=2, color="rgba(0,0,0,0.85)"),
                ),
                hovertext=df_pick["Ticker"],
                hovertemplate="<b>%{hovertext}</b><extra></extra>",
                showlegend=False,
                name="Picked",
            )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)", title="Annualized Return (log)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)", title="Annualized Volatility (log)"),
        legend_title_text="Cluster",
    )
    return fig