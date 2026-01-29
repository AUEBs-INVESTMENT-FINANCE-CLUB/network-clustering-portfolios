import shutil
import warnings

import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from config import (
    IN_SAMPLE_START, IN_SAMPLE_END, OUT_SAMPLE_START, OUT_SAMPLE_END,
    MIN_DATA_AVAILABILITY, NETWORK_THRESHOLD_QUANTILE,
    HIER_MAX_CLUSTERS, HERC_MAX_DEPTH, PERIODS_PER_YEAR,
    KMEANS_K, KMEANS_SCALE_FEATURES, KMEANS_REMOVE_OUTLIERS, KMEANS_OUTLIER_Z,
    OUTPUT_DIR
)

from data_io import load_ftse100_data, preprocess_returns
from portfolio_utils import compute_weighted_portfolio, compute_equal_weighted_portfolio, validate_portfolio_weights
from networks import (
    create_graph, create_full_correlation_graph, degeneracy_ordering,
    eigenvector_centrality_weights, plotly_network
)
from clustering import (
    hierarchical_clusters_from_corr, cluster_equal_weights, save_dendrogram,
    herc_weights, corr_distance_of_distance_matrix,
    compute_annualized_return_vol, remove_outliers_zscore,
    kmeans_cluster_retvol, build_kmeans_weights, plot_kmeans_scatter
)
from plots import (
    plotly_correlation_heatmap, save_plotly, save_plotly_png,
    plot_cumulative_returns, plot_mean_variance_scatter
)
from metrics import metrics_table_from_values


def restrict_weights_to_universe(weights: pd.Series, universe: list, name: str) -> pd.Series:
    w = weights.reindex(universe).fillna(0.0).astype(float)
    s = float(w.sum())
    if s != 0.0:
        w = w / s
    w.name = name
    return w


def save_weights_csv(w: pd.Series, name: str) -> None:
    w = w.reindex(w.index).fillna(0.0).astype(float)
    w = w / w.sum() if float(w.sum()) != 0.0 else w
    w = w.sort_values(ascending=False)
    df = pd.DataFrame({"Weight": w, "Weight_%": 100.0 * w})
    df.to_csv(OUTPUT_DIR / f"weights_{name}.csv")


def build_appendix_holdings_table(
    weights_map: dict[str, pd.Series],
    top_n_eigen_herc: int = 15
) -> pd.DataFrame:
    picked = set()

    if "degeneracy" in weights_map:
        picked |= set(weights_map["degeneracy"][weights_map["degeneracy"] > 0].index)

    if "kmeans" in weights_map:
        picked |= set(weights_map["kmeans"][weights_map["kmeans"] > 0].index)

    if "eigen_central" in weights_map:
        picked |= set(weights_map["eigen_central"].sort_values(ascending=False).head(top_n_eigen_herc).index)

    if "herc" in weights_map:
        picked |= set(weights_map["herc"].sort_values(ascending=False).head(top_n_eigen_herc).index)

    picked = sorted(picked)

    pretty_cols = {
        "degeneracy": "Degeneracy (%)",
        "eigen_central": "Eigen tilt (%)",
        "cluster_equal": "Cluster eq (%)",
        "kmeans": "K-means (%)",
        "herc": "HERC (%)",
    }

    df = pd.DataFrame(index=picked)
    df.index.name = "Ticker"

    for key, w in weights_map.items():
        if key not in pretty_cols:
            continue
        df[pretty_cols[key]] = 100.0 * w.reindex(picked).fillna(0.0)

    df = df.loc[(df.abs().sum(axis=1) > 0)].copy()
    return df


def save_appendix_table_files(df: pd.DataFrame, base_name: str = "appendix_weights_table") -> None:
    df.to_csv(OUTPUT_DIR / f"{base_name}.csv")
    latex = df.to_latex(
        escape=True,
        float_format="%.3f",
        column_format="l" + "r" * df.shape[1]
    )
    with open(OUTPUT_DIR / f"{base_name}.tex", "w", encoding="utf-8") as f:
        f.write(latex)


if __name__ == "__main__":
    metadata, all_components = load_ftse100_data("ftse_stock_prices.csv")
    if metadata.empty:
        raise SystemExit(1)

    metadata = metadata.sort_index()

    out_sample_raw_all = metadata.loc[OUT_SAMPLE_START:OUT_SAMPLE_END].copy()
    if len(out_sample_raw_all) == 0:
        raise ValueError("Out-of-sample window is empty. Check OUT_SAMPLE_START/OUT_SAMPLE_END.")

    out_prices_all = out_sample_raw_all.reindex(columns=all_components).sort_index().ffill()
    nonempty_idx = out_prices_all.dropna(how="all").index
    if len(nonempty_idx) == 0:
        raise ValueError("Out-of-sample prices are all NaN even after ffill. Check your data coverage.")
    test_start_date = nonempty_idx.min()

    universe_start = out_prices_all.loc[test_start_date].dropna().index.tolist()
    if len(universe_start) == 0:
        raise ValueError("Frozen universe is empty at out-of-sample start (after ffill).")

    in_sample_raw = metadata.loc[IN_SAMPLE_START:IN_SAMPLE_END].copy()
    in_sample_returns, in_sample_correlation, in_sample_prices = preprocess_returns(
        in_sample_raw, universe_start, min_data_availability=MIN_DATA_AVAILABILITY
    )

    components = list(in_sample_prices.columns)
    if len(components) == 0:
        raise ValueError("No components remain after in-sample preprocessing.")

    out_prices_all = out_prices_all.reindex(columns=components)
    if out_prices_all.loc[test_start_date].isna().any():
        missing = out_prices_all.columns[out_prices_all.loc[test_start_date].isna()].tolist()
        raise ValueError(f"Some in-sample components have no price at test start: {missing}")

    fig_corr_is = plotly_correlation_heatmap(
        in_sample_correlation,
        title=f"FTSE100 Constituents Correlation Heatmap ({IN_SAMPLE_START}-{IN_SAMPLE_END})",
        show_values=False
    )
    save_plotly(fig_corr_is, f"corr_heatmap_{IN_SAMPLE_START}_{IN_SAMPLE_END}.html")
    save_plotly_png(fig_corr_is, "correlation.png")

    in_sample_data = in_sample_prices[components].copy()
    in_sample_data["FTSE100"] = metadata.loc[in_sample_data.index, "FTSE100"].ffill()

    corr_vals = in_sample_correlation.values
    upper = corr_vals[np.triu_indices_from(corr_vals, k=1)]
    upper = upper[~np.isnan(upper)]
    threshold = np.quantile(upper, NETWORK_THRESHOLD_QUANTILE)

    in_sample_graph, in_sample_layout = create_graph(components, in_sample_correlation, threshold)
    _ = create_full_correlation_graph(components, in_sample_correlation)

    isolated, independence = degeneracy_ordering(in_sample_graph, components)
    selected_stocks_deg = [s for s in (isolated + independence) if s in components]
    deg_w = pd.Series(0.0, index=components, dtype=float)
    if len(selected_stocks_deg) > 0:
        deg_w.loc[selected_stocks_deg] = 1.0 / len(selected_stocks_deg)
    deg_w = restrict_weights_to_universe(deg_w, components, "degeneracy")
    validate_portfolio_weights(deg_w, "degeneracy")
    in_sample_data["degeneracy"] = compute_weighted_portfolio(in_sample_data, deg_w, "degeneracy")

    fig_deg = plotly_network(
        in_sample_graph,
        in_sample_layout,
        title="Degeneracy Network (yellow = selected stocks)",
        selected=selected_stocks_deg,
        show_labels=False,
        color_mode="selected",
        selected_color="#F9D423",
        default_color="#D3D3D3",
    )
    save_plotly(fig_deg, f"network_degeneracy_{IN_SAMPLE_START}_{IN_SAMPLE_END}.html")
    save_plotly_png(fig_deg, "degeneracy.png")

    eigen_w_central = eigenvector_centrality_weights(in_sample_graph, components, inverse=True)
    eigen_w_central = restrict_weights_to_universe(eigen_w_central, components, "eigen_central")
    validate_portfolio_weights(eigen_w_central, "eigen_central")
    in_sample_data["eigen_central"] = compute_weighted_portfolio(in_sample_data, eigen_w_central, "eigen_central")

    try:
        ec_plot_dict = nx.eigenvector_centrality(in_sample_graph, weight="weight", max_iter=2000)
    except Exception:
        ec_plot_dict = nx.eigenvector_centrality_numpy(in_sample_graph, weight="weight")

    ec_plot = pd.Series(ec_plot_dict).reindex(list(in_sample_graph.nodes())).fillna(0.0).astype(float)
    ec_vals = ec_plot.values

    fig_eigen = plotly_network(
        in_sample_graph,
        in_sample_layout,
        title="Eigenvector Centrality Network",
        show_labels=False,
        color_mode="continuous",
        node_values=ec_vals,
        colorbar_title="Eigenvector Centrality",
    )
    save_plotly(fig_eigen, f"network_eigen_centrality_{IN_SAMPLE_START}_{IN_SAMPLE_END}.html")
    save_plotly_png(fig_eigen, "eigencentrality.png")

    cluster_labels, Z_cluster = hierarchical_clusters_from_corr(
        in_sample_correlation, method="average", max_clusters=HIER_MAX_CLUSTERS
    )
    cluster_w = cluster_equal_weights(cluster_labels)
    cluster_w = restrict_weights_to_universe(cluster_w, components, "cluster_equal")
    validate_portfolio_weights(cluster_w, "cluster_equal")
    in_sample_data["cluster_equal"] = compute_weighted_portfolio(in_sample_data, cluster_w, "cluster_equal")

    save_dendrogram(
        Z_cluster,
        list(in_sample_correlation.index),
        "Dendrogram: Cluster Equal-Weight (average, corr-distance)",
        f"dendrogram_cluster_equal_{IN_SAMPLE_START}_{IN_SAMPLE_END}.png"
    )
    try:
        shutil.copy(
            OUTPUT_DIR / f"dendrogram_cluster_equal_{IN_SAMPLE_START}_{IN_SAMPLE_END}.png",
            OUTPUT_DIR / "dendrogram.png",
        )
    except Exception as e:
        warnings.warn(
            f"Article copy dendrogram.png skipped: {e}",
            UserWarning,
            stacklevel=2,
        )

    in_sample_cov = in_sample_returns.cov()
    herc_w = herc_weights(
        in_sample_cov,
        max_depth=HERC_MAX_DEPTH,
        linkage_method="ward",
        use_distance_of_distance=True
    )
    herc_w = restrict_weights_to_universe(herc_w, components, "herc")
    validate_portfolio_weights(herc_w, "herc")
    in_sample_data["herc"] = compute_weighted_portfolio(in_sample_data, herc_w, "herc")

    std_dev = np.sqrt(np.diag(in_sample_cov.values))
    denom = np.outer(std_dev, std_dev)
    denom = np.where(denom == 0, np.nan, denom)
    corr_from_cov = in_sample_cov.values / denom
    corr_from_cov = np.nan_to_num(corr_from_cov, nan=0.0, posinf=0.0, neginf=0.0)
    corr_from_cov = np.clip(corr_from_cov, -1.0, 1.0)
    np.fill_diagonal(corr_from_cov, 1.0)
    corr_cov_df = pd.DataFrame(corr_from_cov, index=in_sample_cov.index, columns=in_sample_cov.index)

    D_herc = corr_distance_of_distance_matrix(corr_cov_df)
    Z_herc = linkage(squareform(D_herc, checks=False), method="ward")

    save_dendrogram(
        Z_herc,
        list(corr_cov_df.index),
        "Dendrogram: HERC (ward, distance-of-distance)",
        f"dendrogram_herc_{IN_SAMPLE_START}_{IN_SAMPLE_END}.png"
    )
    try:
        shutil.copy(
            OUTPUT_DIR / f"dendrogram_herc_{IN_SAMPLE_START}_{IN_SAMPLE_END}.png",
            OUTPUT_DIR / "herc.png",
        )
    except Exception as e:
        warnings.warn(
            f"Article copy herc.png skipped: {e}",
            UserWarning,
            stacklevel=2,
        )

    stats = compute_annualized_return_vol(in_sample_returns, periods_per_year=PERIODS_PER_YEAR)

    if KMEANS_REMOVE_OUTLIERS:
        stats_for_kmeans, _ = remove_outliers_zscore(stats, ["Return", "Volatility"], z=KMEANS_OUTLIER_Z)
    else:
        stats_for_kmeans = stats

    kmeans_labels, stats_with_clusters, _, _ = kmeans_cluster_retvol(
        stats_for_kmeans, k=KMEANS_K, scale=KMEANS_SCALE_FEATURES
    )

    kmeans_w, picked_clusters_df = build_kmeans_weights(
        stats_with_clusters,
        kmeans_labels,
        components,
        top_clusters=2,
        min_cluster_size=4
    )
    kmeans_w = restrict_weights_to_universe(kmeans_w, components, "kmeans")
    validate_portfolio_weights(kmeans_w, "kmeans")
    in_sample_data["kmeans"] = compute_weighted_portfolio(in_sample_data, kmeans_w, "kmeans")

    fig_km_scatter = plot_kmeans_scatter(
        stats_with_clusters,
        picked_clusters=list(picked_clusters_df.index),
        title=f"K-Means (k={KMEANS_K}) Clusters: Annualized Return vs Volatility"
    )
    save_plotly(fig_km_scatter, f"kmeans_scatter_{IN_SAMPLE_START}_{IN_SAMPLE_END}.html")
    save_plotly_png(fig_km_scatter, "kmeans.png")

    weights_map = {
        "degeneracy": deg_w,
        "eigen_central": eigen_w_central,
        "cluster_equal": cluster_w,
        "kmeans": kmeans_w,
        "herc": herc_w,
    }

    portfolios = [
        "FTSE100",
        "degeneracy",
        "eigen_central",
        "cluster_equal",
        "kmeans",
        "herc",
    ]

    style_map = {
        "FTSE100": ("grey", "-", "FTSE 100 Index"),
        "degeneracy": ("green", "-", "Degeneracy Selection"),
        "eigen_central": ("orange", "-", "Eigen Centrality (Inverse)"),
        "cluster_equal": ("magenta", ":", "Cluster Equal Weight"),
        "kmeans": ("black", "-", "K-Means"),
        "herc": ("brown", ":", "HERC"),
    }

    label_map = {p: style_map[p][2] for p in portfolios}
    color_map = {p: style_map[p][0] for p in portfolios}

    metrics_is = metrics_table_from_values(
        in_sample_data,
        portfolios,
        bench_name="FTSE100",
        periods_per_year=PERIODS_PER_YEAR,
        asset_returns=in_sample_returns[components],
        weights_map=weights_map,
    )
    metrics_is.to_csv(OUTPUT_DIR / f"metrics_in_sample_{IN_SAMPLE_START}_{IN_SAMPLE_END}.csv")

    fig_cum_is = plot_cumulative_returns(in_sample_data, portfolios, style_map)
    save_plotly(fig_cum_is, f"cumulative_in_sample_{IN_SAMPLE_START}_{IN_SAMPLE_END}.html")

    fig_mv_is = plot_mean_variance_scatter(
        in_sample_data,
        portfolios,
        label_map,
        color_map,
        f"Mean-Variance Analysis ({IN_SAMPLE_START}-{IN_SAMPLE_END})"
    )
    save_plotly(fig_mv_is, f"mean_variance_in_sample_{IN_SAMPLE_START}_{IN_SAMPLE_END}.html")

    print("\n--- IN-SAMPLE METRICS (Training Phase) ---")
    print(metrics_is)

    print(f"\n--- RUNNING OUT-OF-SAMPLE ANALYSIS ({OUT_SAMPLE_START}-{OUT_SAMPLE_END}) ---")

    out_sample_prices = out_prices_all.loc[test_start_date:, components].copy()
    out_sample_data = out_sample_prices.copy()
    out_sample_data["FTSE100"] = metadata.loc[out_sample_data.index, "FTSE100"].ffill()

    out_sample_data["degeneracy"] = compute_weighted_portfolio(out_sample_data, deg_w, "degeneracy")
    out_sample_data["eigen_central"] = compute_weighted_portfolio(out_sample_data, eigen_w_central, "eigen_central")
    out_sample_data["cluster_equal"] = compute_weighted_portfolio(out_sample_data, cluster_w, "cluster_equal")
    out_sample_data["herc"] = compute_weighted_portfolio(out_sample_data, herc_w, "herc")
    out_sample_data["kmeans"] = compute_weighted_portfolio(out_sample_data, kmeans_w, "kmeans")

    out_sample_asset_returns = np.log(out_sample_prices).diff().dropna()

    metrics_os = metrics_table_from_values(
        out_sample_data,
        portfolios,
        bench_name="FTSE100",
        periods_per_year=PERIODS_PER_YEAR,
        asset_returns=out_sample_asset_returns,
        weights_map=weights_map,
    )
    metrics_os.to_csv(OUTPUT_DIR / f"metrics_out_sample_{OUT_SAMPLE_START}_{OUT_SAMPLE_END}.csv")

    print("\n--- OUT-OF-SAMPLE METRICS (Test Phase) ---")
    print(metrics_os)

    fig_cum_os = plot_cumulative_returns(out_sample_data, portfolios, style_map)
    save_plotly(fig_cum_os, f"cumulative_out_sample_{OUT_SAMPLE_START}_{OUT_SAMPLE_END}.html")
    save_plotly_png(fig_cum_os, "returns_oos.png")

    fig_mv_os = plot_mean_variance_scatter(
        out_sample_data,
        portfolios,
        label_map,
        color_map,
        f"Mean-Variance Analysis ({OUT_SAMPLE_START}-{OUT_SAMPLE_END})"
    )
    save_plotly(fig_mv_os, f"mean_variance_out_sample_{OUT_SAMPLE_START}_{OUT_SAMPLE_END}.html")

    for name, w in weights_map.items():
        save_weights_csv(w, name)

    appendix_df = build_appendix_holdings_table(weights_map, top_n_eigen_herc=15)
    save_appendix_table_files(appendix_df, base_name="appendix_weights_table")

    print(f"\nCompleted successfully. All outputs saved to: {OUTPUT_DIR.resolve()}")
