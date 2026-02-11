import shutil
import warnings

import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from config import (
    IN_SAMPLE_START, IN_SAMPLE_END, OUT_SAMPLE_START, OUT_SAMPLE_END,
    MIN_DATA_AVAILABILITY, MIN_PRICE_THRESHOLD, NETWORK_THRESHOLD_QUANTILE,
    HIER_MAX_CLUSTERS, HIER_MIN_CLUSTER_SIZE, HERC_MAX_DEPTH,
    HERC_LINKAGE_METHOD, HERC_USE_DISTANCE_OF_DISTANCE,
    KMEANS_K, KMEANS_SCALE_FEATURES, KMEANS_REMOVE_OUTLIERS, KMEANS_OUTLIER_Z,
    KMEANS_TOP_CLUSTERS, KMEANS_MIN_CLUSTER_SIZE,
    PERIODS_PER_YEAR, OUTPUT_DIR
)

from data_io import load_bloomberg_data, preprocess_returns
from portfolio_utils import compute_weighted_portfolio, validate_portfolio_weights
from networks import (
    create_graph, degeneracy_ordering,
    eigenvector_centrality_weights, plotly_network
)
from clustering import (
    hierarchical_clusters_from_corr, cluster_equal_weights, save_dendrogram,
    herc_weights, corr_distance_of_distance_matrix,
    compute_annualized_return_vol, remove_outliers_zscore,
    kmeans_cluster_retvol, build_kmeans_weights, plot_kmeans_scatter
)
from plots import (
    plotly_correlation_heatmap, save_plotly,
    plot_cumulative_returns
)
from metrics import metrics_table_from_values


def restrict_weights_to_universe(weights: pd.Series, universe: list, name: str) -> pd.Series:
    w = weights.reindex(universe).fillna(0.0).astype(float)
    s = float(w.sum())
    if abs(s) > 1e-10:
        w = w / s
    w.name = name
    return w


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
        "eigen_central": "Inv. eigen. (%)",
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


if __name__ == "__main__":
    print("=" * 80)
    print("NETWORK AND CLUSTERING PORTFOLIOS - BLOOMBERG DATA")
    print("=" * 80)

    print("\n[1/10] Loading Bloomberg data...")
    full_data, all_components = load_bloomberg_data("bloomberg_prices.csv")
    if full_data.empty:
        raise SystemExit("Error: Could not load Bloomberg data")
    print(f"  - Loaded {len(full_data)} dates from {full_data.index.min()} to {full_data.index.max()}")
    print(f"  - Found {len(all_components)} stock tickers")

    print(f"\n[2/10] Preprocessing in-sample data ({IN_SAMPLE_START} to {IN_SAMPLE_END})...")
    in_sample_returns, in_sample_correlation, in_sample_prices, prices_full_filtered = preprocess_returns(
        full_data,
        all_components,
        IN_SAMPLE_START,
        IN_SAMPLE_END,
        min_data_availability=MIN_DATA_AVAILABILITY,
        min_price=MIN_PRICE_THRESHOLD
    )
    components = list(in_sample_prices.columns)
    print(f"  - Final universe: {len(components)} stocks")
    print(f"  - In-sample returns shape: {in_sample_returns.shape}")

    print(f"\n[3/10] Preparing out-of-sample data ({OUT_SAMPLE_START} to {OUT_SAMPLE_END})...")
    out_sample_mask = (prices_full_filtered.index >= OUT_SAMPLE_START) & (prices_full_filtered.index <= OUT_SAMPLE_END)
    out_sample_prices = prices_full_filtered.loc[out_sample_mask].copy()
    if len(out_sample_prices) == 0:
        raise ValueError("Out-of-sample window is empty. Check OUT_SAMPLE_START/OUT_SAMPLE_END.")
    print(f"  - Out-of-sample dates: {len(out_sample_prices)} ({out_sample_prices.index.min()} to {out_sample_prices.index.max()})")
    if "FTSE Index" in full_data.columns:
        ftse_in_sample = full_data.loc[in_sample_prices.index, "FTSE Index"].copy()
        ftse_out_sample = full_data.loc[out_sample_prices.index, "FTSE Index"].copy()
    else:
        ftse_in_sample = None
        ftse_out_sample = None

    print("\n[4/10] Generating correlation heatmap...")
    fig_corr_is = plotly_correlation_heatmap(
        in_sample_correlation,
        title=f"Stock Correlation Heatmap ({IN_SAMPLE_START} to {IN_SAMPLE_END})",
        show_values=False
    )
    save_plotly(fig_corr_is, f"corr_heatmap_{IN_SAMPLE_START}_{IN_SAMPLE_END}.html")

    print("\n[5/10] Building correlation network...")
    corr_vals = in_sample_correlation.values
    upper = corr_vals[np.triu_indices_from(corr_vals, k=1)]
    upper = upper[~np.isnan(upper)]
    threshold = np.quantile(upper, NETWORK_THRESHOLD_QUANTILE)
    print(f"  - Network threshold (quantile {NETWORK_THRESHOLD_QUANTILE}): {threshold:.4f}")
    in_sample_graph, in_sample_layout = create_graph(components, in_sample_correlation, threshold)
    print(f"  - Network: {len(in_sample_graph.nodes)} nodes, {len(in_sample_graph.edges)} edges")

    print("\n[6/10] Building Degeneracy portfolio...")
    isolated, independence = degeneracy_ordering(in_sample_graph, components)
    selected_stocks_deg = [s for s in (isolated + independence) if s in components]
    print(f"  - Selected {len(selected_stocks_deg)} stocks (isolated: {len(isolated)}, independent: {len(independence)})")
    deg_w = pd.Series(0.0, index=components, dtype=float)
    if len(selected_stocks_deg) > 0:
        deg_w.loc[selected_stocks_deg] = 1.0 / len(selected_stocks_deg)
    deg_w = restrict_weights_to_universe(deg_w, components, "degeneracy")
    validate_portfolio_weights(deg_w, "degeneracy")
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

    print("\n[7/10] Building Inverse Eigenvector Centrality portfolio...")
    eigen_w_central = eigenvector_centrality_weights(in_sample_graph, components, inverse=True)
    eigen_w_central = restrict_weights_to_universe(eigen_w_central, components, "eigen_central")
    validate_portfolio_weights(eigen_w_central, "eigen_central")
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

    print("\n[8/10] Building Hierarchical Clustering portfolio...")
    cluster_labels, Z_cluster = hierarchical_clusters_from_corr(
        in_sample_correlation,
        method="average",
        max_clusters=HIER_MAX_CLUSTERS,
        min_cluster_size=HIER_MIN_CLUSTER_SIZE
    )
    cluster_w = cluster_equal_weights(cluster_labels)
    cluster_w = restrict_weights_to_universe(cluster_w, components, "cluster_equal")
    validate_portfolio_weights(cluster_w, "cluster_equal")
    save_dendrogram(
        Z_cluster,
        list(in_sample_correlation.index),
        "Dendrogram: Cluster Equal-Weight (average, corr-distance)",
        f"dendrogram_cluster_equal_{IN_SAMPLE_START}_{IN_SAMPLE_END}.png"
    )

    print("\n[9/10] Building HERC portfolio...")
    in_sample_cov = in_sample_returns.cov()
    herc_w = herc_weights(
        in_sample_cov,
        max_depth=HERC_MAX_DEPTH,
        linkage_method=HERC_LINKAGE_METHOD,
        use_distance_of_distance=HERC_USE_DISTANCE_OF_DISTANCE
    )
    herc_w = restrict_weights_to_universe(herc_w, components, "herc")
    validate_portfolio_weights(herc_w, "herc")
    std_dev = np.sqrt(np.diag(in_sample_cov.values))
    denom = np.outer(std_dev, std_dev)
    denom = np.where(abs(denom) < 1e-10, np.nan, denom)
    corr_from_cov = in_sample_cov.values / denom
    corr_from_cov = np.nan_to_num(corr_from_cov, nan=0.0, posinf=0.0, neginf=0.0)
    corr_from_cov = np.clip(corr_from_cov, -1.0, 1.0)
    np.fill_diagonal(corr_from_cov, 1.0)
    corr_cov_df = pd.DataFrame(corr_from_cov, index=in_sample_cov.index, columns=in_sample_cov.index)
    D_herc = corr_distance_of_distance_matrix(corr_cov_df)
    Z_herc = linkage(squareform(D_herc, checks=False), method=HERC_LINKAGE_METHOD)
    save_dendrogram(
        Z_herc,
        list(corr_cov_df.index),
        "Dendrogram: HERC (ward, distance-of-distance)",
        f"dendrogram_herc_{IN_SAMPLE_START}_{IN_SAMPLE_END}.png"
    )

    print("\n[10/10] Building K-Means portfolio...")
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
        top_clusters=KMEANS_TOP_CLUSTERS,
        min_cluster_size=KMEANS_MIN_CLUSTER_SIZE
    )
    kmeans_w = restrict_weights_to_universe(kmeans_w, components, "kmeans")
    validate_portfolio_weights(kmeans_w, "kmeans")
    fig_km_scatter = plot_kmeans_scatter(
        stats_with_clusters,
        picked_clusters=list(picked_clusters_df.index),
        title=f"K-Means (k={KMEANS_K}) Clusters: Annualized Return vs Volatility"
    )
    save_plotly(fig_km_scatter, f"kmeans_scatter_{IN_SAMPLE_START}_{IN_SAMPLE_END}.html")

    print("\n" + "=" * 80)
    print("IN-SAMPLE EVALUATION")
    print("=" * 80)
    in_sample_data = in_sample_prices[components].copy()
    if ftse_in_sample is not None:
        in_sample_data["FTSE100"] = ftse_in_sample
    in_sample_data["degeneracy"] = compute_weighted_portfolio(in_sample_data, deg_w, "degeneracy")
    in_sample_data["eigen_central"] = compute_weighted_portfolio(in_sample_data, eigen_w_central, "eigen_central")
    in_sample_data["cluster_equal"] = compute_weighted_portfolio(in_sample_data, cluster_w, "cluster_equal")
    in_sample_data["herc"] = compute_weighted_portfolio(in_sample_data, herc_w, "herc")
    in_sample_data["kmeans"] = compute_weighted_portfolio(in_sample_data, kmeans_w, "kmeans")
    weights_map = {
        "degeneracy": deg_w,
        "eigen_central": eigen_w_central,
        "cluster_equal": cluster_w,
        "kmeans": kmeans_w,
        "herc": herc_w,
    }
    portfolios = ["degeneracy", "eigen_central", "cluster_equal", "kmeans", "herc"]
    if ftse_in_sample is not None:
        portfolios = ["FTSE100"] + portfolios
    style_map = {
        "FTSE100": ("grey", "-", "FTSE 100 Index"),
        "degeneracy": ("green", "-", "Degeneracy Selection"),
        "eigen_central": ("orange", "-", "Eigen Centrality (Inverse)"),
        "cluster_equal": ("magenta", ":", "Cluster Equal Weight"),
        "kmeans": ("black", "-", "K-Means"),
        "herc": ("brown", ":", "HERC"),
    }
    metrics_is = metrics_table_from_values(
        in_sample_data,
        portfolios,
        bench_name="FTSE100" if ftse_in_sample is not None else None,
        periods_per_year=PERIODS_PER_YEAR,
        asset_returns=in_sample_returns[components],
        weights_map=weights_map,
    )
    print("\nIn-Sample Performance Metrics:")
    print(metrics_is.to_string())
    fig_cum_is = plot_cumulative_returns(in_sample_data, portfolios, style_map)
    save_plotly(fig_cum_is, f"cumulative_in_sample_{IN_SAMPLE_START}_{IN_SAMPLE_END}.html")

    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE EVALUATION")
    print("=" * 80)
    out_sample_data = out_sample_prices.copy()
    if ftse_out_sample is not None:
        out_sample_data["FTSE100"] = ftse_out_sample
    out_sample_data["degeneracy"] = compute_weighted_portfolio(out_sample_data, deg_w, "degeneracy")
    out_sample_data["eigen_central"] = compute_weighted_portfolio(out_sample_data, eigen_w_central, "eigen_central")
    out_sample_data["cluster_equal"] = compute_weighted_portfolio(out_sample_data, cluster_w, "cluster_equal")
    out_sample_data["herc"] = compute_weighted_portfolio(out_sample_data, herc_w, "herc")
    out_sample_data["kmeans"] = compute_weighted_portfolio(out_sample_data, kmeans_w, "kmeans")
    out_sample_asset_returns = np.log(out_sample_prices).diff().dropna()
    metrics_os = metrics_table_from_values(
        out_sample_data,
        portfolios,
        bench_name="FTSE100" if ftse_out_sample is not None else None,
        periods_per_year=PERIODS_PER_YEAR,
        asset_returns=out_sample_asset_returns,
        weights_map=weights_map,
    )
    print("\nOut-of-Sample Performance Metrics:")
    print(metrics_os.to_string())
    fig_cum_os = plot_cumulative_returns(out_sample_data, portfolios, style_map)
    save_plotly(fig_cum_os, f"cumulative_out_sample_{OUT_SAMPLE_START}_{OUT_SAMPLE_END}.html")

    print("\n" + "=" * 80)
    print("SAVING OUTPUTS")
    print("=" * 80)
    appendix_df = build_appendix_holdings_table(weights_map, top_n_eigen_herc=15)
    appendix_df.to_csv(OUTPUT_DIR / "appendix_weights_table.csv")
    print(f"\nCompleted. Outputs saved to: {OUTPUT_DIR.resolve()}")