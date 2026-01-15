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
from portfolio_utils import (
    compute_weighted_portfolio, compute_equal_weighted_portfolio, validate_portfolio_weights
)
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
    plotly_correlation_heatmap, save_plotly,
    plot_cumulative_returns, plot_mean_variance_scatter
)
from metrics import metrics_table_from_values


def compute_markowitz_weights(returns: pd.DataFrame, allow_short: bool = False):
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


if __name__ == "__main__":

    metadata, components = load_ftse100_data("ftse_stock_prices.csv")
    if metadata.empty:
        raise SystemExit(1)

    in_sample_raw = metadata[IN_SAMPLE_START:IN_SAMPLE_END].copy()
    in_sample_returns, in_sample_correlation, in_sample_prices = preprocess_returns(
        in_sample_raw, components, min_data_availability=MIN_DATA_AVAILABILITY
    )

    fig_corr_is = plotly_correlation_heatmap(
        in_sample_correlation,
        title=f"FTSE100 Constituents Correlation Heatmap ({IN_SAMPLE_START}-{IN_SAMPLE_END})",
        show_values=False
    )
    save_plotly(fig_corr_is, f"corr_heatmap_{IN_SAMPLE_START}_{IN_SAMPLE_END}.html")

    in_sample_data = in_sample_prices.copy()
    in_sample_data["FTSE100"] = metadata.loc[in_sample_data.index, "FTSE100"].ffill()

    components = list(in_sample_returns.columns)

    corr_vals = in_sample_correlation.values
    upper = corr_vals[np.triu_indices_from(corr_vals, k=1)]
    upper = upper[~np.isnan(upper)]
    threshold = np.quantile(upper, NETWORK_THRESHOLD_QUANTILE)

    in_sample_graph, in_sample_layout = create_graph(
        components, in_sample_correlation, threshold
    )
    in_sample_full_graph = create_full_correlation_graph(
        components, in_sample_correlation
    )

    minvar_w, maxsharpe_w = compute_markowitz_weights(in_sample_returns, allow_short=False)
    validate_portfolio_weights(minvar_w, "markowitz_minvar")
    validate_portfolio_weights(maxsharpe_w, "markowitz_maxsharpe")

    in_sample_data["markowitz_minvar"] = compute_weighted_portfolio(
        in_sample_data, minvar_w, "markowitz_minvar"
    )
    in_sample_data["markowitz_maxsharpe"] = compute_weighted_portfolio(
        in_sample_data, maxsharpe_w, "markowitz_maxsharpe"
    )

    isolated, independence = degeneracy_ordering(in_sample_graph, components)
    selected_stocks_deg = isolated + independence
    in_sample_data["degeneracy"] = compute_equal_weighted_portfolio(
        in_sample_data, selected_stocks_deg, "degeneracy"
    )

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

    eigen_w_central = eigenvector_centrality_weights(
        in_sample_graph, components, inverse=True
    )
    validate_portfolio_weights(eigen_w_central, "eigen_central")
    in_sample_data["eigen_central"] = compute_weighted_portfolio(
        in_sample_data, eigen_w_central, "eigen_central"
    )

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

    cluster_labels, Z_cluster = hierarchical_clusters_from_corr(
        in_sample_correlation, method="average", max_clusters=HIER_MAX_CLUSTERS
    )
    cluster_w = cluster_equal_weights(cluster_labels)
    validate_portfolio_weights(cluster_w, "cluster_equal")
    in_sample_data["cluster_equal"] = compute_weighted_portfolio(
        in_sample_data, cluster_w, "cluster_equal"
    )
    save_dendrogram(
        Z_cluster,
        list(in_sample_correlation.index),
        "Dendrogram: Cluster Equal-Weight (average, corr-distance)",
        f"dendrogram_cluster_equal_{IN_SAMPLE_START}_{IN_SAMPLE_END}.png"
    )

    in_sample_cov = in_sample_returns.cov()
    herc_w = herc_weights(
        in_sample_cov,
        max_depth=HERC_MAX_DEPTH,
        linkage_method="ward",
        use_distance_of_distance=True
    )
    validate_portfolio_weights(herc_w, "herc")
    in_sample_data["herc"] = compute_weighted_portfolio(
        in_sample_data, herc_w, "herc"
    )

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

    stats = compute_annualized_return_vol(
        in_sample_returns, periods_per_year=PERIODS_PER_YEAR
    )

    if KMEANS_REMOVE_OUTLIERS:
        stats_for_kmeans, _ = remove_outliers_zscore(stats, ["Return", "Volatility"], z=KMEANS_OUTLIER_Z)
    else:
        stats_for_kmeans = stats

    # ELBOW REMOVED (function + call removed)

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
    validate_portfolio_weights(kmeans_w, "kmeans")
    in_sample_data["kmeans"] = compute_weighted_portfolio(
        in_sample_data, kmeans_w, "kmeans"
    )

    fig_km_scatter = plot_kmeans_scatter(
        stats_with_clusters,
        picked_clusters=list(picked_clusters_df.index),
        title=f"K-Means (k={KMEANS_K}) Clusters: Annualized Return vs Volatility"
    )
    save_plotly(fig_km_scatter, f"kmeans_scatter_{IN_SAMPLE_START}_{IN_SAMPLE_END}.html")

    portfolios = [
        "FTSE100",
        "markowitz_minvar",
        "markowitz_maxsharpe",
        "degeneracy",
        "eigen_central",
        "cluster_equal",
        "kmeans",
        "herc",
    ]

    style_map = {
        "FTSE100": ("grey", "-", "FTSE 100 Index"),
        "markowitz_minvar": ("blue", "-", "Markowitz Min-Var"),
        "markowitz_maxsharpe": ("purple", "--", "Markowitz Max-Sharpe"),
        "degeneracy": ("green", "-", "Degeneracy Selection"),
        "eigen_central": ("orange", "-", "Eigen Centrality (Inverse)"),
        "cluster_equal": ("magenta", ":", "Cluster Equal Weight"),
        "kmeans": ("black", "-", "K-Means"),
        "herc": ("brown", ":", "HERC"),
    }

    label_map = {p: style_map[p][2] for p in portfolios}
    color_map = {p: style_map[p][0] for p in portfolios}

    metrics_is = metrics_table_from_values(
        in_sample_data, portfolios, bench_name="FTSE100"
    )
    metrics_is.to_csv(OUTPUT_DIR / f"metrics_in_sample_{IN_SAMPLE_START}_{IN_SAMPLE_END}.csv")

    fig_cum_is = plot_cumulative_returns(
        in_sample_data,
        portfolios,
        style_map,
        f"Cumulative Returns (%) ({IN_SAMPLE_START}-{IN_SAMPLE_END})"
    )
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

    out_sample_raw = metadata[OUT_SAMPLE_START:OUT_SAMPLE_END].copy()

    out_sample_returns, out_sample_correlation, out_sample_prices = preprocess_returns(
        out_sample_raw, components, min_data_availability=MIN_DATA_AVAILABILITY
    )

    out_sample_data = out_sample_prices.copy()
    out_sample_data["FTSE100"] = metadata.loc[out_sample_data.index, "FTSE100"].ffill()

    out_sample_data["markowitz_minvar"] = compute_weighted_portfolio(out_sample_data, minvar_w, "markowitz_minvar")
    out_sample_data["markowitz_maxsharpe"] = compute_weighted_portfolio(out_sample_data, maxsharpe_w, "markowitz_maxsharpe")
    out_sample_data["degeneracy"] = compute_equal_weighted_portfolio(out_sample_data, selected_stocks_deg, "degeneracy")
    out_sample_data["eigen_central"] = compute_weighted_portfolio(out_sample_data, eigen_w_central, "eigen_central")
    out_sample_data["cluster_equal"] = compute_weighted_portfolio(out_sample_data, cluster_w, "cluster_equal")
    out_sample_data["herc"] = compute_weighted_portfolio(out_sample_data, herc_w, "herc")
    out_sample_data["kmeans"] = compute_weighted_portfolio(out_sample_data, kmeans_w, "kmeans")

    metrics_os = metrics_table_from_values(
        out_sample_data, portfolios, bench_name="FTSE100"
    )
    metrics_os.to_csv(OUTPUT_DIR / f"metrics_out_sample_{OUT_SAMPLE_START}_{OUT_SAMPLE_END}.csv")

    print("\n--- OUT-OF-SAMPLE METRICS (Test Phase) ---")
    print(metrics_os)

    fig_cum_os = plot_cumulative_returns(
        out_sample_data,
        portfolios,
        style_map,
        f"Cumulative Returns (%) ({OUT_SAMPLE_START}-{OUT_SAMPLE_END})"
    )
    save_plotly(fig_cum_os, f"cumulative_out_sample_{OUT_SAMPLE_START}_{OUT_SAMPLE_END}.html")

    fig_mv_os = plot_mean_variance_scatter(
        out_sample_data,
        portfolios,
        label_map,
        color_map,
        f"Mean-Variance Analysis ({OUT_SAMPLE_START}-{OUT_SAMPLE_END})"
    )
    save_plotly(fig_mv_os, f"mean_variance_out_sample_{OUT_SAMPLE_START}_{OUT_SAMPLE_END}.html")

    print(f"\nCompleted successfully. All outputs saved to: {OUTPUT_DIR.resolve()}")
