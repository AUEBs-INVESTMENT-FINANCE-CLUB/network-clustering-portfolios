from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go


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
    
    if len(graph.nodes) > 0:
        layout = nx.spring_layout(
            graph, 
            k=2.5,            
            iterations=100,     
            weight=None,        
            seed=42
        )
    else:
        layout = {}
        
    return graph, layout


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

    ec = pd.Series(ec_dict, name="eigen_centrality").reindex(components).fillna(0.0).astype(float)

    if ec.max() <= 0:
        w = pd.Series(1.0, index=components, name="eigen_centrality")
        return w / w.sum()

    if inverse:
        eps = 1e-12
        inv = (ec.max() - ec).clip(lower=0.0) + eps
        ec = inv

    if abs(ec.sum()) > 1e-10:
        ec = ec / ec.sum()

    ec.name = "eigen_centrality"
    return ec


def plotly_network(
    graph: nx.Graph,
    layout: dict,
    title: str = "Network",
    selected: list | None = None,
    show_labels: bool = False,
    width: int = 1000,
    height: int = 700,
    selected_color: str = "#EC2049",
    default_color: str = "#D3D3D3",
    color_mode: str = "selected",
    node_values: np.ndarray | None = None,
    colorbar_title: str = "Value",
) -> go.Figure:
    if selected is None:
        selected = []

    nodes = list(graph.nodes())
    if len(nodes) == 0:
        fig = go.Figure()
        fig.update_layout(title=title, width=width, height=height)
        return fig

    edge_x, edge_y = [], []
    for u, v, _d in graph.edges(data=True):
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
    is_selected = np.array([n in selected for n in nodes])

    if color_mode == "continuous":
        if node_values is None:
            node_values = deg_vals.copy()
        vals = np.asarray(node_values, dtype=float)
        vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 0.0
        if vmax <= 0:
            vmax = 1.0
        size = 12 + 28 * (vals / vmax)
        size = np.clip(size, 10, 45)

        color_vals = vals
        colorscale = "Plasma"
        showscale = True
        colorbar = dict(title=colorbar_title)

        line_width = np.where(is_selected, 2.5, 1.0)
        line_color = np.where(is_selected, "rgba(0,0,0,0.85)", "rgba(0,0,0,0.35)")

        hovertext = [
            f"<b>{n}</b><br>value: {float(vals[i]):.6f}<br>degree: {degrees.get(n, 0)}"
            for i, n in enumerate(nodes)
        ]
    else:
        size = 12 + 28 * (deg_vals / (deg_vals.max() if deg_vals.max() > 0 else 1.0))
        size = np.clip(size, 10, 45)

        if color_mode == "selected":
            color_vals = np.where(is_selected, selected_color, default_color)
            colorscale = None
            showscale = False
            colorbar = None
        else:
            color_vals = deg_vals
            colorscale = "Plasma"
            showscale = True
            colorbar = dict(title="Degree")

        line_width = np.where(is_selected, 2.5, 1.0)
        line_color = np.where(is_selected, "rgba(0,0,0,0.85)", "rgba(0,0,0,0.35)")

        hovertext = [f"<b>{n}</b><br>degree: {degrees.get(n, 0)}" for n in nodes]

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