"""
graph_metrics.py

Graph-based structural metrics for benchmark evaluation.

This module assumes that each record contains a graph representation
of entities or reasoning structure.

Expected input per record:
    record["graph"] = networkx.Graph or networkx.DiGraph

If directed graphs are used, metrics are automatically converted
to undirected where needed for consistency.

Metrics implemented
-------------------
- Graph density
- Average node degree
- Graph diameter
- Connected components
- Average shortest path length
- Clustering coefficient
- Degree centralization

Dependencies:
    pip install networkx
"""

import networkx as nx


###########################################################
# Utility
###########################################################

def safe_graph(record):
    """
    Retrieve graph object from record safely.
    """
    return record.get("graph", None)


def to_undirected(G):
    """
    Convert to undirected graph if needed.
    """
    if G is None:
        return None
    if G.is_directed():
        return G.to_undirected()
    return G


###########################################################
# Core Metrics
###########################################################

def graph_density(G):
    if G is None or len(G) == 0:
        return 0.0
    return nx.density(G)


def average_node_degree(G):
    if G is None or len(G) == 0:
        return 0.0
    degrees = [deg for _, deg in G.degree()]
    return sum(degrees) / len(degrees)


def graph_diameter(G):
    """
    Diameter of largest connected component.
    """
    if G is None or len(G) == 0:
        return 0

    G = to_undirected(G)

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
    else:
        subgraph = G

    try:
        return nx.diameter(subgraph)
    except nx.NetworkXError:
        return 0


def connected_components(G):
    if G is None or len(G) == 0:
        return 0

    G = to_undirected(G)

    return nx.number_connected_components(G)


def average_shortest_path(G):
    """
    Average shortest path over largest connected component.
    """
    if G is None or len(G) == 0:
        return 0.0

    G = to_undirected(G)

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
    else:
        subgraph = G

    try:
        return nx.average_shortest_path_length(subgraph)
    except nx.NetworkXError:
        return 0.0


def clustering_coefficient(G):
    if G is None or len(G) == 0:
        return 0.0
    G = to_undirected(G)
    return nx.average_clustering(G)


###########################################################
# Degree Centralization
###########################################################

def degree_centralization(G):
    """
    Freeman degree centralization:
    measures inequality of node degrees.
    """
    if G is None or len(G) == 0:
        return 0.0

    degrees = [deg for _, deg in G.degree()]
    n = len(degrees)

    if n <= 2:
        return 0.0

    max_deg = max(degrees)

    numerator = sum(max_deg - d for d in degrees)
    denominator = (n - 1) * (n - 2)

    if denominator == 0:
        return 0.0

    return numerator / denominator


###########################################################
# Main Compute Function
###########################################################

def compute(records):
    """
    Compute graph metrics for each record.
    """

    for record in records:

        G = safe_graph(record)

        # -------------------------
        # Global structure metrics
        # -------------------------
        record["graph_density"] = graph_density(G)
        record["average_node_degree"] = average_node_degree(G)
        record["graph_diameter"] = graph_diameter(G)
        record["connected_components"] = connected_components(G)
        record["average_shortest_path"] = average_shortest_path(G)
        record["clustering_coefficient"] = clustering_coefficient(G)
        record["degree_centralization"] = degree_centralization(G)

    return records