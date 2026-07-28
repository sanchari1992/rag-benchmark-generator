"""
graph_metrics.py

Graph-based structural metrics for benchmark evaluation.

This module builds an entity graph from each benchmark question and
computes graph-based metrics that approximate reasoning complexity.

Metrics implemented
-------------------
Structure
- Graph density
- Average node degree
- Graph diameter
- Connected components
- Average shortest path length
- Clustering coefficient
- Degree centralization

Reasoning
- Reasoning depth
- Maximum reasoning distance
- Multi-hop entity pair count
- Graph expansion ratio

Dependencies:
    pip install networkx
"""

import networkx as nx
from benchmark_generation.metrics.entity_graph_builder import build_entity_graph


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
    Average shortest path over the largest connected component.
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
    Freeman degree centralization.
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
# Reasoning Metrics
###########################################################

def reasoning_depth(G):
    """
    Approximate reasoning depth.

    If the graph is a DAG, use the longest directed path.
    Otherwise fall back to graph diameter.
    """
    if G is None or len(G) == 0:
        return 0

    try:
        if G.is_directed() and nx.is_directed_acyclic_graph(G):
            return nx.dag_longest_path_length(G)
    except Exception:
        pass

    return graph_diameter(G)


def reasoning_distance(G):
    """
    Maximum reasoning distance between two connected entities.
    """
    if G is None or len(G) == 0:
        return 0

    return graph_diameter(G)


def multihop_entity_pair_count(G):
    """
    Count entity pairs requiring two or more hops.
    """
    if G is None or len(G) == 0:
        return 0

    G = to_undirected(G)

    count = 0

    for source, paths in nx.all_pairs_shortest_path_length(G):

        for target, distance in paths.items():

            if source < target and distance >= 2:
                count += 1

    return count


def graph_expansion_ratio(G):
    """
    Edge-to-node ratio.

    Higher values indicate denser reasoning structures.
    """
    if G is None or G.number_of_nodes() == 0:
        return 0.0

    return G.number_of_edges() / G.number_of_nodes()


###########################################################
# Main Compute Function
###########################################################

def compute(records):
    """
    Compute graph metrics for each record.
    """

    for record in records:

        G = build_entity_graph(record["question"])

        # -------------------------
        # Structural metrics
        # -------------------------

        record["graph_density"] = round(
            graph_density(G), 3
        )

        record["average_node_degree"] = round(
            average_node_degree(G), 3
        )

        record["graph_diameter"] = graph_diameter(G)

        record["connected_components"] = connected_components(G)

        record["average_shortest_path"] = round(
            average_shortest_path(G), 3
        )

        record["clustering_coefficient"] = round(
            clustering_coefficient(G), 3
        )

        record["degree_centralization"] = round(
            degree_centralization(G), 3
        )

        # -------------------------
        # Reasoning metrics
        # -------------------------

        record["reasoning_depth"] = reasoning_depth(G)

        record["reasoning_distance"] = reasoning_distance(G)

        record["multihop_entity_pair_count"] = (
            multihop_entity_pair_count(G)
        )

        record["graph_expansion_ratio"] = round(
            graph_expansion_ratio(G), 3
        )

    return records