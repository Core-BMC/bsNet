"""Graph analysis utilities for network topology and community detection.

This module consolidates graph analysis functions including thresholding,
degree distribution, small-worldness computation, and community detection.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

# Constants
DEFAULT_DENSITY: float = 0.15
DEFAULT_N_NETWORKS: int = 7
RANDOM_GRAPH_TRIALS: int = 1


def threshold_matrix(corr_matrix: np.ndarray, density: float = 0.15) -> (
    np.ndarray
):
    """Apply density-based thresholding to a correlation matrix.

    Zeros the diagonal, computes quantile on absolute values, and returns
    a binary adjacency matrix where edges above the threshold are set to 1.

    Args:
        corr_matrix: Square correlation or connectivity matrix (n x n).
        density: Target edge density as fraction of possible edges (0 to 1).
            Default is 0.15.

    Returns:
        Binary adjacency matrix (n x n) with dtype int.
    """
    fc = corr_matrix.copy()
    np.fill_diagonal(fc, 0)
    fc_flat = fc[np.triu_indices_from(fc, k=1)]
    thresh = np.quantile(np.abs(fc_flat), 1 - density)
    adj = (np.abs(fc) >= thresh).astype(int)
    return adj


def compute_degree_variance(adj_matrix: np.ndarray) -> float:
    """Compute degree distribution variance from adjacency matrix.

    Args:
        adj_matrix: Binary or weighted adjacency matrix (n x n).

    Returns:
        Variance of node degrees.
    """
    degrees = np.sum(adj_matrix, axis=1)
    deg_var = float(np.var(degrees))
    return deg_var


def compute_small_worldness(adj_matrix: np.ndarray) -> float:
    """Compute small-world sigma index of network.

    Compares clustering coefficient and average shortest path length to
    random graph equivalents. Handles disconnected graphs gracefully.

    Args:
        adj_matrix: Binary adjacency matrix (n x n).

    Returns:
        Small-worldness index sigma (C/C_rand) / (L/L_rand).
        Returns np.nan if graph is disconnected or computation fails.
    """
    G = nx.from_numpy_array(adj_matrix)

    # Handle disconnected graphs by taking largest component
    if not nx.is_connected(G):
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        if len(components) == 0 or len(components[0]) < 3:
            return np.nan
        G = G.subgraph(components[0]).copy()

    try:
        # Compute clustering and path length
        C = nx.average_clustering(G)
        L = nx.average_shortest_path_length(G)

        # Random graph equivalent
        G_rand = nx.erdos_renyi_graph(
            len(G.nodes()), nx.density(G)
        )
        if not nx.is_connected(G_rand):
            comps = sorted(
                nx.connected_components(G_rand), key=len, reverse=True
            )
            if len(comps) > 0:
                G_rand = G_rand.subgraph(comps[0]).copy()
            else:
                return np.nan

        C_rand = nx.average_clustering(G_rand)
        L_rand = nx.average_shortest_path_length(G_rand)

        # Compute sigma
        gamma = C / C_rand if C_rand > 0 else 1.0
        lambda_ = L / L_rand if L_rand > 0 else 1.0
        sigma = gamma / lambda_ if lambda_ > 0 else np.nan

        return float(sigma)

    except (nx.NetworkXError, ZeroDivisionError):
        return np.nan


def get_communities(adj_matrix: np.ndarray) -> np.ndarray:
    """Detect communities using Louvain with fallback to greedy modularity.

    Args:
        adj_matrix: Binary adjacency matrix (n x n).

    Returns:
        Array of community label per node (n,).

    Raises:
        ImportError: If neither louvain nor greedy_modularity is available.
    """
    G = nx.from_numpy_array(adj_matrix)

    try:
        from networkx.algorithms.community import louvain_communities

        comms = louvain_communities(G)
    except ImportError:
        try:
            from networkx.algorithms.community import (
                greedy_modularity_communities,
            )

            comms = greedy_modularity_communities(G)
        except ImportError as e:
            raise ImportError(
                "Neither louvain_communities nor greedy_modularity_communities "
                "available in networkx"
            ) from e

    labels = np.zeros(len(G.nodes()), dtype=int)
    for i, community in enumerate(comms):
        for node in community:
            labels[node] = i

    return labels


def compute_jaccard_overlap(
    true_communities: list[set],
    pred_labels: np.ndarray,
    network_names: list[str] | None = None,
) -> list[dict[str, float | str]]:
    """Compute per-network Jaccard similarity between true and predicted communities.

    For each true community, finds the best-matching predicted community and
    computes Jaccard overlap.

    Args:
        true_communities: List of sets, each containing node indices for a
            ground truth community.
        pred_labels: Array (n,) of predicted community labels per node.
        network_names: Optional list of names corresponding to true_communities.
            If None, uses integer indices.

    Returns:
        List of dicts with keys "Network" and "Jaccard Overlap".
    """
    if network_names is None:
        network_names = [str(i) for i in range(len(true_communities))]

    # Build predicted communities from labels
    pred_comms = {}
    for node, label in enumerate(pred_labels):
        if label not in pred_comms:
            pred_comms[label] = set()
        pred_comms[label].add(node)

    results = []
    for net_idx, true_set in enumerate(true_communities):
        best_jaccard = 0.0

        for pred_set in pred_comms.values():
            intersection = len(true_set.intersection(pred_set))
            union = len(true_set.union(pred_set))

            if union > 0:
                jaccard = intersection / union
                if jaccard > best_jaccard:
                    best_jaccard = jaccard

        results.append(
            {
                "Network": network_names[net_idx],
                "Jaccard Overlap": float(best_jaccard),
            }
        )

    return results


def compute_network_block_assignments(
    n_rois: int, n_networks: int = DEFAULT_N_NETWORKS
) -> list[set]:
    """Generate ground truth community sets based on block structure.

    Creates a block-diagonal assignment where ROIs are evenly partitioned
    into n_networks communities (as used in simulate.py for Yeo 7-networks).

    Args:
        n_rois: Total number of ROIs/nodes.
        n_networks: Number of ground truth networks/communities.
            Default is 7.

    Returns:
        List of sets, each containing node indices for a network/community.
    """
    block_size = n_rois // n_networks
    true_communities = []

    for i in range(n_networks):
        start = i * block_size
        end = ((i + 1) * block_size) if (i < (n_networks - 1)) else n_rois
        true_communities.append(set(range(start, end)))

    return true_communities
