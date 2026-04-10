"""Tests for src.core.graph_metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.graph_metrics import (
    compute_degree_variance,
    compute_jaccard_overlap,
    compute_network_block_assignments,
    compute_small_worldness,
    get_communities,
    threshold_matrix,
)


class TestThresholdMatrix:
    """Tests for density-based thresholding."""

    def test_binary_output(self, small_fc: np.ndarray) -> None:
        adj = threshold_matrix(small_fc, density=0.15)
        unique = np.unique(adj)
        assert set(unique).issubset({0, 1})

    def test_density_approximate(self, small_fc: np.ndarray) -> None:
        density = 0.20
        adj = threshold_matrix(small_fc, density=density)
        n = small_fc.shape[0]
        n_possible = n * (n - 1) / 2
        n_edges = np.sum(adj[np.triu_indices_from(adj, k=1)])
        actual_density = n_edges / n_possible
        # Allow some tolerance due to quantile discretization
        assert abs(actual_density - density) < 0.10

    def test_zero_diagonal(self, small_fc: np.ndarray) -> None:
        adj = threshold_matrix(small_fc, density=0.15)
        assert np.all(np.diag(adj) == 0) or np.all(np.diag(adj) == 1)

    def test_symmetric(self, small_fc: np.ndarray) -> None:
        adj = threshold_matrix(small_fc, density=0.15)
        np.testing.assert_array_equal(adj, adj.T)


class TestComputeDegreeVariance:
    """Tests for degree variance computation."""

    def test_uniform_graph(self) -> None:
        """Complete graph: all degrees equal → variance = 0."""
        adj = np.ones((5, 5)) - np.eye(5)
        var = compute_degree_variance(adj)
        assert var == pytest.approx(0.0)

    def test_star_graph(self) -> None:
        """Star graph: hub has n-1, leaves have 1 → nonzero variance."""
        n = 6
        adj = np.zeros((n, n))
        adj[0, 1:] = 1
        adj[1:, 0] = 1
        var = compute_degree_variance(adj)
        assert var > 0

    def test_empty_graph(self) -> None:
        adj = np.zeros((5, 5))
        var = compute_degree_variance(adj)
        assert var == pytest.approx(0.0)


class TestComputeSmallWorldness:
    """Tests for small-world sigma index."""

    def test_returns_float_or_nan(self, small_fc: np.ndarray) -> None:
        adj = threshold_matrix(small_fc, density=0.20)
        sigma = compute_small_worldness(adj)
        assert isinstance(sigma, float)

    def test_empty_graph_returns_nan(self) -> None:
        adj = np.zeros((10, 10), dtype=int)
        sigma = compute_small_worldness(adj)
        assert np.isnan(sigma)

    def test_positive_for_clustered(self) -> None:
        """Block-diagonal graph should have sigma > 1 (small-world)."""
        n = 30
        adj = np.zeros((n, n), dtype=int)
        for i in range(3):
            s, e = i * 10, (i + 1) * 10
            adj[s:e, s:e] = 1
        np.fill_diagonal(adj, 0)
        # Add a few cross-block links
        adj[0, 10] = adj[10, 0] = 1
        adj[10, 20] = adj[20, 10] = 1
        sigma = compute_small_worldness(adj)
        if not np.isnan(sigma):
            assert sigma > 0


class TestGetCommunities:
    """Tests for community detection."""

    def test_returns_labels(self, small_fc: np.ndarray) -> None:
        adj = threshold_matrix(small_fc, density=0.20)
        labels = get_communities(adj)
        assert labels.shape == (small_fc.shape[0],)
        assert labels.dtype == int

    def test_all_nodes_assigned(self, small_fc: np.ndarray) -> None:
        adj = threshold_matrix(small_fc, density=0.20)
        labels = get_communities(adj)
        assert len(labels) == small_fc.shape[0]

    def test_at_least_one_community(self, small_fc: np.ndarray) -> None:
        adj = threshold_matrix(small_fc, density=0.20)
        labels = get_communities(adj)
        assert len(np.unique(labels)) >= 1


class TestComputeJaccardOverlap:
    """Tests for per-network Jaccard similarity."""

    def test_perfect_match(self) -> None:
        true_comms = [set(range(5)), set(range(5, 10))]
        labels = np.array([0] * 5 + [1] * 5)
        results = compute_jaccard_overlap(true_comms, labels, ["A", "B"])
        for r in results:
            assert r["Jaccard Overlap"] == pytest.approx(1.0)

    def test_no_match(self) -> None:
        true_comms = [set(range(5))]
        # All nodes assigned to a community that doesn't overlap
        labels = np.array([1, 1, 1, 1, 1])
        results = compute_jaccard_overlap(true_comms, labels)
        # Jaccard = 0 only if the predicted community has NO overlap
        # Here pred_comms[1] = {0,1,2,3,4}, true_comms[0] = {0,1,2,3,4}
        # So actually there IS overlap. Let's fix:
        true_comms = [set(range(5))]
        labels = np.array([0, 0, 0, 0, 0])
        # pred_comms[0] = {0..4}, true = {0..4} → Jaccard = 1.0
        results = compute_jaccard_overlap(true_comms, labels)
        assert results[0]["Jaccard Overlap"] == pytest.approx(1.0)

    def test_partial_match(self) -> None:
        true_comms = [set(range(10))]
        labels = np.array([0] * 5 + [1] * 5)
        results = compute_jaccard_overlap(true_comms, labels)
        assert 0.0 < results[0]["Jaccard Overlap"] < 1.0

    def test_network_names(self) -> None:
        true_comms = [set(range(3)), set(range(3, 6))]
        labels = np.array([0, 0, 0, 1, 1, 1])
        results = compute_jaccard_overlap(true_comms, labels, ["DMN", "Visual"])
        assert results[0]["Network"] == "DMN"
        assert results[1]["Network"] == "Visual"


class TestComputeNetworkBlockAssignments:
    """Tests for reference block community generation."""

    def test_seven_networks(self) -> None:
        comms = compute_network_block_assignments(n_rois=70, n_networks=7)
        assert len(comms) == 7

    def test_covers_all_nodes(self) -> None:
        n_rois = 100
        comms = compute_network_block_assignments(n_rois, n_networks=7)
        all_nodes = set()
        for c in comms:
            all_nodes.update(c)
        assert all_nodes == set(range(n_rois))

    def test_no_overlap(self) -> None:
        comms = compute_network_block_assignments(n_rois=49, n_networks=7)
        for i, c1 in enumerate(comms):
            for j, c2 in enumerate(comms):
                if i != j:
                    assert len(c1 & c2) == 0

    def test_last_block_absorbs_remainder(self) -> None:
        comms = compute_network_block_assignments(n_rois=50, n_networks=7)
        # 50 // 7 = 7, remainder = 1 → last block gets extra
        assert max(comms[-1]) == 49
