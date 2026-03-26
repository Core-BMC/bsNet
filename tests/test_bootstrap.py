"""Tests for src.core.bootstrap module."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.bootstrap import (
    block_bootstrap_indices,
    correct_attenuation,
    estimate_optimal_block_length,
    fisher_z,
    fisher_z_inv,
    spearman_brown,
)


class TestFisherZ:
    """Tests for Fisher-Z transform and inverse."""

    def test_roundtrip(self) -> None:
        for r in [0.0, 0.3, 0.5, 0.8, -0.5]:
            z = fisher_z(r)
            r_back = fisher_z_inv(z)
            assert r_back == pytest.approx(r, abs=1e-4)

    def test_clipping(self) -> None:
        z = fisher_z(1.5)  # beyond valid range
        assert np.isfinite(z)

    def test_zero(self) -> None:
        assert fisher_z(0.0) == pytest.approx(0.0, abs=1e-6)


class TestSpearmanBrown:
    """Tests for Spearman-Brown prophecy formula."""

    def test_k_equals_1(self) -> None:
        """k=1 should return the same reliability."""
        r = 0.7
        assert spearman_brown(r, k=1) == pytest.approx(r, abs=1e-4)

    def test_k_doubles(self) -> None:
        """Doubling test length should increase reliability."""
        r = 0.5
        r_doubled = spearman_brown(r, k=2)
        assert r_doubled > r

    def test_monotonic_in_k(self) -> None:
        """Reliability should increase with k."""
        r = 0.4
        prev = r
        for k in [2, 4, 8, 16]:
            cur = spearman_brown(r, k)
            assert cur >= prev
            prev = cur

    def test_clipping(self) -> None:
        """Extreme inputs should not produce NaN."""
        result = spearman_brown(0.0001, 100)
        assert np.isfinite(result)


class TestCorrectAttenuation:
    """Tests for attenuation correction."""

    def test_output_range(self) -> None:
        result = correct_attenuation(0.8, 0.98, 0.6, k=7.5)
        assert -1.0 <= result <= 1.0

    def test_with_prior(self) -> None:
        r1 = correct_attenuation(0.8, 0.98, 0.6, k=7.5, empirical_prior=None)
        r2 = correct_attenuation(0.8, 0.98, 0.6, k=7.5, empirical_prior=(0.25, 0.05))
        # Prior should shift the result
        assert r1 != pytest.approx(r2, abs=1e-6)

    def test_finite_output(self) -> None:
        """Should not produce NaN/Inf even with edge-case inputs."""
        result = correct_attenuation(0.01, 0.01, 0.01, k=100)
        assert np.isfinite(result)


class TestEstimateOptimalBlockLength:
    """Tests for dynamic block length estimation."""

    def test_returns_positive_int(self, synthetic_ts: np.ndarray) -> None:
        bl = estimate_optimal_block_length(synthetic_ts)
        assert isinstance(bl, int)
        assert bl >= 1

    def test_short_timeseries(self) -> None:
        ts = np.random.randn(3, 5)
        bl = estimate_optimal_block_length(ts)
        assert bl <= 3

    def test_upper_bound(self, synthetic_ts: np.ndarray) -> None:
        n_samples = synthetic_ts.shape[0]
        bl = estimate_optimal_block_length(synthetic_ts)
        assert bl <= n_samples // 4 or bl == 5  # max clamp


class TestBlockBootstrapIndices:
    """Tests for block bootstrap index generation."""

    def test_output_shape(self) -> None:
        idx = block_bootstrap_indices(100, block_size=10, n_blocks=5)
        assert len(idx) == 50  # 5 blocks * 10

    def test_within_range(self) -> None:
        idx = block_bootstrap_indices(100, block_size=10, n_blocks=10)
        assert idx.min() >= 0
        assert idx.max() < 100

    def test_block_size_clamping(self) -> None:
        """block_size > n_samples should be clamped."""
        idx = block_bootstrap_indices(5, block_size=20, n_blocks=2)
        assert idx.max() < 5

    def test_zero_block_size(self) -> None:
        """block_size=0 should be clamped to 1."""
        idx = block_bootstrap_indices(10, block_size=0, n_blocks=3)
        assert len(idx) == 3
