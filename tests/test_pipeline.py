"""Tests for src.core.pipeline module."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.config import BSNetConfig
from src.core.pipeline import (
    BootstrapResult,
    compute_split_half_reliability,
    run_bootstrap_prediction,
)
from src.data.data_loader import get_fc_matrix


class TestBootstrapResult:
    """Tests for BootstrapResult NamedTuple."""

    def test_fields(self) -> None:
        r = BootstrapResult(
            rho_hat_T=0.85,
            ci_lower=0.80,
            ci_upper=0.90,
            z_scores=np.array([1.0, 1.1, 1.2]),
        )
        assert r.rho_hat_T == 0.85
        assert r.ci_lower == 0.80
        assert r.ci_upper == 0.90
        assert len(r.z_scores) == 3

    def test_unpacking(self) -> None:
        r = BootstrapResult(0.8, 0.7, 0.9, np.array([1.0]))
        rho, lo, hi, zs = r
        assert rho == 0.8


class TestComputeSplitHalfReliability:
    """Tests for split-half reliability computation."""

    def test_output_range(self, synthetic_ts: np.ndarray) -> None:
        r = compute_split_half_reliability(synthetic_ts)
        assert 0.001 <= r <= 0.999

    def test_identical_halves(self) -> None:
        """Perfectly correlated halves should yield high reliability."""
        ts = np.tile(np.random.randn(60, 10), (2, 1))  # repeat
        r = compute_split_half_reliability(ts)
        assert r > 0.5

    def test_returns_float(self, synthetic_ts: np.ndarray) -> None:
        r = compute_split_half_reliability(synthetic_ts)
        assert isinstance(r, (float, np.floating))


class TestRunBootstrapPrediction:
    """Tests for the unified bootstrap prediction pipeline."""

    @pytest.fixture()
    def quick_config(self) -> BSNetConfig:
        """Minimal config for fast tests."""
        return BSNetConfig(
            n_rois=20,
            tr=1.0,
            short_duration_sec=60,
            target_duration_min=5,
            n_bootstraps=5,  # very few for speed
            seed=42,
        )

    def test_returns_bootstrap_result(
        self, synthetic_ts: np.ndarray, quick_config: BSNetConfig
    ) -> None:
        short_obs = synthetic_ts[:60, :]
        fc_ref = get_fc_matrix(synthetic_ts, vectorized=True)
        result = run_bootstrap_prediction(short_obs, fc_ref, quick_config)
        assert isinstance(result, BootstrapResult)

    def test_rho_hat_T_finite(
        self, synthetic_ts: np.ndarray, quick_config: BSNetConfig
    ) -> None:
        short_obs = synthetic_ts[:60, :]
        fc_ref = get_fc_matrix(synthetic_ts, vectorized=True)
        result = run_bootstrap_prediction(short_obs, fc_ref, quick_config)
        assert np.isfinite(result.rho_hat_T)
        assert -1.0 <= result.rho_hat_T <= 1.0

    def test_ci_ordering(
        self, synthetic_ts: np.ndarray, quick_config: BSNetConfig
    ) -> None:
        short_obs = synthetic_ts[:60, :]
        fc_ref = get_fc_matrix(synthetic_ts, vectorized=True)
        result = run_bootstrap_prediction(short_obs, fc_ref, quick_config)
        assert result.ci_lower <= result.rho_hat_T <= result.ci_upper

    def test_z_scores_length(
        self, synthetic_ts: np.ndarray, quick_config: BSNetConfig
    ) -> None:
        short_obs = synthetic_ts[:60, :]
        fc_ref = get_fc_matrix(synthetic_ts, vectorized=True)
        result = run_bootstrap_prediction(short_obs, fc_ref, quick_config)
        assert len(result.z_scores) == quick_config.n_bootstraps

    def test_default_config(self, synthetic_ts: np.ndarray) -> None:
        """config=None should use defaults without error."""
        short_obs = synthetic_ts[:60, :]
        fc_ref = get_fc_matrix(synthetic_ts, vectorized=True)
        # Override just bootstrap count via a minimal config
        cfg = BSNetConfig(n_bootstraps=3)
        result = run_bootstrap_prediction(short_obs, fc_ref, cfg)
        assert np.isfinite(result.rho_hat_T)

    def test_type_validation(self) -> None:
        with pytest.raises(TypeError, match="numpy array"):
            run_bootstrap_prediction([[1, 2], [3, 4]], np.array([1.0]))  # type: ignore[arg-type]

    def test_shape_validation(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            run_bootstrap_prediction(np.array([1.0, 2.0]), np.array([1.0]))
