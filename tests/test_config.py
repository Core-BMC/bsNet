"""Tests for src.core.config module."""

from __future__ import annotations

import pytest

from src.core.config import NETWORK_NAMES, BSNetConfig


class TestBSNetConfig:
    """Tests for BSNetConfig dataclass."""

    def test_default_values(self) -> None:
        cfg = BSNetConfig()
        assert cfg.n_rois == 400
        assert cfg.tr == 1.0
        assert cfg.short_duration_sec == 120
        assert cfg.target_duration_min == 15
        assert cfg.n_bootstraps == 100
        assert cfg.reliability_coeff == 0.98
        assert cfg.seed == 42
        assert cfg.fc_density == 0.15

    def test_computed_short_samples(self) -> None:
        cfg = BSNetConfig(short_duration_sec=120, tr=1.0)
        assert cfg.short_samples == 120

        cfg2 = BSNetConfig(short_duration_sec=120, tr=2.0)
        assert cfg2.short_samples == 60

    def test_computed_target_samples(self) -> None:
        cfg = BSNetConfig(target_duration_min=15, tr=1.0)
        assert cfg.target_samples == 900

    def test_computed_k_factor(self) -> None:
        cfg = BSNetConfig(short_duration_sec=120, target_duration_min=15, tr=1.0)
        assert cfg.k_factor == pytest.approx(7.5)

    def test_frozen_immutable(self) -> None:
        cfg = BSNetConfig()
        with pytest.raises(AttributeError):
            cfg.n_rois = 100  # type: ignore[misc]

    def test_custom_override(self) -> None:
        cfg = BSNetConfig(n_rois=100, n_bootstraps=50, seed=0)
        assert cfg.n_rois == 100
        assert cfg.n_bootstraps == 50
        assert cfg.seed == 0

    def test_create_output_dirs(self, tmp_path) -> None:
        cfg = BSNetConfig(
            artifacts_dir=str(tmp_path / "art"),
            figure_dir=str(tmp_path / "fig"),
        )
        cfg.create_output_dirs()
        assert (tmp_path / "art").is_dir()
        assert (tmp_path / "fig").is_dir()


class TestNetworkNames:
    """Tests for NETWORK_NAMES constant."""

    def test_length(self) -> None:
        assert len(NETWORK_NAMES) == 7

    def test_contains_dmn(self) -> None:
        assert "Default Mode" in NETWORK_NAMES

    def test_contains_visual(self) -> None:
        assert "Visual" in NETWORK_NAMES

    def test_all_strings(self) -> None:
        assert all(isinstance(n, str) for n in NETWORK_NAMES)
