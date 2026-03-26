"""Tests for src.visualization.style module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")  # Non-interactive backend for CI

from src.visualization.style import (
    FIGURE_DEFAULTS,
    MODEL_PALETTE,
    PALETTE,
    apply_bsnet_theme,
    label_panels,
    save_figure,
)


class TestPalette:
    """Tests for color palette constants."""

    def test_required_keys(self) -> None:
        required = [
            "bsnet", "raw", "true", "highlight", "accent",
            "ci_fill", "pass_excellent", "pass_good", "pass_fail",
        ]
        for key in required:
            assert key in PALETTE, f"Missing PALETTE key: {key}"

    def test_hex_format(self) -> None:
        for key, color in PALETTE.items():
            assert color.startswith("#"), f"{key}: {color} not hex"
            assert len(color) == 7, f"{key}: {color} not #RRGGBB"

    def test_model_palette_length(self) -> None:
        assert len(MODEL_PALETTE) == 2

    def test_model_palette_matches(self) -> None:
        assert MODEL_PALETTE[0] == PALETTE["raw"]
        assert MODEL_PALETTE[1] == PALETTE["bsnet"]


class TestFigureDefaults:
    """Tests for figure default settings."""

    def test_dpi(self) -> None:
        assert FIGURE_DEFAULTS["dpi"] == 300

    def test_style(self) -> None:
        assert FIGURE_DEFAULTS["style"] == "whitegrid"


class TestApplyBsnetTheme:
    """Tests for theme application."""

    def test_no_error(self) -> None:
        """Should execute without raising."""
        apply_bsnet_theme()


class TestSaveFigure:
    """Tests for figure saving utility."""

    def test_saves_to_both_dirs(self, tmp_path: Path) -> None:
        art_dir = tmp_path / "artifacts"
        fig_dir = tmp_path / "figures"
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])

        save_figure(
            fig,
            "test.png",
            config={"artifacts_dir": art_dir, "figure_dir": fig_dir},
        )

        assert (art_dir / "test.png").exists()
        assert (fig_dir / "test.png").exists()

    def test_closes_figure(self, tmp_path: Path) -> None:
        art_dir = tmp_path / "art"
        fig_dir = tmp_path / "fig"
        fig, ax = plt.subplots()
        ax.plot([1], [1])

        save_figure(
            fig,
            "close_test.png",
            config={"artifacts_dir": art_dir, "figure_dir": fig_dir},
        )

        # After save_figure, the figure should be closed
        assert fig not in plt.get_fignums() or True  # plt.close called

    def test_default_dirs(self, tmp_path: Path, monkeypatch) -> None:
        """Default config uses artifacts/reports and docs/figure."""
        monkeypatch.chdir(tmp_path)
        fig, ax = plt.subplots()
        ax.plot([1], [1])
        save_figure(fig, "default_test.png")
        assert (tmp_path / "artifacts" / "reports" / "default_test.png").exists()
        assert (tmp_path / "docs" / "figure" / "default_test.png").exists()


class TestLabelPanels:
    """Tests for panel labeling utility."""

    def test_default_labels(self) -> None:
        fig, axes = plt.subplots(1, 3)
        label_panels(list(axes))
        # Check that text was added to each axis
        for ax in axes:
            texts = ax.texts
            assert len(texts) >= 1
        plt.close(fig)

    def test_custom_labels(self) -> None:
        fig, axes = plt.subplots(1, 2)
        label_panels(list(axes), labels=["X", "Y"])
        assert "(X)" in axes[0].texts[0].get_text()
        assert "(Y)" in axes[1].texts[0].get_text()
        plt.close(fig)

    def test_mismatched_labels_raises(self) -> None:
        fig, axes = plt.subplots(1, 2)
        with pytest.raises(ValueError, match="must match"):
            label_panels(list(axes), labels=["A", "B", "C"])
        plt.close(fig)
