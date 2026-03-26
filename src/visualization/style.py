"""Centralized visualization styling and theme management for BS-NET.

This module provides unified color palettes, figure configuration, and helper
functions for consistent matplotlib/seaborn visualization across all plots.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

# Configure module logger
_logger = logging.getLogger(__name__)

# ============================================================================
# COLOR PALETTES
# ============================================================================

PALETTE: dict[str, str] = {
    "bsnet": "#4A90E2",  # BS-NET predictions (blue)
    "raw": "#E27396",  # Raw 2-min FC (pink/red)
    "true": "#2c7bb6",  # Ground truth 15-min (dark blue)
    "highlight": "#d7191c",  # Emphasis/threshold markers (red)
    "accent": "#fdae61",  # Secondary accent (amber)
    "ci_fill": "#abd9e9",  # Confidence interval fill (light blue)
    "pass_excellent": "#2ecc71",  # Excellent category (green)
    "pass_good": "#f1c40f",  # Good category (yellow)
    "pass_fail": "#e74c3c",  # Failed category (red)
}
"""Dict[str, str]: Centralized color definitions for all visualization types."""

MODEL_PALETTE: list[str] = [PALETTE["raw"], PALETTE["bsnet"]]
"""List[str]: Color palette for comparing raw vs. BS-NET models in plots."""

# ============================================================================
# FIGURE DEFAULTS
# ============================================================================

FIGURE_DEFAULTS: dict[str, Any] = {
    "dpi": 300,
    "font_scale": 1.2,
    "style": "whitegrid",
    "context": "paper",
}
"""Dict[str, Any]: Default seaborn theme and matplotlib figure settings."""


# ============================================================================
# THEME FUNCTIONS
# ============================================================================


def apply_bsnet_theme() -> None:
    """Apply BS-NET standard theme to matplotlib/seaborn.

    Sets seaborn theme with consistent whitegrid style, paper context,
    and font scaling. Call once at the beginning of visualization scripts.

    Returns:
        None

    Example:
        >>> apply_bsnet_theme()
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 2, 3])
    """
    sns.set_theme(
        style=FIGURE_DEFAULTS["style"],
        context=FIGURE_DEFAULTS["context"],
        font_scale=FIGURE_DEFAULTS["font_scale"],
    )


# ============================================================================
# FIGURE SAVING
# ============================================================================


def save_figure(
    fig: plt.Figure,
    name: str,
    config: dict[str, Path] | None = None,
) -> None:
    """Save a matplotlib figure to multiple standard locations.

    Saves figure to both artifacts and documentation directories using
    consistent DPI and bounding box settings. Automatically closes the
    figure after saving to free memory.

    Args:
        fig: The matplotlib figure object to save.
        name: Filename for the figure (e.g., "Figure1_Combined.png").
        config: Optional dict with "artifacts_dir" and "figure_dir" Path
            objects. If None, uses defaults: "artifacts/reports" and
            "docs/figure".

    Returns:
        None

    Raises:
        OSError: If directories cannot be created or figure cannot be saved.

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 2, 3])
        >>> save_figure(fig, "example.png")
        >>> # Saves to "artifacts/reports/example.png" and "docs/figure/..."
    """
    if config is None:
        artifacts_dir = Path("artifacts/reports")
        figure_dir = Path("docs/figure")
    else:
        artifacts_dir = config.get("artifacts_dir", Path("artifacts/reports"))
        figure_dir = config.get("figure_dir", Path("docs/figure"))

    # Create directories if they don't exist
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    artifacts_path = artifacts_dir / name
    figure_path = figure_dir / name

    # Save to both locations
    fig.savefig(artifacts_path, dpi=FIGURE_DEFAULTS["dpi"], bbox_inches="tight")
    fig.savefig(figure_path, dpi=FIGURE_DEFAULTS["dpi"], bbox_inches="tight")

    # Log the saved paths
    _logger.info(f"Figure saved to {artifacts_path}")
    _logger.info(f"Figure saved to {figure_path}")

    # Close the figure to free memory
    plt.close(fig)


# ============================================================================
# PANEL LABELING
# ============================================================================


def label_panels(
    axes: list[plt.Axes],
    labels: list[str] | None = None,
) -> None:
    """Add panel labels (a), (b), (c)... to subplot axes.

    Automatically generates alphabetic labels starting from 'a' if no
    custom labels provided. Labels are positioned at the top-left corner
    of each axis with bold font.

    Args:
        axes: List of matplotlib Axes objects to label.
        labels: Optional list of custom label strings (e.g., ["A", "B"]).
            If None, generates labels "A", "B", "C", etc.

    Returns:
        None

    Raises:
        ValueError: If number of custom labels doesn't match number of axes.

    Example:
        >>> fig, axes = plt.subplots(2, 2)
        >>> axes_flat = axes.flatten()
        >>> label_panels(axes_flat)
        >>> # Adds "(A)", "(B)", "(C)", "(D)" to subplots
        >>> label_panels(axes_flat, labels=["Row 1A", "Row 1B", ...])
    """
    if labels is None:
        # Generate default labels A, B, C, ...
        labels = [chr(ord("A") + i) for i in range(len(axes))]
    elif len(labels) != len(axes):
        raise ValueError(
            f"Number of labels ({len(labels)}) must match "
            f"number of axes ({len(axes)})"
        )

    for ax, label in zip(axes, labels):
        ax.text(
            -0.1,
            1.1,
            f"({label})",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="right",
        )
