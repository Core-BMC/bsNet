"""Centralized visualization styling and theme management for BS-NET.

This module provides unified color palettes, figure configuration, and helper
functions for consistent matplotlib/seaborn visualization across all plots.

Style Reference (Figure 1 standard):
  - Title: 15pt bold, pad=10
  - Axis labels: 13pt
  - Legend: 10-11pt
  - Line width: 2.5-3.0 (main), 1.5 (secondary)
  - Marker size: 8 (main), 6 (secondary)
  - Panel labels: "A. Title" format inside set_title()
  - Layout: tight_layout(pad=3.0)
  - Save: 300 DPI, bbox_inches="tight"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configure module logger
_logger = logging.getLogger(__name__)

# ============================================================================
# COLOR PALETTES
# ============================================================================

PALETTE: dict[str, str] = {
    "bsnet": "#4A90E2",  # BS-NET predictions (blue)
    "raw": "#E27396",  # Raw 2-min FC (pink/red)
    "true": "#2c7bb6",  # Reference FC 15-min (dark blue)
    "highlight": "#d7191c",  # Emphasis/threshold markers (red)
    "accent": "#fdae61",  # Secondary accent (amber)
    "ci_fill": "#abd9e9",  # Confidence interval fill (light blue)
    "pass_excellent": "#2ecc71",  # Excellent category (green)
    "pass_good": "#f1c40f",  # Good category (yellow)
    "pass_fail": "#e74c3c",  # Failed category (red)
    # Group coloring for clinical datasets
    "control": "#2c7bb6",  # Control group (dark blue, same as "true")
    "adhd": "#d7191c",  # ADHD/patient group (red, same as "highlight")
    # Atlas comparison
    "cc200": "#4A90E2",  # CC200 atlas (blue, same as "bsnet")
    "cc400": "#fdae61",  # CC400 atlas (amber, same as "accent")
    # Correction method comparison
    "original": "#E27396",  # Original correction (pink)
    "fisher_z": "#4A90E2",  # Fisher z correction (blue)
    "partial": "#2ecc71",  # Partial correction (green)
    "soft_clamp": "#f1c40f",  # Soft clamp correction (yellow)
}
"""Dict[str, str]: Centralized color definitions for all visualization types."""

MODEL_PALETTE: list[str] = [PALETTE["raw"], PALETTE["bsnet"]]
"""List[str]: Color palette for comparing raw vs. BS-NET models in plots."""

GROUP_PALETTE: dict[str, str] = {
    "control": PALETTE["control"],
    "adhd": PALETTE["adhd"],
    "unknown": "#95a5a6",
}
"""Dict[str, str]: Color mapping for clinical groups."""

ATLAS_PALETTE: dict[str, str] = {
    "cc200": PALETTE["cc200"],
    "cc400": PALETTE["cc400"],
}
"""Dict[str, str]: Color mapping for atlas comparisons."""

CORRECTION_PALETTE: dict[str, str] = {
    "original": PALETTE["original"],
    "fisher_z": PALETTE["fisher_z"],
    "partial": PALETTE["partial"],
    "soft_clamp": PALETTE["soft_clamp"],
}
"""Dict[str, str]: Color mapping for attenuation correction methods."""

CONDITION_PALETTE: dict[str, str] = {
    "reference": "#95a5a6",  # Reference FC (gray)
    "raw": "#fdae61",        # Raw FC (amber)
    "bsnet": "#4A90E2",      # bs-Net (blue)
}
"""Dict[str, str]: 3-color scheme for Fig 3–7 (Gray/Amber/Blue)."""

# ============================================================================
# ATLAS METADATA  (Figure 1 multi-atlas standard)
# ============================================================================

ATLAS_META: dict[str, dict] = {
    "schaefer200":    {"label": "Schaefer 200",   "color": "#4A90E2", "ls": "-",  "marker": "s", "lw": 2.5},
    "schaefer400":    {"label": "Schaefer 400",   "color": "#1A5FAC", "ls": "--", "marker": "^", "lw": 1.8},
    "cc200":          {"label": "CC 200",          "color": "#F4A261", "ls": "-",  "marker": "o", "lw": 1.8},
    "cc400":          {"label": "CC 400",          "color": "#C05C22", "ls": "--", "marker": "D", "lw": 1.8},
    "aal":            {"label": "AAL",             "color": "#52B788", "ls": "-",  "marker": "P", "lw": 1.8},
    "harvard_oxford": {"label": "Harvard-Oxford",  "color": "#9B72CF", "ls": "--", "marker": "X", "lw": 1.8},
}
"""Dict[str, dict]: Per-atlas color, linestyle, marker, linewidth for multi-atlas panels."""

# ============================================================================
# TYPOGRAPHY CONSTANTS (Figure 1 standard)
# ============================================================================

FONT: dict[str, int | float | str] = {
    "title": 18,            # Panel label (A, B, C …) — bold
    "title_weight": "bold",
    "title_pad": 10,
    "axis_label": 10.5,     # x/y axis labels
    "legend": 9.5,          # Legend entries
    "legend_small": 8.5,    # Small legend (multi-atlas ncol=2)
    "tick": 9.5,            # Tick labels
    "annotation": 9.5,      # Inline annotations / text boxes
    "suptitle": 14,         # Figure suptitle
}
"""Dict: Standardized font sizes (Figure 1 standard, updated from Fig1 ds007535)."""

# Convenience kwargs dicts — pass directly as **FONT_PANEL, **FONT_AXIS
FONT_PANEL: dict[str, int | str] = {
    "fontsize": FONT["title"],
    "fontweight": FONT["title_weight"],
}
"""dict: kwargs for panel-label set_title() calls."""

FONT_AXIS: dict[str, float] = {
    "fontsize": FONT["axis_label"],
}
"""dict: kwargs for set_xlabel() / set_ylabel() calls."""

FONT_TICK: float = FONT["tick"]
"""float: labelsize for ax.tick_params()."""

# ============================================================================
# LINE / MARKER CONSTANTS
# ============================================================================

LINE: dict[str, float] = {
    "main": 3.0,
    "secondary": 2.5,
    "thin": 1.5,
    "reference": 2.0,
    "error": 1.2,
    "individual": 0.7,   # per-subject / thin trajectory lines
}
"""Dict[str, float]: Standardized line widths."""

MARKER: dict[str, int] = {
    "main": 8,
    "secondary": 6,
    "small": 4,
    "scatter": 50,
    "scatter_small": 30,
}
"""Dict[str, int]: Standardized marker sizes."""

# ============================================================================
# BAR CHART STYLE  (Raw = solid, bs-Net = diagonal stripe)
# ============================================================================

BAR_STYLE: dict[str, dict] = {
    "raw": {
        "alpha": 0.85,
        "edgecolor": "none",
        "hatch": None,
    },
    "bsnet": {
        "alpha": 0.75,
        "edgecolor": "white",
        "hatch": "////",
        "linewidth": 0.4,
    },
}
"""Dict: Bar plot style — Raw FC (solid Amber) vs bs-Net (diagonal stripe, atlas color).

Usage example::

    ax.bar(x, raw_vals,   color=CONDITION_PALETTE["raw"],   **BAR_STYLE["raw"])
    ax.bar(x, bsnet_vals, color=CONDITION_PALETTE["bsnet"], **BAR_STYLE["bsnet"])
"""

# ============================================================================
# FIGURE SIZE PRESETS
# ============================================================================

FIGSIZE: dict[str, tuple[float, float]] = {
    "single":   (8,  6),
    "wide":     (14, 7),
    "1x2":      (14, 6),
    "2x2":      (16, 10),
    "2x2_tall": (16, 12),
    "2x3":      (18, 11),   # Figure 1 rows A–C / D–F
    "3x3":      (18, 15),   # Figure 1 full (A–F + G row)
}
"""Dict[str, tuple]: Standardized figure sizes."""

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


# ============================================================================
# AXIS STYLING HELPERS
# ============================================================================


def style_axis(
    ax: plt.Axes,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    legend_loc: str = "best",
    legend_fontsize: int | None = None,
) -> None:
    """Apply Figure 1 standard styling to a single axis.

    Args:
        ax: Matplotlib Axes object.
        title: Panel title (e.g., "A. Scatter Plot"). Include panel letter.
        xlabel: X-axis label text.
        ylabel: Y-axis label text.
        legend_loc: Legend location string.
        legend_fontsize: Legend font size override. Defaults to FONT["legend"].
    """
    ax.set_title(
        title,
        fontweight=FONT["title_weight"],
        fontsize=FONT["title"],
        pad=FONT["title_pad"],
    )
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT["axis_label"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT["axis_label"])
    ax.tick_params(labelsize=FONT["tick"])
    if ax.get_legend_handles_labels()[1]:
        ax.legend(
            loc=legend_loc,
            fontsize=legend_fontsize or FONT["legend"],
        )


def add_identity_line(
    ax: plt.Axes,
    lims: tuple[float, float] | None = None,
    label: str = "identity",
) -> None:
    """Add a diagonal identity (y=x) reference line.

    Args:
        ax: Matplotlib Axes object.
        lims: (min, max) range for the line. If None, uses current axis limits.
        label: Legend label for the line.
    """
    if lims is None:
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        lims = (lo, hi)
    ax.plot(
        lims, lims, "k--", alpha=0.3, linewidth=LINE["thin"], label=label,
    )


def add_threshold_line(
    ax: plt.Axes,
    value: float,
    direction: str = "horizontal",
    color: str | None = None,
    linestyle: str = "--",
    label: str | None = None,
) -> None:
    """Add a horizontal or vertical threshold reference line.

    Args:
        ax: Matplotlib Axes object.
        value: Position of the threshold line.
        direction: "horizontal" or "vertical".
        color: Line color. Defaults to PALETTE["highlight"].
        linestyle: Line style string.
        label: Legend label.
    """
    color = color or PALETTE["highlight"]
    kwargs = dict(
        color=color, linestyle=linestyle,
        linewidth=LINE["reference"], label=label,
    )
    if direction == "horizontal":
        ax.axhline(y=value, **kwargs)
    else:
        ax.axvline(x=value, **kwargs)


def create_figure(
    layout: str = "2x2",
    **kwargs: Any,
) -> tuple[plt.Figure, np.ndarray | plt.Axes]:
    """Create a figure with standardized size and theme applied.

    Args:
        layout: One of FIGSIZE keys ("2x2", "1x2", "2x2_tall", "single", "wide").
        **kwargs: Additional kwargs passed to plt.subplots() (e.g., nrows, ncols).

    Returns:
        Tuple of (figure, axes).
    """
    apply_bsnet_theme()
    figsize = FIGSIZE.get(layout, FIGSIZE["2x2"])

    nrows = kwargs.pop("nrows", None)
    ncols = kwargs.pop("ncols", None)
    if nrows is None and ncols is None:
        # Infer from layout name
        layout_map = {
            "2x2": (2, 2), "1x2": (1, 2), "2x2_tall": (2, 2),
            "single": (1, 1), "wide": (1, 1),
        }
        nrows, ncols = layout_map.get(layout, (2, 2))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes
