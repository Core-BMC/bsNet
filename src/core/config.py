"""
Configuration module for BS-NET project.

Centralizes all magic numbers and configuration parameters used across
the codebase, including simulation, bootstrap, and graph analysis settings.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

NETWORK_NAMES: list[str] = [
    "Visual",
    "Somatomotor",
    "Dorsal Attn",
    "Ventral Attn",
    "Limbic",
    "Control",
    "Default Mode",
]


@dataclass(frozen=True)
class BSNetConfig:
    """
    Frozen configuration dataclass for BS-NET parameters.

    Centralizes all magic numbers and configuration constants used throughout
    the codebase, including simulation parameters, bootstrap settings, and
    output paths.

    Attributes:
        n_rois: Number of regions of interest (Schaefer parcellation).
        tr: Repetition time in seconds.
        short_duration_sec: Duration of short observation window in seconds.
        target_duration_min: Duration of target/long observation window in minutes.
        noise_level: Standard deviation of added Gaussian noise.
        ar1: AR(1) autocorrelation coefficient for synthetic time series.
        n_networks: Number of canonical brain networks (Yeo parcellation).
        n_bootstraps: Number of bootstrap iterations.
        reliability_coeff: Within-session scanner measurement reliability.
            Default 0.98 based on Friedman et al. (2008) who reported
            within-session ICC > 0.95 for rsfMRI FC metrics.
            DOI: 10.1016/j.neuroimage.2008.02.005
        empirical_prior: Tuple of (mean, variance) for Bayesian empirical prior.
        seed: Random seed for reproducibility.
        fc_density: Target edge density for thresholded functional connectivity graphs.
        artifacts_dir: Directory path for output artifacts and reports.
        figure_dir: Directory path for figure outputs.
    """

    # Simulation parameters
    n_rois: int = 400
    tr: float = 1.0
    short_duration_sec: int = 120
    target_duration_min: int = 15
    noise_level: float = 0.25
    ar1: float = 0.6
    n_networks: int = 7

    # Bootstrap parameters
    n_bootstraps: int = 100
    # Within-session scanner measurement reliability (Friedman et al. 2008,
    # DOI: 10.1016/j.neuroimage.2008.02.005, within-session ICC > 0.95).
    # This represents the measurement precision of the fMRI scanner itself,
    # distinct from cross-session test-retest reliability (Noble et al. 2019,
    # ICC ≈ 0.29). Sensitivity analysis (Track A) confirms robustness across
    # the range [0.70, 0.99].
    reliability_coeff: float = 0.98
    empirical_prior: tuple[float, float] = (0.25, 0.05)
    seed: int = 42

    # Graph analysis
    fc_density: float = 0.15

    # Paths
    artifacts_dir: str = "artifacts/reports"
    figure_dir: str = "docs/figure"

    @property
    def short_samples(self) -> int:
        """
        Compute number of samples in short observation window.

        Returns:
            int: short_duration_sec / tr
        """
        return int(self.short_duration_sec / self.tr)

    @property
    def target_samples(self) -> int:
        """
        Compute number of samples in target observation window.

        Returns:
            int: target_duration_min * 60 / tr
        """
        return int(self.target_duration_min * 60 / self.tr)

    @property
    def k_factor(self) -> float:
        """
        Compute scaling factor (k) for Spearman-Brown correction.

        Represents the ratio of long to short observation durations.

        Returns:
            float: target_samples / short_samples
        """
        return self.target_samples / self.short_samples

    def create_output_dirs(self) -> None:
        """
        Create output directories for artifacts and figures.

        Ensures that both artifacts_dir and figure_dir exist, creating
        parent directories as needed.

        Raises:
            OSError: If directory creation fails.
        """
        artifacts_path = Path(self.artifacts_dir)
        figure_path = Path(self.figure_dir)

        try:
            artifacts_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created artifacts directory: {artifacts_path}")
        except OSError as e:
            logger.error(f"Failed to create artifacts directory: {e}")
            raise

        try:
            figure_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created figure directory: {figure_path}")
        except OSError as e:
            logger.error(f"Failed to create figure directory: {e}")
            raise
