"""Shared fixtures for BS-NET test suite."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── Mock heavy dependencies before any src import ──────────────────
# nilearn, nibabel, sklearn.covariance are not installed in CI/dry-run.
# We stub them so that `from src.data.data_loader import get_fc_matrix`
# resolves without error.

_nilearn_mock = MagicMock()
_nilearn_mock.datasets.fetch_atlas_schaefer_2018.return_value = MagicMock()

sys.modules.setdefault("nilearn", _nilearn_mock)
sys.modules.setdefault("nilearn.datasets", _nilearn_mock.datasets)
sys.modules.setdefault("nilearn.maskers", MagicMock())

# sklearn.covariance — provide a lightweight LedoitWolf stub
_sklearn_cov = MagicMock()


class _FakeLW:
    """Minimal Ledoit-Wolf stub for get_fc_matrix."""

    def fit(self, X: np.ndarray) -> _FakeLW:
        self.covariance_ = np.cov(X, rowvar=False)
        return self


_sklearn_cov.LedoitWolf = _FakeLW
sys.modules.setdefault("sklearn", MagicMock())
sys.modules.setdefault("sklearn.covariance", _sklearn_cov)
sys.modules.setdefault("sklearn.metrics", MagicMock())
sys.modules.setdefault("sklearn.metrics.cluster", MagicMock())

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ── Reusable fixtures ──────────────────────────────────────────────


@pytest.fixture()
def rng() -> np.random.Generator:
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture()
def synthetic_ts(rng: np.random.Generator) -> np.ndarray:
    """Synthetic time series (120 samples x 20 ROIs)."""
    return rng.standard_normal((120, 20))


@pytest.fixture()
def small_fc(synthetic_ts: np.ndarray) -> np.ndarray:
    """Small FC correlation matrix (20 x 20)."""
    corr = np.corrcoef(synthetic_ts.T)
    np.fill_diagonal(corr, 0)
    return corr
