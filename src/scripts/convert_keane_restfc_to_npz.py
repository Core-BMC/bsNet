#!/usr/bin/env python3
"""Convert ds003404/ds005073 restFC .mat derivatives into unified NPZ.

This converter prepares a consistent FC dataset for downstream classification.
It does not run BS-NET correction because only precomputed FC matrices are
available in the downloaded derivatives.

Inputs (default):
  - data/ds003404/derivatives/restFCArray.mat       (HC)
  - data/ds005073/derivatives/restFCArray_BP.mat    (BP)
  - data/ds005073/derivatives/restFCArray_SZ.mat    (SZ)
  - corresponding participants.tsv files

Outputs:
  - data/ds005073/results/keane_restfc_combined.npz
  - data/ds005073/results/keane_restfc_metadata.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DS003404_DIR = Path("data/ds003404")
DS005073_DIR = Path("data/ds005073")
OUT_DIR = Path("data/ds005073/results")


def _read_tsv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = []
        for row in reader:
            pid = (row.get("participant_id") or "").strip()
            if not pid:
                continue
            rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    return rows


def _filter_has_rest(rows: list[dict], key: str) -> list[dict]:
    out = []
    for r in rows:
        v = (r.get(key) or "").strip()
        if v == "1":
            out.append(r)
    return out


def _load_mat_fc(path: Path) -> np.ndarray:
    """Load FC tensor from MAT file as (n_subjects, n_rois, n_rois)."""
    try:
        from scipy.io import loadmat
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "scipy is required to read .mat files. "
            "Install it in the active environment (pip install scipy).",
        ) from e

    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    candidates: list[np.ndarray] = []
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 3:
            candidates.append(v)

    if not candidates:
        raise RuntimeError(f"No 3D array found in MAT file: {path}")

    # Prefer arrays with square FC dimensions.
    best = None
    best_score = -1
    for arr in candidates:
        score = 0
        if arr.shape[0] == arr.shape[1]:
            score += 2
        if arr.shape[1] == arr.shape[2]:
            score += 2
        score += max(arr.shape)
        if score > best_score:
            best = arr
            best_score = score

    assert best is not None
    arr = np.asarray(best, dtype=np.float64)

    if arr.shape[0] == arr.shape[1]:
        # (roi, roi, n_sub)
        fc = np.transpose(arr, (2, 0, 1))
    elif arr.shape[1] == arr.shape[2]:
        # (n_sub, roi, roi)
        fc = arr
    else:
        raise RuntimeError(f"Cannot infer FC tensor orientation for {path}: shape={arr.shape}")

    if fc.shape[1] != fc.shape[2]:
        raise RuntimeError(f"Non-square FC matrix after conversion: {path}, shape={fc.shape}")
    return fc


def _as_float(s: str | None) -> float:
    if s is None:
        return float("nan")
    s = s.strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _normalize_sex(s: str | None) -> str:
    x = (s or "").strip().upper()
    if x in ("M", "MALE"):
        return "M"
    if x in ("F", "FEMALE"):
        return "F"
    return "U"


def _build_metadata(
    hc_rows: list[dict],
    bp_rows: list[dict],
    sz_rows: list[dict],
) -> list[dict]:
    meta: list[dict] = []

    for i, r in enumerate(hc_rows):
        meta.append({
            "sample_id": f"HC_{i+1:03d}",
            "participant_id": r.get("participant_id", ""),
            "source_dataset": "ds003404",
            "cohort": "HC",
            "label_binary": 0,
            "label_threeclass": 0,
            "age": _as_float(r.get("Age")),
            "sex": _normalize_sex(r.get("Gender")),
        })

    for i, r in enumerate(bp_rows):
        meta.append({
            "sample_id": f"BP_{i+1:03d}",
            "participant_id": r.get("participant_id", ""),
            "source_dataset": "ds005073",
            "cohort": "BP",
            "label_binary": 1,
            "label_threeclass": 1,
            "age": _as_float(r.get("age")),
            "sex": _normalize_sex(r.get("gender")),
        })

    for i, r in enumerate(sz_rows):
        meta.append({
            "sample_id": f"SZ_{i+1:03d}",
            "participant_id": r.get("participant_id", ""),
            "source_dataset": "ds005073",
            "cohort": "SZ",
            "label_binary": 1,
            "label_threeclass": 2,
            "age": _as_float(r.get("age")),
            "sex": _normalize_sex(r.get("gender")),
        })

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Keane restFC derivatives to NPZ.")
    parser.add_argument(
        "--out-npz",
        type=Path,
        default=OUT_DIR / "keane_restfc_combined.npz",
        help="Output NPZ path.",
    )
    parser.add_argument(
        "--out-meta-csv",
        type=Path,
        default=OUT_DIR / "keane_restfc_metadata.csv",
        help="Output metadata CSV path.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    hc_tsv = DS003404_DIR / "participants.tsv"
    bp_sz_tsv = DS005073_DIR / "participants.tsv"
    hc_mat = DS003404_DIR / "derivatives" / "restFCArray.mat"
    bp_mat = DS005073_DIR / "derivatives" / "restFCArray_BP.mat"
    sz_mat = DS005073_DIR / "derivatives" / "restFCArray_SZ.mat"

    for p in (hc_tsv, bp_sz_tsv, hc_mat, bp_mat, sz_mat):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    # Participants
    hc_rows_all = _read_tsv(hc_tsv)
    bp_sz_rows_all = _read_tsv(bp_sz_tsv)
    hc_rows = _filter_has_rest(hc_rows_all, key="HasRestData")
    bp_rows = [r for r in bp_sz_rows_all if (r.get("groupID") or "").strip() == "1"]
    sz_rows = [r for r in bp_sz_rows_all if (r.get("groupID") or "").strip() == "2"]
    bp_rows = _filter_has_rest(bp_rows, key="hasrestdata")
    sz_rows = _filter_has_rest(sz_rows, key="hasrestdata")

    # FC tensors
    fc_hc = _load_mat_fc(hc_mat)
    fc_bp = _load_mat_fc(bp_mat)
    fc_sz = _load_mat_fc(sz_mat)

    logger.info(f"HC rest subjects (participants.tsv): {len(hc_rows)} | FC tensor: {fc_hc.shape[0]}")
    logger.info(f"BP rest subjects (participants.tsv): {len(bp_rows)} | FC tensor: {fc_bp.shape[0]}")
    logger.info(f"SZ rest subjects (participants.tsv): {len(sz_rows)} | FC tensor: {fc_sz.shape[0]}")

    if len(hc_rows) != fc_hc.shape[0]:
        raise RuntimeError(
            f"HC subject count mismatch: tsv={len(hc_rows)} vs FC={fc_hc.shape[0]}",
        )
    if len(bp_rows) != fc_bp.shape[0]:
        raise RuntimeError(
            f"BP subject count mismatch: tsv={len(bp_rows)} vs FC={fc_bp.shape[0]}",
        )
    if len(sz_rows) != fc_sz.shape[0]:
        raise RuntimeError(
            f"SZ subject count mismatch: tsv={len(sz_rows)} vs FC={fc_sz.shape[0]}",
        )

    n_roi = fc_hc.shape[1]
    if fc_bp.shape[1] != n_roi or fc_sz.shape[1] != n_roi:
        raise RuntimeError(
            f"ROI mismatch across cohorts: HC={fc_hc.shape}, BP={fc_bp.shape}, SZ={fc_sz.shape}",
        )

    fc_all = np.concatenate([fc_hc, fc_bp, fc_sz], axis=0).astype(np.float32)
    cohort_codes = np.array(
        ([0] * fc_hc.shape[0]) + ([1] * fc_bp.shape[0]) + ([2] * fc_sz.shape[0]),
        dtype=np.int64,
    )  # 0=HC, 1=BP, 2=SZ
    y_binary = np.array([0 if c == 0 else 1 for c in cohort_codes], dtype=np.int64)  # HC vs patient

    meta_rows = _build_metadata(hc_rows, bp_rows, sz_rows)
    assert len(meta_rows) == fc_all.shape[0]

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        fc_all=fc_all,
        cohort_codes=cohort_codes,
        y_binary_hc_vs_psychosis=y_binary,
    )
    logger.info(f"Saved NPZ: {args.out_npz}")

    with open(args.out_meta_csv, "w", newline="") as f:
        fields = [
            "sample_id",
            "participant_id",
            "source_dataset",
            "cohort",
            "label_binary",
            "label_threeclass",
            "age",
            "sex",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(meta_rows)
    logger.info(f"Saved metadata CSV: {args.out_meta_csv}")

    logger.info(
        "Done. Combined restFC dataset: "
        f"N={fc_all.shape[0]} (HC={fc_hc.shape[0]}, BP={fc_bp.shape[0]}, SZ={fc_sz.shape[0]}), "
        f"ROI={n_roi}",
    )


if __name__ == "__main__":
    main()
