"""Download 100 HC subjects balanced across OpenNeuro datasets.

Reads hc_adult_index.csv (output of index_openneuro_hc.py) and downloads
anat + func data for ~100 subjects, balanced proportionally across datasets.

Usage:
    # Step 1: Index first
    python src/scripts/index_openneuro_hc.py

    # Step 2: Download 100 HC
    python src/scripts/download_hc_100.py
    python src/scripts/download_hc_100.py --n-subjects 100 --seed 42
    python src/scripts/download_hc_100.py --dry-run  # preview only
"""

import argparse
import csv
import logging
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def load_index(csv_path: Path) -> list[dict]:
    """Load HC index CSV."""
    records = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["age"] = float(row["age"]) if row["age"] and row["age"] != "None" else None
            records.append(row)
    return records


def balanced_sample(
    records: list[dict],
    n_total: int,
    seed: int,
) -> list[dict]:
    """Sample n_total subjects balanced proportionally across datasets.

    Each dataset contributes proportionally to its pool size.
    Minimum 1 subject per dataset (if available).

    Args:
        records: Full HC index.
        n_total: Target total subjects.
        seed: Random seed.

    Returns:
        Selected records.
    """
    rng = random.Random(seed)

    # Group by dataset
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_dataset[r["dataset_id"]].append(r)

    datasets = sorted(by_dataset.keys())
    total_pool = sum(len(v) for v in by_dataset.values())

    if total_pool <= n_total:
        logger.warning(
            f"Pool ({total_pool}) ≤ target ({n_total}). Using all subjects."
        )
        return records

    # Proportional allocation with minimum 1
    allocation: dict[str, int] = {}
    remaining = n_total

    # First pass: proportional
    for ds_id in datasets:
        pool = len(by_dataset[ds_id])
        n_alloc = max(1, math.floor(n_total * pool / total_pool))
        # Cap at pool size
        n_alloc = min(n_alloc, pool)
        allocation[ds_id] = n_alloc
        remaining -= n_alloc

    # Distribute remainder to largest pools
    if remaining > 0:
        pools_sorted = sorted(
            datasets, key=lambda d: len(by_dataset[d]), reverse=True
        )
        for ds_id in pools_sorted:
            if remaining <= 0:
                break
            available = len(by_dataset[ds_id]) - allocation[ds_id]
            add = min(remaining, available)
            allocation[ds_id] += add
            remaining -= add

    # Sample from each dataset
    selected: list[dict] = []
    for ds_id in datasets:
        pool = by_dataset[ds_id]
        n = allocation[ds_id]
        sampled = rng.sample(pool, min(n, len(pool)))
        selected.extend(sampled)

    return selected


def download_subject(
    dataset_id: str,
    participant_id: str,
    target_dir: Path,
    dry_run: bool = False,
) -> bool:
    """Download anat + func for one subject via openneuro-py.

    Args:
        dataset_id: OpenNeuro dataset ID.
        participant_id: Subject ID (e.g. 'sub-10159').
        target_dir: Root download directory.
        dry_run: If True, only print command.

    Returns:
        True if successful.
    """
    sub = participant_id
    if not sub.startswith("sub-"):
        sub = f"sub-{sub}"

    # openneuro-py 2026.3.0 behavior:
    #   openneuro-py download --dataset ds000030 --include sub-10280 --target-dir OUT
    #   → OUT/sub-10280/anat/  OUT/sub-10280/func/  etc.
    # We use --target-dir {target_dir}/{dataset_id} to get proper BIDS nesting.
    ds_dir = target_dir / dataset_id
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded (both anat + func with .nii.gz)
    anat_check = ds_dir / sub / "anat"
    func_check = ds_dir / sub / "func"
    if anat_check.exists() and func_check.exists():
        anat_files = list(anat_check.glob("*.nii.gz"))
        func_files = list(func_check.glob("*.nii.gz"))
        if anat_files and func_files:
            logger.info(f"  Already downloaded: {dataset_id}/{sub}")
            return True

    # Verified working pattern:
    #   openneuro-py download --dataset ds000030 --include sub-10280 --target-dir ds_dir
    cmd = [
        "openneuro-py", "download",
        "--dataset", dataset_id,
        "--include", sub,
        "--target-dir", str(ds_dir),
    ]

    if dry_run:
        logger.info(f"  [DRY RUN] {' '.join(cmd)}")
        return True

    logger.info(f"  Downloading {dataset_id}/{sub} ...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            logger.warning(f"  Download failed: {result.stderr[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning(f"  Download timeout: {dataset_id}/{sub}")
        return False
    except FileNotFoundError:
        logger.error("openneuro-py not found. Install: pip install openneuro-py")
        return False


def main() -> None:
    """Select and download 100 balanced HC subjects."""
    parser = argparse.ArgumentParser(
        description="Download 100 HC subjects balanced across datasets"
    )
    parser.add_argument(
        "--index-csv",
        type=str,
        default="data/hc_adult_index.csv",
        help="Input HC index CSV (from index_openneuro_hc.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/openneuro",
        help="Download target directory",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=100,
        help="Target number of subjects (default: 100)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview selection without downloading",
    )
    parser.add_argument(
        "--save-selection",
        type=str,
        default="data/hc_100_selection.csv",
        help="Save selected subjects to CSV",
    )
    args = parser.parse_args()

    index_path = Path(args.index_csv)
    if not index_path.exists():
        logger.error(f"Index CSV not found: {index_path}")
        logger.error("Run first: python src/scripts/index_openneuro_hc.py")
        sys.exit(1)

    records = load_index(index_path)
    logger.info(f"Loaded {len(records)} HC adults from {index_path}")

    # Balanced sampling
    selected = balanced_sample(records, args.n_subjects, args.seed)
    logger.info(f"\nSelected {len(selected)} subjects (seed={args.seed}):")

    # Summary
    from collections import Counter
    ds_counts = Counter(r["dataset_id"] for r in selected)
    ds_pool = Counter(r["dataset_id"] for r in records)
    for ds_id in sorted(ds_counts.keys()):
        logger.info(f"  {ds_id}: {ds_counts[ds_id]} / {ds_pool[ds_id]} available")

    # Save selection CSV
    sel_path = Path(args.save_selection)
    sel_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset_id", "participant_id", "age", "gender", "diagnosis"]
    with open(sel_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in selected:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    logger.info(f"Selection saved to: {sel_path}")

    if args.dry_run:
        logger.info("\n[DRY RUN] No downloads performed.")
        return

    # Download
    target_dir = Path(args.output_dir)
    success = 0
    failed = 0
    for i, rec in enumerate(selected, 1):
        logger.info(f"\n[{i}/{len(selected)}] {rec['dataset_id']} / {rec['participant_id']}")
        ok = download_subject(
            rec["dataset_id"],
            rec["participant_id"],
            target_dir,
            dry_run=False,
        )
        if ok:
            success += 1
        else:
            failed += 1

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Download complete: {success} success, {failed} failed")
    logger.info(f"Data location: {target_dir}")
    logger.info(f"Selection CSV: {sel_path}")


if __name__ == "__main__":
    main()
