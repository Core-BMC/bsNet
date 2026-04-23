#!/usr/bin/env python3
"""Patch Diffusion-TS solver.py for reliability-weighted guidance.

Applies minimal modification to engine/solver.py to support BS-NET
reliability weighting in reconstruction-guided imputation.

Usage:
  cd external/Diffusion-TS
  python3 ../../src/scripts/patch_diffusion_ts_solver.py

  # Or with explicit path:
  python3 ../../src/scripts/patch_diffusion_ts_solver.py --solver-path engine/solver.py

What it does:
  1. Finds the restore() method's observation loss computation
  2. Adds an optional `roi_weights` parameter
  3. Modifies loss: F.mse_loss(pred, obs) → (roi_weights * (pred - obs)**2).mean()

This is test-time only — no model architecture changes.
"""
from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path


# Expected patterns in Diffusion-TS solver.py
# These may vary by version — inspect before applying

PATCH_SPECS = [
    {
        "description": "Add roi_weights parameter to restore() method",
        "search_pattern": r"(def restore\(self[^)]*)\)",
        "replacement": r"\1, roi_weights=None)",
        "max_occurrences": 1,
    },
    {
        "description": "Add reliability-weighted observation loss",
        "comment": """
# This patch modifies the observation consistency loss in the restore() method.
# The exact code depends on the Diffusion-TS version.
#
# MANUAL PATCH GUIDE (if automatic patching fails):
#
# Find in engine/solver.py, inside restore() method, the line computing
# observation loss. It typically looks like one of:
#
#   loss_obs = F.mse_loss(x_hat[mask], x_obs[mask])
#   loss = ((x_hat - x_obs) ** 2 * mask).mean()
#   recon_loss = F.mse_loss(pred_obs, target_obs)
#
# Replace with:
#
#   if roi_weights is not None:
#       # BS-NET reliability weighting: reliable ROIs get stronger constraints
#       # roi_weights shape: [B, 1, N_ROI], broadcast over time dimension
#       loss_obs = (roi_weights * mask * (x_hat - x_obs) ** 2).mean()
#   else:
#       loss_obs = (mask * (x_hat - x_obs) ** 2).mean()
#
# The roi_weights tensor comes from BS-NET:
#   roi_weights[i] = normalized reliability of ROI i
#   High weight → this ROI's observed values are trustworthy → strong constraint
#   Low weight → this ROI is noisy → relax constraint, let diffusion generate freely
""",
    },
]


def find_solver(base_dir: Path) -> Path | None:
    """Find solver.py in Diffusion-TS directory."""
    candidates = [
        base_dir / "engine" / "solver.py",
        base_dir / "Engine" / "solver.py",
        base_dir / "solver.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def backup_file(filepath: Path) -> Path:
    """Create timestamped backup."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = filepath.parent / ".backup"
    backup_dir.mkdir(exist_ok=True)
    backup_path = backup_dir / f"{filepath.stem}_{timestamp}{filepath.suffix}"
    shutil.copy2(filepath, backup_path)
    return backup_path


def analyze_solver(solver_path: Path) -> dict:
    """Analyze solver.py to find patch targets."""
    content = solver_path.read_text()
    analysis = {
        "has_restore_method": "def restore" in content,
        "has_mse_loss": "mse_loss" in content,
        "has_mask_multiply": "* mask" in content or "*mask" in content,
        "restore_signature": None,
        "loss_lines": [],
    }

    # Find restore() signature
    match = re.search(r"(def restore\([^)]+\))", content)
    if match:
        analysis["restore_signature"] = match.group(1)

    # Find potential loss computation lines
    for i, line in enumerate(content.split("\n"), 1):
        if any(kw in line.lower() for kw in ["mse_loss", "loss_obs", "recon_loss"]):
            if "mask" in line.lower() or "obs" in line.lower():
                analysis["loss_lines"].append((i, line.strip()))

    return analysis


def apply_patch(solver_path: Path, dry_run: bool = True) -> bool:
    """Apply reliability-weighting patch to solver.py.

    Args:
        solver_path: path to engine/solver.py
        dry_run: if True, only analyze and report (don't modify)

    Returns:
        True if patch was applied (or would be applied in dry_run)
    """
    content = solver_path.read_text()
    analysis = analyze_solver(solver_path)

    print(f"\n{'='*60}")
    print(f"Analyzing: {solver_path}")
    print(f"{'='*60}")

    for key, val in analysis.items():
        if key != "loss_lines":
            print(f"  {key}: {val}")

    if analysis["loss_lines"]:
        print(f"\n  Candidate loss lines ({len(analysis['loss_lines'])}):")
        for lineno, line in analysis["loss_lines"]:
            print(f"    L{lineno}: {line}")

    if not analysis["has_restore_method"]:
        print("\n[ERROR] No restore() method found. Manual patching required.")
        print("  See MANUAL PATCH GUIDE in this script's PATCH_SPECS.")
        return False

    if dry_run:
        print(f"\n[DRY RUN] Would patch {solver_path}")
        print("  Run with --apply to apply patch.")
        return True

    # Backup
    backup = backup_file(solver_path)
    print(f"\n[BACKUP] {backup}")

    # Apply parameter addition
    modified = content
    if "roi_weights" not in modified:
        modified = re.sub(
            r"(def restore\(self[^)]*)\)",
            r"\1, roi_weights=None)",
            modified,
            count=1,
        )
        print("[PATCHED] Added roi_weights parameter to restore()")

    # Write
    if modified != content:
        solver_path.write_text(modified)
        print(f"\n[SAVED] {solver_path}")
        print("\n[IMPORTANT] Manual step remaining:")
        print("  Find the observation loss line in restore() and add weighting.")
        print("  See MANUAL PATCH GUIDE above for exact code.")
        return True
    else:
        print("\n[SKIP] No changes needed (already patched?)")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch Diffusion-TS for reliability weighting")
    parser.add_argument("--solver-path", type=str, default=None,
                        help="Path to engine/solver.py (auto-detected if not given)")
    parser.add_argument("--base-dir", type=str, default=".",
                        help="Diffusion-TS base directory (default: current dir)")
    parser.add_argument("--apply", action="store_true",
                        help="Actually apply the patch (default: dry-run)")
    args = parser.parse_args()

    if args.solver_path:
        solver_path = Path(args.solver_path)
    else:
        solver_path = find_solver(Path(args.base_dir))

    if solver_path is None or not solver_path.exists():
        print(f"[ERROR] solver.py not found. Specify --solver-path or --base-dir.")
        print(f"  Searched in: {Path(args.base_dir).resolve()}/engine/solver.py")
        return

    # Print manual patch guide
    for spec in PATCH_SPECS:
        if "comment" in spec:
            print(spec["comment"])

    apply_patch(solver_path, dry_run=not args.apply)


if __name__ == "__main__":
    main()
