#!/usr/bin/env python3
"""Patch Diffusion-TS for BS-NET reliability-weighted imputation guidance.

Applies minimal modifications to two files:
  1. engine/solver.py — pass roi_weights through restore() → model_kwargs
  2. Models/interpretable_diffusion/gaussian_diffusion.py — weight infill_loss in langevin_fn

Usage:
  cd external/Diffusion-TS
  python3 ../../src/scripts/patch_diffusion_ts_solver.py          # dry-run (analyze only)
  python3 ../../src/scripts/patch_diffusion_ts_solver.py --apply  # apply patches

What changes:
  In langevin_fn(), the observation consistency loss:
    BEFORE: infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
    AFTER:  weighted by roi_weights (per-ROI reliability from BS-NET)

  This is test-time only — no model architecture or training changes.

How roi_weights works:
  - Shape: [B, 1, N_ROI], broadcast over time dimension
  - High weight → ROI is reliable → strong observation constraint
  - Low weight → ROI is noisy → relax constraint, let diffusion generate freely
  - None → standard uniform weighting (backward compatible)
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


def backup_file(filepath: Path) -> Path:
    """Create timestamped backup."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = filepath.parent / ".backup"
    backup_dir.mkdir(exist_ok=True)
    backup_path = backup_dir / f"{filepath.stem}_{timestamp}{filepath.suffix}"
    shutil.copy2(filepath, backup_path)
    return backup_path


# ======================================================================
# Patch 1: engine/solver.py — pass roi_weights through restore()
# ======================================================================

def patch_solver(solver_path: Path, dry_run: bool = True) -> bool:
    """Patch engine/solver.py to pass roi_weights via model_kwargs.

    Changes restore() to accept roi_weights and inject into model_kwargs.
    """
    content = solver_path.read_text()

    if "roi_weights" in content:
        print(f"  [SKIP] {solver_path} — already patched")
        return False

    # --- Patch 1a: Add roi_weights parameter to restore() ---
    old_sig = "def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):"
    new_sig = "def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50, roi_weights=None):"

    if old_sig not in content:
        print(f"  [ERROR] restore() signature not found in {solver_path}")
        print(f"  Expected: {old_sig}")
        return False

    modified = content.replace(old_sig, new_sig, 1)

    # --- Patch 1b: Pass roi_weights into model_kwargs ---
    old_kwargs = "model_kwargs['learning_rate'] = stepsize"
    new_kwargs = (
        "model_kwargs['learning_rate'] = stepsize\n"
        "        model_kwargs['roi_weights'] = roi_weights  # BS-NET reliability weights"
    )

    if old_kwargs not in modified:
        print(f"  [ERROR] model_kwargs setup line not found")
        return False

    modified = modified.replace(old_kwargs, new_kwargs, 1)

    if dry_run:
        print(f"  [DRY RUN] Would patch {solver_path}")
        print(f"    + roi_weights parameter in restore()")
        print(f"    + roi_weights passed to model_kwargs")
        return True

    backup = backup_file(solver_path)
    print(f"  [BACKUP] {backup}")
    solver_path.write_text(modified)
    print(f"  [PATCHED] {solver_path}")
    return True


# ======================================================================
# Patch 2: gaussian_diffusion.py — weight infill_loss in langevin_fn()
# ======================================================================

def patch_gaussian_diffusion(gd_path: Path, dry_run: bool = True) -> bool:
    """Patch langevin_fn() to apply reliability weighting to infill_loss.

    The infill_loss appears in two branches (sigma==0 and sigma>0).
    Both need identical weighting logic.

    Current code:
        infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
        infill_loss = infill_loss.mean(dim=0).sum()

    Patched code:
        # Compute element-wise squared error over full tensor, then apply weights
        if roi_weights is not None:
            sq_err = (x_start - tgt_embs) ** 2  # [B, seq_len, N_ROI]
            weighted_err = roi_weights * sq_err  # roi_weights: [B, 1, N_ROI]
            infill_loss = weighted_err[partial_mask]
        else:
            infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
        infill_loss = infill_loss.mean(dim=0).sum()
    """
    content = gd_path.read_text()

    if "roi_weights" in content:
        print(f"  [SKIP] {gd_path} — already patched")
        return False

    # --- Patch 2a: Add roi_weights to langevin_fn signature ---
    old_langevin_sig = (
        "    def langevin_fn(\n"
        "        self,\n"
        "        coef,\n"
        "        partial_mask,\n"
        "        tgt_embs,\n"
        "        learning_rate,\n"
        "        sample,\n"
        "        mean,\n"
        "        sigma,\n"
        "        t,\n"
        "        coef_=0.\n"
        "    ):"
    )
    new_langevin_sig = (
        "    def langevin_fn(\n"
        "        self,\n"
        "        coef,\n"
        "        partial_mask,\n"
        "        tgt_embs,\n"
        "        learning_rate,\n"
        "        sample,\n"
        "        mean,\n"
        "        sigma,\n"
        "        t,\n"
        "        coef_=0.,\n"
        "        roi_weights=None,\n"
        "    ):"
    )

    if old_langevin_sig not in content:
        print(f"  [ERROR] langevin_fn signature not found in {gd_path}")
        print(f"  Try manual patching — see docstring for exact changes.")
        return False

    modified = content.replace(old_langevin_sig, new_langevin_sig, 1)

    # --- Patch 2b: Weight infill_loss in sigma==0 branch ---
    # Original (inside `if sigma.mean() == 0:` block):
    old_infill_sigma0 = (
        "                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2\n"
        "                    infill_loss = infill_loss.mean(dim=0).sum()"
    )
    new_infill_sigma0 = (
        "                    if roi_weights is not None:\n"
        "                        _sq_err = (x_start - tgt_embs) ** 2\n"
        "                        infill_loss = (roi_weights * _sq_err)[partial_mask]\n"
        "                    else:\n"
        "                        infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2\n"
        "                    infill_loss = infill_loss.mean(dim=0).sum()"
    )

    if old_infill_sigma0 not in modified:
        print(f"  [WARN] sigma==0 branch infill_loss not found (may have different indent)")
        # Try without leading spaces
    else:
        modified = modified.replace(old_infill_sigma0, new_infill_sigma0, 1)

    # --- Patch 2c: Weight infill_loss in else (sigma>0) branch ---
    old_infill_sigma_pos = (
        "                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2\n"
        "                    infill_loss = (infill_loss/sigma.mean()).mean(dim=0).sum()"
    )
    new_infill_sigma_pos = (
        "                    if roi_weights is not None:\n"
        "                        _sq_err = (x_start - tgt_embs) ** 2\n"
        "                        infill_loss = (roi_weights * _sq_err)[partial_mask]\n"
        "                    else:\n"
        "                        infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2\n"
        "                    infill_loss = (infill_loss/sigma.mean()).mean(dim=0).sum()"
    )

    if old_infill_sigma_pos not in modified:
        print(f"  [WARN] sigma>0 branch infill_loss not found")
    else:
        modified = modified.replace(old_infill_sigma_pos, new_infill_sigma_pos, 1)

    if dry_run:
        print(f"  [DRY RUN] Would patch {gd_path}")
        print(f"    + roi_weights parameter in langevin_fn()")
        print(f"    + Weighted infill_loss in sigma==0 branch")
        print(f"    + Weighted infill_loss in sigma>0 branch")
        return True

    backup = backup_file(gd_path)
    print(f"  [BACKUP] {backup}")
    gd_path.write_text(modified)
    print(f"  [PATCHED] {gd_path}")
    return True


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch Diffusion-TS for BS-NET reliability-weighted guidance"
    )
    parser.add_argument(
        "--base-dir", type=str, default=".",
        help="Diffusion-TS base directory (default: current dir)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Apply patches (default: dry-run analysis only)",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    solver_path = base / "engine" / "solver.py"
    gd_path = base / "Models" / "interpretable_diffusion" / "gaussian_diffusion.py"

    print("=" * 60)
    print("BS-NET Reliability-Weighted Guidance Patch")
    print("=" * 60)

    # Verify files exist
    for path, name in [(solver_path, "solver.py"), (gd_path, "gaussian_diffusion.py")]:
        if not path.exists():
            print(f"\n[ERROR] {name} not found at {path}")
            print(f"  Run from Diffusion-TS root, or use --base-dir")
            return

    print(f"\nBase directory: {base.resolve()}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")

    # Apply patches
    print(f"\n--- Patch 1: engine/solver.py ---")
    patch_solver(solver_path, dry_run=not args.apply)

    print(f"\n--- Patch 2: gaussian_diffusion.py ---")
    patch_gaussian_diffusion(gd_path, dry_run=not args.apply)

    if not args.apply:
        print(f"\n{'='*60}")
        print("Dry run complete. Review above, then run with --apply")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("Patches applied. Backups in .backup/ directories.")
        print("")
        print("Usage in run_signal_recovery.py:")
        print("  # Condition A (naive): roi_weights=None (default)")
        print("  # Condition B (guided): roi_weights=[B, 1, N_ROI] tensor")
        print("")
        print("  In solver.restore():")
        print("    samples, reals, masks = solver.restore(")
        print("        dataloader, shape=(...), roi_weights=weights_tensor)")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
