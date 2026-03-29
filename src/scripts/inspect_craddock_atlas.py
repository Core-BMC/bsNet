#!/usr/bin/env python3
"""Inspect Craddock 2012 atlas: list unique label count per volume.

Usage:
    python src/scripts/inspect_craddock_atlas.py

Output: volume_idx → n_unique_labels for all 43 volumes.
Identifies which volume corresponds to CC200 (~200 ROIs) and CC400 (~400 ROIs).
"""
import nibabel as nib
import numpy as np
from nilearn.datasets import fetch_atlas_craddock_2012

atlas = fetch_atlas_craddock_2012()
nii_path = getattr(atlas, "maps", None) or atlas.scorr_mean
img = nib.load(nii_path)
data = img.get_fdata()

print(f"Atlas shape: {data.shape}")
print(f"{'vol_idx':>8} {'n_labels':>10} {'max_label':>10}")
print("-" * 32)

target_200 = None
target_400 = None

for i in range(data.shape[-1]):
    vol = data[..., i]
    labels = np.unique(vol[vol > 0]).astype(int)
    n = len(labels)
    mx = int(labels.max()) if len(labels) > 0 else 0
    marker = ""
    if mx == 200:
        marker = "  ← CC200"
        target_200 = i
    elif mx == 400:
        marker = "  ← CC400"
        target_400 = i
    print(f"{i:>8} {n:>10} {mx:>10}{marker}")

print()
if target_200 is not None:
    print(f"CC200: volume_idx = {target_200}")
else:
    print("CC200: not found in ±5 range of 200")
if target_400 is not None:
    print(f"CC400: volume_idx = {target_400}")
else:
    print("CC400: not found in ±5 range of 400")
