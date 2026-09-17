## XCP-D Atlas 명칭 가이드 (v26.x NIfTI 모드)

XCP-D v26.x NIfTI 처리 시 `Schaefer200`/`Schaefer400` 이름은 인식되지 않음.
내장 아틀라스는 **Schaefer (피질) + Tian (피질하) 결합 4S 시리즈**로 제공됨.

| BS-NET 아틀라스 | XCP-D 4S 이름 | 피질 ROI | 피질하 ROI | 총 ROI |
|----------------|---------------|----------|------------|--------|
| schaefer200    | `4S256Parcels` | 200      | 56         | 256    |
| schaefer400    | `4S456Parcels` | 400      | 56         | 456    |
| schaefer100    | `4S156Parcels` | 100      | 56         | 156    |
| schaefer300    | `4S356Parcels` | 300      | 56         | 356    |

**XCP-D Docker 실행 옵션** (ds000243, NIfTI 모드):
```bash
docker run --rm \
  -v /path/to/fmriprep:/data:ro \
  -v /path/to/xcpd:/out \
  -v /path/to/work:/work \
  pennlinc/xcp_d:latest \
  /data /out participant \
  --mode linc --input-type fmriprep \
  --file-format nifti \
  -p 36P --fd-thresh 0.5 \
  --lower-bpf 0.01 --upper-bpf 0.1 \
  --smoothing 0 --combine-runs y \
  --atlases 4S256Parcels 4S456Parcels \
  --skip connectivity --min-time 120 \
  --nprocs 8 --mem-mb 16000 -w /work --notrack
```

**필수 전처리**: fMRIPrep 출력에 native-space T1w symlink 필요
```bash
# fMRIPrep이 MNI-space T1w만 출력한 경우
cd data/derivatives/fmriprep/sub-XXX/anat/
ln -s sub-XXX_space-MNI152NLin6Asym_res-2_desc-preproc_T1w.nii.gz \
      sub-XXX_desc-preproc_T1w.nii.gz
ln -s sub-XXX_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz \
      sub-XXX_desc-brain_mask.nii.gz
```

**convert_xcpd_to_npy.py ATLAS_NAME_MAP** — 4S 시리즈 키:
- `"4S156Parcels"` → `"4s156parcels"`
- `"4S256Parcels"` → `"4s256parcels"`
- `"4S356Parcels"` → `"4s356parcels"`
- `"4S456Parcels"` → `"4s456parcels"`
