# BS-NET 로컬 환경 설정 가이드

## 사전 요구사항

- Python 3.9+ (권장: 3.10)
- conda (Miniconda 또는 Anaconda)
- 디스크 여유 공간: ~5GB (데이터 + 패키지)

---

## 방법 1: 자동 설정 (권장)

```bash
cd bsNet/
chmod +x src/scripts/setup_local_env.sh
./src/scripts/setup_local_env.sh
```

이 스크립트가 수행하는 작업:
1. conda env `bsnet` 생성 (Python 3.10)
2. Python 패키지 설치
3. Schaefer 100 atlas 다운로드
4. OpenNeuro ds000030 1명 데이터 다운로드 (anat + func)
5. 1-subject end-to-end 테스트 실행

---

## 방법 2: 수동 설정

### Step 1: Python 환경

```bash
conda create -n bsnet python=3.10 -y
conda activate bsnet

pip install \
    numpy scipy pandas matplotlib seaborn \
    nibabel nilearn dipy scikit-learn \
    templateflow openneuro-py \
    ruff pytest
```

### Step 2: Schaefer 2018 Atlas 다운로드

100 parcels, 7 networks, MNI152 2mm 해상도.

**다운로드 URL (택 1):**

1. **templateflow (Python)**:
```python
from templateflow import api as tflow
import shutil
path = tflow.get('MNI152NLin6Asym', atlas='Schaefer2018',
                 desc='100Parcels7Networks', resolution=2,
                 suffix='dseg', extension='.nii.gz')
shutil.copy2(str(path), 'data/atlas/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz')
```

2. **직접 다운로드 (curl)**:
```bash
mkdir -p data/atlas
curl -L -o data/atlas/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz \
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
```

3. **GitHub 웹 브라우저에서 직접 다운로드**:
   - https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI
   - 파일: `Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz`
   - 저장 위치: `data/atlas/`

**검증:**
```python
import nibabel as nib, numpy as np
img = nib.load('data/atlas/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz')
n = len(np.unique(img.get_fdata())) - 1
print(f"Shape: {img.shape}, Parcels: {n}")  # (91, 109, 91), 100
```

### Step 3: OpenNeuro ds000030 데이터 다운로드

UCLA Consortium for Neuropsychiatric Phenomics (272 subjects, 130 HC).
각 subject에 anat (T1w) + func (rest BOLD) 필요.

**방법 A: openneuro-py (권장)**

```bash
# 1명 테스트 (sub-10159, healthy control)
pip install openneuro-py
cd data/openneuro

openneuro-py download --dataset ds000030 \
    --include "sub-10159/anat" \
    --include "sub-10159/func/sub-10159_task-rest_bold*"
```

**방법 B: AWS CLI (대량 다운로드 시 빠름)**

```bash
# AWS CLI 설치 후
pip install awscli

# 전체 HC 다운로드 (약 40GB)
aws s3 sync --no-sign-request \
    s3://openneuro.org/ds000030 \
    data/openneuro/ds000030/ \
    --exclude "*" \
    --include "participants.tsv" \
    --include "sub-*/anat/*T1w*" \
    --include "sub-*/func/*task-rest_bold*"
```

**방법 C: DataLad**

```bash
pip install datalad
datalad install https://github.com/OpenNeuroDatasets/ds000030.git
cd ds000030
datalad get sub-10159/anat/ sub-10159/func/sub-10159_task-rest_bold*
```

**방법 D: 웹 브라우저**
   - https://openneuro.org/datasets/ds000030/versions/1.0.0
   - "Download" 버튼 → 원하는 subject 선택

**검증:**
```python
import nibabel as nib
t1 = nib.load('data/openneuro/ds000030/sub-10159/anat/sub-10159_T1w.nii.gz')
bold = nib.load('data/openneuro/ds000030/sub-10159/func/sub-10159_task-rest_bold.nii.gz')
print(f"T1w:  {t1.shape}, voxel={list(t1.header.get_zooms()[:3])}")
print(f"BOLD: {bold.shape}, TR={bold.header.get_zooms()[-1]}s, {bold.shape[-1]} vols")
# 예상: T1w: (176, 256, 256), BOLD: (64, 64, 34, 152), TR=2.0s
```

### Step 4: 전처리 실행

```bash
cd bsNet/

# 1명 테스트
python src/scripts/preprocess_real_data.py --subject sub-10159 --verbose

# 전체 HC 실행 (약 2-5시간, subject당 ~2분)
python src/scripts/preprocess_real_data.py --run-all
```

---

## 출력 파일 구조

```
data/derivatives/
├── sub-10159/
│   ├── sub-10159_fc_full.npy         # 100×100 FC (전체 스캔)
│   ├── sub-10159_fc_short.npy        # 100×100 FC (첫 2분)
│   ├── sub-10159_fc_predicted.npy    # 100×100 BS-NET 예측 FC
│   ├── sub-10159_ts_mni.npy          # (n_vols, 100) time series
│   ├── sub-10159_quality.json        # QC 메트릭
│   └── sub-10159_bsnet_results.json  # BS-NET 파이프라인 결과
├── sub-10171/
│   └── ...
└── preprocessing_summary.json        # 전체 요약
```

---

## Healthy Control Subject 목록 (122명)

participants.tsv에서 `diagnosis == CONTROL` 이고 rest + T1w 보유한 subjects.
전체 목록은 `src/scripts/preprocess_real_data.py` 내 `get_hc_subject_list()` 에서 자동 필터링.

수동으로 확인:
```python
import pandas as pd
df = pd.read_csv('data/openneuro/ds000030/participants.tsv', sep='\t')
hc = df[df['diagnosis'] == 'CONTROL']
print(f"HC subjects: {len(hc)}")
```

---

## 트러블슈팅

| 문제 | 해결 |
|------|------|
| `templateflow` 다운로드 실패 | curl 직접 다운로드 사용 (Step 2 방법 2) |
| `openneuro-py` 느림/실패 | AWS CLI 사용 (Step 3 방법 B) |
| 메모리 부족 (OOM) | volume-by-volume 처리가 기본값이라 4GB RAM이면 충분 |
| dipy registration 경고 | 무시 가능 (convergence warning은 정상) |
| ROI 일부 NaN | QC json의 `n_rois_nonzero` 확인, <90이면 해당 subject 제외 권장 |
