# BS-NET (Bootstrapped Spearman-Brown Network)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/AntiGravityWorks/bsNet)

**BS-NET** is a neuroimaging analytical pipeline that extrapolates long-duration (15+ min) resting-state Functional Connectivity (FC) from ultra-short scan windows (≤ 2 minutes). It leverages block-bootstrapping, Spearman-Brown prophecy, empirical Bayesian priors, and Fisher z-space attenuation correction to bridge clinical scan-time constraints with high-fidelity network topology resolution.

**Core pipeline**: Short scan → Ledoit-Wolf shrinkage FC → Block bootstrap resampling → Spearman-Brown prophecy (k=7.5) → Bayesian empirical prior → Attenuation correction → **Fisher z-space bounding** → ρ̂T

---

## Documentation

Full documentation is in `docs/` with 6 categories. See [`docs/INDEX.md`](docs/INDEX.md) for the complete directory.

| Category | Contents |
|----------|----------|
| 1. Theory & Architecture | Spearman-Brown derivation, pipeline layout |
| 2. Experiment Logs | Phase 1–4 실험 기록 |
| 3. Results & Legends | Figure 1–6, Supplementary S1–S3/S6–S7 legends |
| 4. Publication Reports | Academic (EN) + General briefing (KR) |
| 5. Development Plans | Defense tracks A–G, ceiling correction, reliability-aware pipelines, Keane design |
| 6. Operations | Local setup, real data pipeline |

---

## Quick Start

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. ABIDE / ADHD-200 Validation
```bash
# ABIDE PCP N=468 (CC200, Fisher z, 10 seeds)
python src/scripts/run_abide_bsnet.py --atlas cc200 --correction-method fisher_z --n-seeds 10 --n-jobs 4

# ADHD-200 N=40 (CC200/CC400, multi-seed)
python src/scripts/run_nilearn_adhd_bsnet.py --atlas all --correction-method fisher_z --n-seeds 10 --n-jobs 4

# ADHD-200 PCP Full N=399
python src/scripts/run_adhd200_pcp_filtered.py --correction-method fisher_z --n-jobs 8
```

### 3. Convergence Validation (ds000243)
```bash
# Preprocessing
python src/scripts/preprocess_ds000243.py --input-dir data/ds000243/raw --n-jobs 8

# Duration sweep (6 atlases × 10 seeds)
bash src/scripts/run_ds000243_batch.sh

# Convergence validation (18 τ_short points)
python src/scripts/run_convergence_validation.py --dataset ds000243 --n-seeds 10 --n-jobs 8
```

### 4. Component Necessity & Progressive Ablation
```bash
# Leave-one-out component necessity (ABIDE N=468)
python src/scripts/run_component_necessity.py --input-npy data/abide/timeseries_cache/cc200/50033_cc200.npy

# Progressive ablation L0→L5
python src/scripts/run_progressive_ablation.py --dataset ds000243 --n-seeds 10 --n-jobs 8
```

### 5. Downstream & Reliability-Aware Analysis
```bash
# Downstream analysis (SVM, fingerprinting, graph metrics, etc.)
python src/scripts/run_downstream_analysis.py --n-jobs 8

# Reliability-aware clustering (ADHD-200 PCP, exploratory)
python src/scripts/run_reliability_aware_clustering.py --n-repeats 20

# Reliability-aware classification (LOSO, tangent FC)
python src/scripts/run_reliability_aware_classification.py --eval-scheme loso --n-repeats 20 --n-permutations 1000
```

### 6. Keane BP vs SZ (ds003404/ds005073)
```bash
# FC conversion
python src/scripts/convert_keane_restfc_to_npz.py

# Streaming pipeline (per-subject: datalad → fMRIPrep → BS-NET → cleanup)
bash src/scripts/run_keane_streaming_pipeline.sh --dataset all --cleanup-level minimal

# BS-NET recompute + classification
python src/scripts/run_keane_bsnet_recompute.py --correction-method fisher_z
python src/scripts/run_keane_bsnet_classification.py --n-permutations 1000
```

### 7. Figure Generation
```bash
# Main Figures
python src/visualization/plot_figure1_combined.py        # Fig 1: Method Overview (Pipeline + Convergence + τ_min)
python src/visualization/plot_figure2_component.py       # Fig 2: Component Necessity (LOO + Progressive + Cross-dataset)
python src/visualization/plot_figure3_abide.py           # Fig 3: ABIDE Validation
python src/visualization/plot_figure4_adhd.py            # Fig 4: ADHD Validation
python src/visualization/plot_figure5_structure.py       # Fig 5: Network Structure Preservation
python src/visualization/plot_figure6_classification.py  # Fig 6: ADHD Classification

# Supplementary Figures
python src/visualization/plot_figure_s1_progressive_full.py       # Fig S1: 6-Level Progressive × k-Group
python src/visualization/plot_figure_s2_k_stratification.py       # Fig S2: k-Stratification Dose-Response
python src/visualization/plot_figure_s3_abide_filtered_consort.py # Fig S3: ABIDE Filtered CONSORT
python src/visualization/plot_patient_utility_clustering.py       # Fig S6: Reliability-Aware Clustering
python src/visualization/plot_patient_utility_classification.py   # Fig S7: Reliability-Aware Classification
```

---

## Figure Index

### Main Figures (논문 본문, 6개)

| # | File | Content | Script |
|---|------|---------|--------|
| Fig 1 | `Fig1_Method_Overview.png` | Pipeline Schematic + Convergence Validation + τ_min Estimation | `plot_figure1_combined.py` |
| Fig 2 | `Fig2_Component_Necessity.png` | LOO + Progressive 4-level + Cross-dataset + Distribution | `plot_figure2_component.py` |
| Fig 3 | `Fig3_ABIDE_Validation.png` | ABIDE N=468 multi-seed Fisher z validation | `plot_figure3_abide.py` |
| Fig 4 | `Fig4_ADHD_Validation.png` | ADHD-200 cross-dataset generalization | `plot_figure4_adhd.py` |
| Fig 5 | `Fig5_Structure_Preservation.png` | Network topology/community preservation | `plot_figure5_structure.py` |
| Fig 6 | `Fig6_ADHD_Classification.png` | Clinical classification (SVM, 3 FC conditions) | `plot_figure6_classification.py` |

### Supplementary Figures (5개)

| # | File | Content | Script |
|---|------|---------|--------|
| Fig S1 | `FigS1_Progressive_6Level.png` | 6-level progressive ablation × k-group (3×2) | `plot_figure_s1_progressive_full.py` |
| Fig S2 | `FigS2_k_Stratification.png` | k-stratification dose-response + per-site summary | `plot_figure_s2_k_stratification.py` |
| Fig S3 | `FigS3_ABIDE_Filtered_CONSORT.png` | ABIDE filtered CONSORT flowchart | `plot_figure_s3_abide_filtered_consort.py` |
| Fig S6 | `FigS6_Reliability_Aware_Clustering.png` | ρ̂T strata unsupervised utility (EXPLORATORY) | `plot_patient_utility_clustering.py` |
| Fig S7 | `FigS7_Reliability_Aware_Classification.png` | LOSO supervised discrimination + permutation p | `plot_patient_utility_classification.py` |

---

## Project Structure
```
bsNet/
├── src/
│   ├── core/            # config, pipeline, bootstrap, stats, simulate
│   ├── data/            # data_loader, synthetic data generator
│   ├── scripts/         # see src/scripts/README.md
│   │   ├── [Validation]   run_abide_bsnet, run_nilearn_adhd_bsnet, run_fmriprep_bsnet, ...
│   │   ├── [Defense]      run_{sensitivity,ablation,stationarity,shrinkage,...} (Track A–G)
│   │   ├── [Convergence]  run_convergence_validation, run_progressive_ablation, run_abide_duration_sweep
│   │   ├── [Downstream]   run_downstream_analysis, run_reliability_aware_{classification,clustering}
│   │   ├── [Keane]        run_keane_bsnet_{recompute,classification}, convert_keane_restfc_to_npz
│   │   ├── [Data]         index_openneuro_hc, download_hc_100, download_adhd200_pcp, convert_*
│   │   ├── [Preprocess]   preprocess_ds007535, preprocess_ds000243, setup_and_preprocess
│   │   ├── [Simulation]   run_synthetic_baseline, sweep_simulation
│   │   └── [Pipeline]     14 shell scripts (fMRIPrep, XCP-D, Keane streaming, etc.)
│   └── visualization/  # Fig 1–6, FigS1–S3/S6–S7, style.py, legacy/
├── tests/               # pytest (74 tests)
├── docs/                # 6-category docs, see docs/INDEX.md
│   ├── dev/             # Session dev logs (12 files)
│   └── figure/          # Final figure PNGs
├── data/
│   ├── abide/           # ABIDE PCP: timeseries_cache/, results/
│   ├── adhd/            # ADHD-200: timeseries_cache/, results/, pcp/
│   ├── ds000243/        # WashU resting-state: raw/, timeseries_cache/, results/
│   ├── ds007535/        # SpeechHemi: raw/, timeseries_cache/, results/
│   └── ds005073/        # Keane BP/SZ: results/
├── artifacts/reports/   # Experiment result CSVs
└── pyproject.toml
```

---

## Key Results

| Dataset | N | r_FC (raw) | ρ̂T (BS-NET) | Δ | Improved |
|---------|---|------------|-------------|---|----------|
| ABIDE PCP (CC200) | 468 | 0.771 ± 0.071 | 0.843 ± 0.036 | +0.072 | 100% |
| ABIDE PCP (CC400) | 468 | 0.757 ± 0.071 | 0.834 ± 0.037 | +0.077 | 97.4% |
| ADHD-200 PCP Full | 399 | 0.525 ± 0.087 | 0.725 ± 0.049 | +0.201 | 100% |
| ds000243 (τ=120s) | 49 | 0.638 (peak) | 0.771 ± 0.032 | +0.134 | 100% |

All results use Fisher z correction (0% ceiling).

---

## Version History & License
* [CHANGELOG.md](CHANGELOG.md): History of pipeline refactoring and empirical calibration achievements.
* [LICENSE](LICENSE): Open-source MIT License.
