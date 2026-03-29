# BS-NET (Bootstrapped Spearman-Brown Network)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/AntiGravityWorks/bsNet)

**BS-NET** is an advanced neuroimaging analytical pipeline engineered to precisely extrapolate long-duration (15+ min) resting-state Functional Connectivity (rsfMRI FC) patterns from ultra-short scan windows ($\le 2$ minutes). By leveraging rigorous block-bootstrapping, empirical Bayesian priors, and Spearman-Brown attenuation correction, BS-NET bridges clinical scan-time constraints with high-fidelity network topology resolution.

---

## 📖 Documentation Index
The documentation has been structurally re-indexed by category to guide users through the BS-NET developmental pipeline. For a standalone structural directory, see `[docs/INDEX.md]`.

### 1. Theory & Pipeline Architecture
* `1.1_theory_concept.md`: Spearman-Brown extrapolation and block-bootstrapping derivations.
* `1.2_arch_pipeline.md`: Pipeline layout — preprocessing, nuisance regression, data loaders.

### 2. Experiment Logs
* `2.1_log_experiment_20260326.md`: Phase 1 — simulator tuning, local cache, 120s optimization.
* `2.2_log_experiment_20260327.md`: Phase 2 — N=100 scale-up, KDE bias correction, Figure 2.
* `2.3_log_sessions.log`: Session-level execution history (one-line per session).
* `2.4_log_experiment_20260328_30.md`: Phase 3 — defense experiments (Track A-H), ABIDE/ADHD validation, ceiling effect resolution, classification.

### 3. Validation Results & Figure Legends
* `3.1_res_figure_legends.md`: Master legends for Figures 1-4.
* `3.2_res_abide_figure_legends.md`: ABIDE PCP legends (Figure 5 series, multi-seed, ceiling).
* `3.3_res_adhd_figure_legends.md`: ADHD-200 legends (Figure 6 series, group comparison).
* `3.4_res_classification_legend.md`: Track H — ADHD classification (Figure 7, Linear SVM, Reference FC paradox).

### 4. Formal Publication Reports
* `4.1_pub_report_academic.md`: Peer-review-ready English academic manuscript.
* `4.2_pub_report_general.md`: Executive Korean briefing — clinical implications.

### 5. Development Plans & Defense
* `5.1`–`5.8`: Refactoring plan, critical review, defense experiments (Track A-G), methods, failure analysis, stationarity, ceiling effect correction.

---

## 🚀 Quick Start
### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Simulation Sweep & Validation (Optimal Duration Phase 4)
Generate synthetic BOLD transients, plot Marginal Utility matrices, and compute Figure 1 validation.
```bash
python3 src/sweep_simulation.py
python3 src/plot_figure1_combined.py
```

### 3. OpenNeuro Empirical Scaling (Phase 3)
Execute large-scale empirical correlation predictions against native open-access fMRI repositories.
```bash
python3 src/run_real_data_scale.py
```

### 4. High-Resolution Topological Validation (Phase 5-7)
```bash
python3 src/visualization/plot_figure3_topology.py
python3 src/visualization/plot_figure4_subnetworks.py
```

### 5. ADHD Classification (Track H)
```bash
python3 src/scripts/run_adhd_classification.py --atlas cc200 cc400 --n-bootstraps 100
python3 src/visualization/plot_figure7_classification.py
```

## Version History & License
* [CHANGELOG.md](CHANGELOG.md): History of pipeline refactoring and empirical calibration achievements.
* [LICENSE](LICENSE): Open-source MIT License enabling free academic and commercial adoption.
