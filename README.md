# BS-NET (Bootstrapped Spearman-Brown Network)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/AntiGravityWorks/bsNet)

**BS-NET** is an advanced neuroimaging analytical pipeline engineered to precisely extrapolate long-duration (15+ min) resting-state Functional Connectivity (rsfMRI FC) patterns from ultra-short scan windows ($\le 2$ minutes). By leveraging rigorous block-bootstrapping, empirical Bayesian priors, and Spearman-Brown attenuation correction, BS-NET bridges clinical scan-time constraints with high-fidelity network topology resolution.

---

## 📖 Documentation Index
The documentation has been structurally indexed to guide users from underlying mathematical theories to final academic reporting. All files are located in the `docs/` repository.

### 🧠 Core Concepts & Pipeline Architecture
* `[01_theory_concept.md]`: Theoretical derivations of Spearman-Brown extrapolation and block-bootstrapping mechanics.
* `[02_arch_pipeline.md]`: Infrastructural layout connecting spatial preprocessing, nuisance regression, and data-loaders.

### 🧪 Experiment Logs & Results
* `[03_log_experiment_20260326.md]`: Raw laboratory logbook tracking daily empirical tuning (e.g., OpenNeuro ds000030).
* `[04_res_optimal_duration.md]`: Comprehensive academic findings mathematically proving 120s as the optimal scan duration (Marginal Gain & Uncertainty Decay).
* `[05_res_figure1_legends.md]`: High-fidelity academic figure legends for statistical threshold mapping (Figure 1).

### 🎓 Final Reports
* `[06_pub_report_academic.md]`: Formal, peer-review-ready technical manuscript of the pipeline.
* `[07_pub_report_general.md]`: Accessible, executive-level summary of the clinical implications.

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

## 📜 Version History & License
* [CHANGELOG.md](CHANGELOG.md): History of pipeline refactoring and empirical calibration achievements.
* [LICENSE](LICENSE): Open-source MIT License enabling free academic and commercial adoption.
