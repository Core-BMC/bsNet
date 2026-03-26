# BS-NET (Bootstrapped Spearman-Brown Network)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/AntiGravityWorks/bsNet)

**BS-NET** is an advanced neuroimaging analytical pipeline engineered to precisely extrapolate long-duration (15+ min) resting-state Functional Connectivity (rsfMRI FC) patterns from ultra-short scan windows ($\le 2$ minutes). By leveraging rigorous block-bootstrapping, empirical Bayesian priors, and Spearman-Brown attenuation correction, BS-NET bridges clinical scan-time constraints with high-fidelity network topology resolution.

---

## 📖 Documentation Index
The documentation has been structurally re-indexed by category to guide users through the BS-NET developmental pipeline. For a standalone structural directory, see `[docs/INDEX.md]`.

### 🧠 1. Theory & Pipeline Architecture
* `[1.1_theory_concept.md]`: Theoretical derivations of Spearman-Brown extrapolation and block-bootstrapping mechanics.
* `[1.2_arch_pipeline.md]`: Infrastructural layout connecting spatial preprocessing, nuisance regression, and data-loaders.

### 📓 2. Operation & Experiment Logs
* `[2.1_log_experiment_20260326.md]`: Baseline laboratory logbook tracking daily empirical tuning and theoretical simulations.
* `[2.2_log_experiment_20260327.md]`: Final laboratory logbook detailing N=100 empirical scale-up, bias correction, and pass-rate validation.

### 📊 3. Quantitative Validation & Figure Legends
* `[3.1_res_optimal_duration.md]`: Comprehensive academic findings scaling the 120s threshold and large-cohort empirical proofs.
* `[3.2_res_figure1_legends.md]`: High-fidelity academic figure legends for statistical extrapolation thresholds (Figure 1).
* `[3.3_res_figure2_legends.md]`: Empirical validation legends and clinical reliability mapping on N=100 real-world cohort (Figure 2).
* `[docs/figure_legends.md]`: Official publish-ready captions mapping the N=400 topology and dense community geometries (Figures 3 & 4).

### 🎓 4. Formal Publication Reports
* `[4.1_pub_report_academic.md]`: Formal, peer-review-ready technical academic manuscript of the BS-NET module.
* `[4.2_pub_report_general.md]`: Accessible, executive-level summary of clinical implications and generalized verification.

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
Generate the Continuous Bi-polar Jaccard Overlaps and Global ARI modulations under the Schaefer 400 space.
```bash
python3 src/visualization/plot_figure3_topology.py
python3 src/visualization/plot_figure4_subnetworks.py
python3 src/visualization/plot_figure4_broken_axis.py
```

## 📜 Version History & License
* [CHANGELOG.md](CHANGELOG.md): History of pipeline refactoring and empirical calibration achievements.
* [LICENSE](LICENSE): Open-source MIT License enabling free academic and commercial adoption.
