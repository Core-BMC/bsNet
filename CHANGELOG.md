# Changelog
All notable changes to the **BS-NET** pipeline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] - 2026-03-30

### Added
- **Track H: ADHD vs Control Classification** (`run_adhd_classification.py`): Linear SVM (C=1.0) with Stratified 5-fold × 5 repeats on N=40 (20 ADHD, 20 Control). Three FC conditions (Raw, BS-NET, Reference) × two atlases (CC200, CC400). BS-NET Acc=0.720 (CC200), 0.700 (CC400), consistently ≥ Raw FC. Per-fold CSV export for scatter visualization.
- **Figure 7 — Classification Results** (`plot_figure7_classification.py`): 2×2 panel (Accuracy bars + scatter, AUC bars + scatter, ΔAccuracy, Summary). Per-fold jittered scatter overlay, error-bar-aware label placement.
- **Track H Figure Legend** (`docs/3.4_res_classification_legend.md`): Panel-level academic caption, Reference FC paradox interpretation, literature comparison (ADHD-200 Competition, Graph SVM, CC200 DL).
- **Defense Experiments Track A–H**: Completed 8 defense tracks including sensitivity, ablation, stationarity, shrinkage, component necessity, noise degradation, ceiling effect correction, and ADHD classification.
- **ABIDE/ADHD Empirical Validation**: ABIDE N=468 (CC200/CC400) Fisher z multi-seed, ADHD-200 N=40 group comparison, cross-dataset consistency confirmed.
- **Ceiling Effect Resolution (Track G)**: Fisher z-space correction adopted — original 85% ceiling → 0%. Four-method comparison (original, Fisher z, partial, soft clamp).
- **Figure Style Unification**: Extended `style.py` with FONT/LINE/MARKER/FIGSIZE constants, GROUP/ATLAS/CORRECTION palettes, helper functions. All figures regenerated at 300 DPI with consistent styling.
- **Component Necessity Batch Mode (Track E)**: `--input-dir`, `--n-subjects`, `--short-samples`, `--n-jobs`, dynamic k calculation.

## [0.1.0] - 2026-03-26

### Added
- **Formal Project Documentation (`docs/`)**: Restructured and structurally categorized theoretical, architectural, and analytical documentation using a grouped numbering schema (`1.1_theory_concept.md` through `4.2_pub_report_general.md`).
- **Academic Results Expansion**: Detailed theoretical proof of `120s` (2 minutes) being the absolute minimum viable scan length utilizing `Marginal Gain` (First Derivative) analysis and `Uncertainty Decay` tracking. Successfully generated large-scale clinical empirical validation legends (Figure 2) detailing 91% cohort pass rate.
- **Advanced Plotting Module**: Synthesized the 2x2 comprehensive visualization grid demonstrating statistical predictive accuracy, diminishing marginal efficiency, and noise extraction (`Figure1_Combined.png`, `Figure2_Validation.png`). 
- **Scale-Up Data Module (`src/run_real_data_scale.py`)**: Embedded local environment handlers extracting large-volume datasets automatically from the OpenNeuro hub (e.g., `ds000030`, `ds000243`), specifically retrieving robust Adult HC targets.
- **Root Architecture Elements**: Provisioned `.gitignore`, `README.md`, and standard permissive MIT `LICENSE`. 
- **Phase 4-7 Topological Verification**: Handled N=100 multi-subject validations on Schaefer 400 and Baseline Yeo-7 parcellations. Produced dense `Continuous Bi-polar Violin` modules replacing deprecated boxplots.
- **Reporting & Logs**: Instantiated `docs/experiments.log` and synthesized the final `docs/figure_legends.md` alongside appending final outputs to `4.1_pub_report_academic.md`. 

### Changed
- Refactored `src/plot_figure1_combined.py` to strip away legacy standalone component-panel plots. Overhauled plot height variables (16:9 ratio) and recalibrated signal axis limits ([-3, 15]) for flawless inner legend embedding.
- Converted isolated image generation pipelines into a unified rendering format outputting to `artifacts/reports/`.
- Isolated obsolete beta-phase scripts and figures into a safe `backup/` repository. 
- Shifted the base simulation engine (`src/core/simulate.py`) dynamically from block-structured N=100 nodes to the fully canonical dense N=400 `Schaefer Parcellation` topology standard.
- Enforced Y-axis scale constraint matching (`1.1 limits`) across all Figure 4 plots to eradicate aspect-ratio visual dead space. 

### Fixed
- Fixed unbounded divergence bugs in Spearman-Brown extensions targeting exceptionally short datasets (< 60s) by introducing Empirical Bayesian Prior Constraints and Politis & White Dynamic Bootstrap bounds in `src/bootstrap.py`.
- Suppressed spatial spurious matrices via Ledoit-Wolf Shrinkage deployment in `src/data_loader.py`.
