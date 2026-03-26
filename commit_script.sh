#!/bin/bash

# Ensure a clean slate
killall git 2>/dev/null
rm -f .git/index.lock

# Set git branch
git branch -M main
git remote add origin https://github.com/Core-BMC/bsNet.git 2>/dev/null

# 1. Project Scaffolding
git add .gitignore LICENSE README.md CHANGELOG.md
git commit -m "Initialize project with core configurations and documentation structure"

# 2. Core Mathematical Engine
git add src/bootstrap.py src/data_loader.py src/simulate.py src/main.py
git commit -m "Implement core BS-NET statistical extrapolation and regularized sampling modules"

# 3. Simulation Modeling
git add src/sweep_simulation.py artifacts/reports/duration_sweep_seeds_*.csv
git commit -m "Introduce synthetic AR(1) duration sweep simulations and validate optimization baselines"

# 4. Empirical Nuisance Pipeline
git add src/run_real_data.py src/run_real_data_scale.py artifacts/reports/scale_up_100_results_*.csv
git commit -m "Integrate OpenNeuro fetching and 100-subject empirical validation workflow"

# 5. Result Visualization
git add src/plot_figure1_combined.py docs/figure/Figure1_Combined.png artifacts/reports/Figure1_Combined.png
git commit -m "Develop comprehensive 2x2 viz grid defining 120s marginal gain and CI decay limits"

# 6. Academic Documentation
git add docs/*.md
git commit -m "Author sequentially indexed academic experiment reports, figure legends, and final analytical manuscripts"

# 7. Remaining artifacts
git add .
git commit -m "Finalize repository integrity and commit remaining validation artifacts"

# Output log
echo "--- Commit Check ---"
git log --oneline -n 10
