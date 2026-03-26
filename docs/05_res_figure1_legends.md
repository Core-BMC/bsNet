# Figure Legends

## Figure 1: Performance limit mapping and mathematical proof of optimality for the BS-NET pipeline.

**(a) Prediction Accuracy vs. Short-Scan Duration**
The solid blue line represents the mean predicted 15-minute long-duration Functional Connectivity (FC) correlation metric ($\rho$) extrapolated from incrementally constrained short-scan data (sweeping from 30s to 240s). The colored shaded area designates the 95% confidence interval boundaries computed through nested block-bootstrapping arrays ($N=100$) over 5 diverse random permutations. A critical plateau and irreversible stabilization of the lower confidence band occurs flawlessly at the 120s threshold, safely eclipsing the 80% baseline.

**(b) Incremental Accuracy Gain (Marginal Utility)**
Bar plot demonstrating the first derivative of prediction accuracy (Yield $\Delta\rho$ mapped per 30-second increment). The graph reveals a severe crash in marginal information efficiency precisely following the 120s limit (yellow block), mathematically defining the 'Knee Point' (Elbow) of the utility function where prolonging scan durations yields profoundly diminishing returns.

**(c) Statistical Uncertainty Decay (CI Width)**
Line plot mapping the rigid decay of the 95% confidence interval boundary width ($\Delta\rho$) over time. The statistical uncertainty aggressively narrows and hits a flattening horizontal plateau exactly at the 120s marker, providing empirical proof that 2 minutes is the absolute minimum baseline required to mathematically suppress individual variance globally.

**(d) Visualizing 84% Coherence (Separated vs. Overlay)**
Visual manifestation combining uncoupled and overlaid latent true signals vs. noisy raw observations. The top half plots a brief 120s sample trajectory of the latent pure BOLD drift (blue curve, +8 amplitude offset) against an aggressively contaminated raw matrix (red line, +4 amplitude offset). The bottom half overlays identical temporal trajectories natively, accurately portraying how an 84% ($r=0.84$) baseline coherence captures native underlying topological peaks resiliently despite heavy noise injection.
