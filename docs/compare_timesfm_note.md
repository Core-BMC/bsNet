# Comparison Note: TimesFM as a Comparator for bsNet rs-fMRI Study

## 1. Purpose
This document defines how to include a generic time-series forecasting model (TimesFM) as a comparator in the bsNet rs-fMRI study.

## 2. Core Framing
- bsNet objective:
  - Estimate long-duration functional connectivity (FC) from short rs-fMRI segments.
  - Key mechanisms: bootstrap aggregation + reliability correction.

- TimesFM objective:
  - General-purpose time-series forecasting foundation model.

### Key Point
This is NOT a task-matched comparison.
This is a **model-identity comparison**:
> Can generic time-series forecasting inductive biases transfer to rs-fMRI FC recovery?

---

## 3. Comparator Positioning

### Role
TimesFM is used as:
- A **cross-paradigm comparator**
- Not a primary baseline
- Not a domain-specific model

### Statement for Paper
> We include a generic time-series forecasting foundation model (TimesFM) as a cross-paradigm comparator to evaluate whether general sequence prediction inductive biases transfer to short-duration rs-fMRI network recovery.

---

## 4. Implementation Strategy

### Option A (Recommended: Identity-preserving)
1. Input short rs-fMRI time series
2. Use TimesFM to forecast future time points
3. Reconstruct longer time series
4. Compute FC from reconstructed signal
5. Compare with long-duration reference FC

Pros:
- Preserves forecasting identity
- Clean interpretation

### Option B (Secondary)
1. Extract TimesFM embeddings
2. Train regression head → predict FC summary

Cons:
- Weakens forecasting identity

---

## 5. Main Baselines (Must Include)

- Short-window raw FC
- Bootstrap only
- Reliability correction only
- Bootstrap + reliability correction (proposed)

---

## 6. Evaluation Metrics

- Correlation with long-duration FC (Pearson/Spearman)
- Fisher z-transformed correlation
- Edge-wise MAE / RMSE
- Reliability (ICC / split-half)
- Scan-length vs performance curve

---

## 7. Interpretation Guidelines

### If TimesFM performs well:
- Suggests transferability of forecasting inductive bias

### If TimesFM performs poorly:
- Suggests mismatch between forecasting and FC recovery tasks

---

## 8. Limitation Statement (Important)

> TimesFM is designed for temporal forecasting rather than reliability-adjusted FC estimation, and thus serves as a non-task-matched comparator.

---

## 9. Summary

- TimesFM comparison is VALID under model-identity framing
- Must explicitly state task mismatch
- Should not be presented as SOTA baseline
- Best used as exploratory comparator
