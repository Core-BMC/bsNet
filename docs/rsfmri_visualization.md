# rs-fMRI Network Visualization Strategies for bsNet Study

## 1. Objective
This document summarizes visualization strategies for representing resting-state functional connectivity (rs-fMRI), focusing on:
- Global structure
- Network relationships
- Spatial brain interpretation

---

## 2. Global Network Representation

### ROI-Sorted Connectivity Matrix
- Reorder ROIs by RSN (DMN, FPN, DAN, VIS, SMN, LIM, SAL)
- Shows within/between network structure

Enhancements:
- Boundary lines
- Network color bars
- Difference matrices

### Network-to-Network Matrix
- Reduce to 7×7 or similar
- Each cell = mean connectivity

### Glass-Brain Connectome
- Nodes: ROI coordinates
- Edges: strongest connections

Settings:
- Top 5–10% edges
- Node color = network
- Node size = degree

### Interactive 3D Connectome
- HTML-based exploration

---

## 3. Network-Level Representation

### Within / Between Network Summary
- Bar or box plots
- Mean FC within and between RSNs

### Block Heatmap
- Network-level matrix + stats

### Chord Diagram
- Circular RSN layout
- Edge thickness = strength

---

## 4. Spatial Representation

### Surface Map
- RSN labels on cortex

### Hub Map
Metrics:
- Degree
- Strength
- Participation

### Seed-Based Maps
- PCC, insula, etc.

---

## 5. bsNet Figure Recommendation

1. ROI-sorted matrix
2. Network-level matrix
3. Connectome graph
4. Hub map

---

## 6. Advanced Ideas

- Positive vs negative FC split
- Reliability overlay
- Difference visualization
- Subject embedding (PCA/UMAP)
- Multiscale views

---

## 7. Summary

Combine:
- Matrix (quantitative)
- Graph (global)
- Spatial (anatomical)

This yields the most interpretable rs-fMRI representation.
