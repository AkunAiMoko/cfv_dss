# CFV-DSS Supplementary Material

## S1. Leakage audit

See `../artifacts/reports/leakage_audit.json` and `construct_dictionary.csv`.

## S2. Exact feature formulas

Executable definitions: `../features.py`. Aggregation definitions: `../aggregate.py`.

## S3. Hyperparameters

Executable model registry: `../models.py`. Eight fixed model classes use seed 42 and a common split.

## S4. Predictive diagnostics

See `../artifacts/metrics/model_comparison.csv`, `predictive_importance.csv`, and `../artifacts/figures/`.

## S5. Causal discovery matrices

See `../artifacts/causal/causal_discovery_stability.csv` and `consensus_graph.graphml`.

## S6. Graph-threshold sensitivity

The consensus graph was reconstructed at 60%, 70%, and 80% bootstrap edge-retention thresholds. The graphs retained 54, 51, and 49 edges, respectively. Adjustment sets and DML effects were re-estimated at each threshold; all six exposure ranks were unchanged (`rank_range = 0`). Detailed estimates are in `../artifacts/causal/graph_threshold_sensitivity.csv` and rank diagnostics in `graph_threshold_rank_stability.csv`; `../artifacts/figures/threshold_sensitivity.png` visualizes coefficient stability.

## S7. Overlap and trimming

Cross-fitted conditional-exposure models produced residual variance shares of 0.8624 (US), 0.8198 (DEBT_RATIO_BUREAU), 0.7651 (DBP), 0.7133 (CE), 0.6421 (CNP), and 0.3306 (AAP). Five exposures therefore had adequate support; AAP had limited support. Re-estimation after trimming the outer 1% of predicted exposure at both tails changed effects by at most 0.00067. Full diagnostics are in `../artifacts/causal/overlap_diagnostics.csv` and `../artifacts/figures/overlap_diagnostics.png`.

## S8. Heterogeneity

Dashboard computes borrower-level priorities from exposure, risk, effect, and feasibility. Prospective CATE validation remains required.

## S9. Sensitivity and placebo

Each exposure includes a shuffled-treatment placebo. Cinelli–Hazlett robustness values to nullify the estimates were 0.0440 (US), 0.0388 (DEBT_RATIO_BUREAU), 0.0245 (CNP), 0.0231 (DBP), 0.0222 (CE), and 0.0133 (AAP). Thus US was most robust and AAP least robust to unobserved confounding under the shared-partial-R² benchmark. Results are in `../artifacts/causal/unobserved_confounding_sensitivity.csv`. These bounds quantify, but do not eliminate, dependence on unmeasured-confounding assumptions.

## S10. External validation

Give Me Some Credit uses construct equivalence: utilization, debt burden, income capacity, delinquency, exposure, household burden, and age.

## S11. Reproducibility

Run commands are in `../README.md`. Environment versions: `../requirements.txt`; source checksums and platform: `../artifacts/reports/`.
