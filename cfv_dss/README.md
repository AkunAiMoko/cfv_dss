# CFV-DSS

CRISP-DM implementation for credit-related financial vulnerability decision support.

## Outputs

1. Calibrated future-distress risk.
2. Held-out predictive importance (not causal).
3. Domain-constrained consensus causal graph.
4. Cross-fitted orthogonal effect estimates with uncertainty and placebo tests.
5. Risk-only, predictive-only, causal, and causal-plus-feasibility policy simulation.
6. External predictive validation on Give Me Some Credit.
7. Streamlit deployment dashboard and DSS manuscript.

## Reproduce

From project root:

```bat
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r cfv_dss\requirements.txt
.venv\Scripts\python.exe -m cfv_dss.train
.venv\Scripts\python.exe -m cfv_dss.sensitivity
.venv\Scripts\python.exe -m cfv_dss.tests.test_pipeline
.venv\Scripts\python.exe -m streamlit run cfv_dss\app.py
```

Historical-table aggregation is cached at `cfv_dss/artifacts/data/home_credit_historical_aggregates.parquet`. Sensitivity analysis reuses the exact standardized design matrix saved by training.

## Main artifacts

- `artifacts/models/best_model_bundle.joblib`
- `artifacts/metrics/model_comparison.csv`
- `artifacts/metrics/predictive_importance.csv`
- `artifacts/causal/causal_discovery_stability.csv`
- `artifacts/causal/consensus_graph.graphml`
- `artifacts/causal/causal_effects_dml.csv`
- `artifacts/causal/predictive_vs_causal_divergence.csv`
- `artifacts/causal/decision_policy_simulation.csv`
- `artifacts/causal/graph_threshold_sensitivity.csv`
- `artifacts/causal/graph_threshold_rank_stability.csv`
- `artifacts/causal/overlap_diagnostics.csv`
- `artifacts/causal/unobserved_confounding_sensitivity.csv`
- `artifacts/reports/leakage_audit.json`
- `artifacts/reports/run_metadata.json`
- `artifacts/reports/sensitivity_summary.json`
- `artifacts/figures/*.png`
- `manuscript/article.md`

## Guardrails

Research prototype only. Observational estimates depend on measured-confounding, positivity, consistency, temporal-order, and model assumptions. Policy values are counterfactual simulations, not observed treatment effects. Do not use as sole basis for credit approval, pricing, collection, or adverse action.
