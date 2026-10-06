"""Assert-based self-checks for the non-trivial CFV-DSS logic. Run: python -m cfv_dss.tests.test_pipeline"""
import numpy as np
import pandas as pd
import networkx as nx
from cfv_dss.config import ART, TIER_OF, EXPOSURES
from cfv_dss.causal import (
    fit_domain_constrained_dag, derive_adjustment_set,
    double_machine_learning, simulate_intervention_policies,
)


def test_tier_constraints_block_impossible_edges():
    rng = np.random.default_rng(0)
    n = 2000
    age = rng.normal(size=n)                       # tier 0
    haic = 0.5 * age + rng.normal(size=n)          # tier 1
    us = 0.6 * haic + rng.normal(size=n)           # tier 3
    target = 0.8 * us + rng.normal(size=n)         # tier 5
    df = pd.DataFrame({"AGE": age, "HAIC": haic, "US": us, "TARGET": target})
    G = fit_domain_constrained_dag(df, corr_threshold=0.03)

    assert nx.is_directed_acyclic_graph(G), "learned graph must be acyclic"
    assert G.out_degree("TARGET") == 0, "outcome (tier 5) must have no outgoing edges"
    for u, v in G.edges():
        assert TIER_OF[u] <= TIER_OF[v], f"edge {u}->{v} violates temporal tier order"


def test_adjustment_set_excludes_descendants_and_outcome():
    G = nx.DiGraph([("HAIC", "US"), ("US", "TARGET"), ("US", "CNP"), ("AGE", "US")])
    adj = derive_adjustment_set(G, "US", outcome="TARGET")
    assert "CNP" not in adj, "descendant of exposure must not enter the adjustment set"
    assert "TARGET" not in adj and "US" not in adj
    assert "HAIC" in adj and "AGE" in adj, "upstream confounders must be adjusted for"


def test_dml_recovers_known_effect_and_placebo_is_null():
    rng = np.random.default_rng(1)
    n = 6000
    w = rng.normal(size=n)
    d = 0.7 * w + rng.normal(size=n)
    true_theta_per_unit = 0.3
    y = true_theta_per_unit * d + 1.2 * w + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"D": d, "W": w, "TARGET": y})

    res = double_machine_learning(df, exposure="D", outcome="TARGET", adj_set=["W"])
    # DML reports the effect per 1 SD of treatment, so rescale the ground truth.
    expected = true_theta_per_unit * np.std(d)
    assert abs(res["causal_effect_per_sd"] - expected) < 0.05, res
    assert res["ci_lower_95"] < expected < res["ci_upper_95"], "CI must cover the true effect"
    assert abs(res["placebo_effect"]) < 0.05, "shuffled treatment must produce a null effect"


def test_absent_exposure_yields_no_simulated_benefit():
    n = 500
    causal_df = pd.DataFrame({
        "exposure": ["US", "CE"],
        "causal_effect_per_sd": [0.02, 0.01],
        "predictive_rank": [1, 2],
    })
    df = pd.DataFrame({"US": np.zeros(n), "CE": np.zeros(n)})
    pol = simulate_intervention_policies(df, np.full(n, 0.5), causal_df, budget_pct=0.2)
    assert pol["simulated_utility"].notna().all(), "policy value must never be NaN"
    assert (pol["simulated_utility"] == 0).all(), "zero exposure leaves nothing to intervene on"


def test_reported_artifacts_are_internally_consistent():
    metrics = pd.read_csv(ART / "metrics" / "model_comparison.csv")
    effects = pd.read_csv(ART / "causal" / "causal_effects_dml.csv")
    assert len(metrics) == 8, "eight algorithms must be compared"
    assert metrics["roc_auc"].between(0, 1).all()
    assert metrics["roc_auc"].is_monotonic_decreasing, "table must be sorted best-first"
    assert set(effects["exposure"]) == set(EXPOSURES)
    assert (effects["ci_lower_95"] <= effects["causal_effect_per_sd"]).all()
    assert (effects["causal_effect_per_sd"] <= effects["ci_upper_95"]).all()
    assert (effects["placebo_effect"].abs() < effects["causal_effect_per_sd"].abs()).all(), \
        "placebo effect must stay smaller than the estimated effect"


def test_sensitivity_artifacts_and_diagnostics_consistent():
    thr = pd.read_csv(ART / "causal" / "graph_threshold_sensitivity.csv")
    ranks = pd.read_csv(ART / "causal" / "graph_threshold_rank_stability.csv")
    overlap = pd.read_csv(ART / "causal" / "overlap_diagnostics.csv")
    confounding = pd.read_csv(ART / "causal" / "unobserved_confounding_sensitivity.csv")

    assert set(thr["threshold"].unique()) == {0.6, 0.7, 0.8}
    assert (ranks["rank_range"] == 0).all(), "causal ranks must be stable across thresholds"
    assert set(overlap["exposure"]) == set(EXPOSURES)
    assert overlap["residual_variance_share"].between(0.0, 1.0).all()
    assert set(overlap["overlap_verdict"]).issubset({"adequate", "limited", "weak"})
    assert np.isfinite(overlap["trimming_shift"]).all()
    assert set(confounding["exposure"]) == set(EXPOSURES)
    assert confounding["robustness_value_to_nullify"].between(0.0, 1.0).all()
    assert confounding.iloc[0]["exposure"] == "US"
    assert confounding.iloc[-1]["exposure"] == "AAP"

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All self-checks passed.")
