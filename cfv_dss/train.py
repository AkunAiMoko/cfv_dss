"""End-to-end CRISP-DM execution for CFV-DSS."""
import json
import platform
import sys
import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from cfv_dss.config import ART, CONSTRUCTS, SEED
from cfv_dss.features import build_home_credit_features, build_gmsc_features
from cfv_dss.models import evaluate_models
from cfv_dss.causal import (
    bootstrap_causal_discovery, build_consensus_dag, estimate_all_causal_effects,
    compare_predictive_vs_causal, simulate_intervention_policies,
)


def save_figures(metrics, importance, stability, effects, divergence, policies, graph):
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5))
    metrics.sort_values("roc_auc").plot.barh(x="model", y=["roc_auc", "pr_auc"], ax=ax)
    ax.set_title("Predictive model comparison")
    ax.set_xlim(0, 1)
    fig.tight_layout(); fig.savefig(ART / "figures" / "model_comparison.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    importance.sort_values("importance_mean").plot.barh(x="feature", y="importance_mean", xerr="importance_std", ax=ax, legend=False)
    ax.set_title("Held-out permutation importance (predictive, not causal)")
    fig.tight_layout(); fig.savefig(ART / "figures" / "predictive_importance.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 8))
    for node in graph.nodes():
        from cfv_dss.config import TIER_OF
        graph.nodes[node]["layer"] = TIER_OF.get(node, 3)
    pos = nx.multipartite_layout(graph, subset_key="layer") if graph.nodes else {}
    if graph.nodes:
        widths = [graph[u][v].get("weight", .5) * 3 for u, v in graph.edges]
        nx.draw_networkx(graph, pos, ax=ax, node_color="#dbeafe", edge_color="#475569", width=widths, node_size=1600, font_size=8, arrows=True)
    ax.set_title("Domain-constrained consensus causal graph (bootstrap stability ≥60%)")
    ax.axis("off"); fig.tight_layout(); fig.savefig(ART / "figures" / "consensus_causal_graph.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, group in divergence.groupby("quadrant"):
        ax.scatter(group["importance_mean"], group["causal_effect_per_sd"].abs(), s=90, label=label)
        for _, row in group.iterrows(): ax.annotate(row["feature"], (row["importance_mean"], abs(row["causal_effect_per_sd"])), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Predictive importance (permutation ΔROC-AUC)"); ax.set_ylabel("Absolute DML effect per 1 SD")
    ax.set_title("Predictive importance versus causal intervention relevance")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(ART / "figures" / "predictive_causal_map.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    policies.plot.barh(x="policy", y="simulated_utility", ax=ax, legend=False, color="#2563eb")
    ax.set_title("Model-based policy value under equal intervention budget")
    fig.tight_layout(); fig.savefig(ART / "figures" / "policy_value.png", dpi=180); plt.close(fig)


def external_validation():
    df = pd.read_parquet(build_gmsc_features())
    features = [c for c in df.columns if c != "TARGET"]
    train, test = train_test_split(df, test_size=.25, random_state=SEED, stratify=df["TARGET"])
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", RobustScaler()),
                     ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))])
    pipe.fit(train[features], train["TARGET"])
    prob = pipe.predict_proba(test[features])[:, 1]
    result = {"dataset": "Give Me Some Credit", "n": len(df), "outcome_prevalence": float(df.TARGET.mean()),
              "roc_auc": float(roc_auc_score(test.TARGET, prob)), "pr_auc": float(average_precision_score(test.TARGET, prob)),
              "brier_score": float(brier_score_loss(test.TARGET, prob)), "constructs": features}
    pd.DataFrame([result]).to_csv(ART / "metrics" / "external_validation.csv", index=False)
    joblib.dump(pipe, ART / "models" / "gmsc_external_model.joblib")
    return result


def main():
    print("[CRISP-DM] 1/6 Business understanding: CFV-DSS intervention prioritization")
    print("[CRISP-DM] 2/6 Data understanding and preparation")
    df = pd.read_parquet(build_home_credit_features())
    features = list(CONSTRUCTS)
    train_val, test = train_test_split(df, test_size=.20, random_state=SEED, stratify=df.TARGET)
    train, val = train_test_split(train_val, test_size=.20, random_state=SEED, stratify=train_val.TARGET)
    split_ids = pd.concat([train.assign(split="train"), val.assign(split="validation"), test.assign(split="test")])[["SK_ID_CURR", "split"]]
    split_ids.to_csv(ART / "data" / "split_assignments.csv", index=False)

    print("[CRISP-DM] 3/6 Modeling: compare 8 algorithms")
    metrics, bundle, importance = evaluate_models(train[features], train.TARGET, val[features], val.TARGET, test[features], test.TARGET)
    pre = bundle["preprocessor"]
    X_causal = pd.DataFrame(pre.transform(df[features]), columns=features)
    X_causal["TARGET"] = df.TARGET.values
    X_causal.to_parquet(ART / "data" / "causal_design_matrix.parquet", index=False)

    print("[CRISP-DM] 4/6 Domain-constrained discovery and DML")
    stability = bootstrap_causal_discovery(X_causal, n_boot=50)
    graph = build_consensus_dag(stability, threshold=.60)
    nx.write_graphml(graph, ART / "causal" / "consensus_graph.graphml")
    effects = estimate_all_causal_effects(X_causal, graph)
    divergence = compare_predictive_vs_causal(importance, effects)

    best_prob = bundle["model"].predict_proba(pre.transform(test[features]))[:, 1]
    policies = simulate_intervention_policies(test.reset_index(drop=True), best_prob, divergence)
    external = external_validation()

    print("[CRISP-DM] 5/6 Evaluation and deployment artifacts")
    save_figures(metrics, importance, stability, effects, divergence, policies, graph)
    run = {"seed": SEED, "python": sys.version, "platform": platform.platform(), "cohort_n": len(df),
           "target_prevalence": float(df.TARGET.mean()), "best_model": bundle["best_model_name"],
           "best_roc_auc": float(bundle["test_roc_auc"]), "external_validation": external}
    (ART / "reports" / "run_metadata.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print("[CRISP-DM] 6/6 Complete")

if __name__ == "__main__":
    main()
