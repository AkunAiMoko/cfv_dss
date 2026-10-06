"""Graph-threshold sensitivity, overlap diagnostics, and unobserved-confounding sensitivity.

Supports Appendices S6, S7, and S9 of the manuscript.
"""
import ast
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from cfv_dss.config import ART, EXPOSURES, SEED, CONSTRUCTS
from cfv_dss.causal import build_consensus_dag, derive_adjustment_set, double_machine_learning


def graph_threshold_sensitivity(X_causal, stability, thresholds=(0.60, 0.70, 0.80)):
    """Appendix S6: re-derive adjustment sets and re-estimate effects at each retention threshold."""
    rows = []
    for thr in thresholds:
        graph = build_consensus_dag(stability, threshold=thr)
        retained = int((stability["bootstrap_freq"] >= thr).sum())
        for exposure in EXPOSURES:
            adj = derive_adjustment_set(graph, exposure, outcome="TARGET")
            res = double_machine_learning(X_causal, exposure=exposure, outcome="TARGET", adj_set=adj)
            rows.append({
                "threshold": thr,
                "edges_retained": retained,
                "exposure": exposure,
                "adjustment_set_size": len(adj),
                "causal_effect_per_sd": res["causal_effect_per_sd"],
                "ci_lower_95": res["ci_lower_95"],
                "ci_upper_95": res["ci_upper_95"],
                "p_value": res["p_value"],
            })
        print(f"[SENSITIVITY] threshold {thr:.2f}: {retained} edges retained")

    df = pd.DataFrame(rows)
    df["causal_rank_within_threshold"] = (
        df.groupby("threshold")["causal_effect_per_sd"]
        .rank(ascending=False, method="min").astype(int)
    )
    df.to_csv(ART / "causal" / "graph_threshold_sensitivity.csv", index=False)

    # Rank stability across thresholds is the decision-relevant question, not coefficient equality.
    pivot = df.pivot(index="exposure", columns="threshold", values="causal_rank_within_threshold")
    pivot["rank_range"] = pivot.max(axis=1) - pivot.min(axis=1)
    pivot.to_csv(ART / "causal" / "graph_threshold_rank_stability.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    for exposure, grp in df.groupby("exposure"):
        ax.plot(grp["threshold"], grp["causal_effect_per_sd"], marker="o", label=exposure)
    ax.axhline(0, color="#64748b", lw=.8, ls="--")
    ax.set_xlabel("Edge-retention threshold"); ax.set_ylabel("DML effect per 1 SD")
    ax.set_title("Effect stability across graph-retention thresholds")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ART / "figures" / "threshold_sensitivity.png", dpi=180); plt.close(fig)
    return df, pivot


def overlap_diagnostics(X_causal, consensus_graph, n_bins=10):
    """Appendix S7: generalized propensity overlap for continuous exposures.

    For a continuous treatment there is no propensity score in [0,1]; the analogue is the
    conditional exposure distribution. We fit E[D|W], then check whether the residual spread
    is wide enough across the predicted-exposure range to support contrast estimation.
    """
    rows = []
    for exposure in EXPOSURES:
        adj = derive_adjustment_set(consensus_graph, exposure, outcome="TARGET")
        if not adj:
            adj = [c for c in CONSTRUCTS if c != exposure]
        D = X_causal[exposure].to_numpy(dtype=float)
        W = X_causal[adj].to_numpy(dtype=float)

        kf = KFold(n_splits=4, shuffle=True, random_state=SEED)
        d_hat = np.zeros_like(D)
        for tr, te in kf.split(W):
            model = HistGradientBoostingRegressor(max_iter=60, random_state=SEED)
            model.fit(W[tr], D[tr])
            d_hat[te] = model.predict(W[te])
        resid = D - d_hat

        # Residual variance share: 1.0 means confounders explain nothing (ideal overlap),
        # near 0 means exposure is nearly determined by confounders (positivity failure).
        resid_share = float(np.var(resid) / (np.var(D) + 1e-12))

        bins = pd.qcut(pd.Series(d_hat), q=n_bins, duplicates="drop")
        per_bin = pd.DataFrame({"resid": resid, "bin": bins}).groupby("bin", observed=True)["resid"]
        bin_sd = per_bin.std()
        min_share = float((bin_sd.min() ** 2) / (np.var(D) + 1e-12))

        # Trimming sensitivity: drop the most extreme 2% of predicted exposure and re-estimate.
        lo, hi = np.quantile(d_hat, [0.01, 0.99])
        keep = (d_hat >= lo) & (d_hat <= hi)
        trimmed = double_machine_learning(
            X_causal.loc[keep].reset_index(drop=True), exposure=exposure, outcome="TARGET", adj_set=adj
        )
        full = double_machine_learning(X_causal, exposure=exposure, outcome="TARGET", adj_set=adj)

        verdict = (
            "adequate" if resid_share >= 0.50 and min_share >= 0.20
            else "limited" if resid_share >= 0.25
            else "weak"
        )
        rows.append({
            "exposure": exposure,
            "residual_variance_share": round(resid_share, 4),
            "worst_bin_variance_share": round(min_share, 4),
            "overlap_verdict": verdict,
            "effect_full": full["causal_effect_per_sd"],
            "effect_trimmed_1_99pct": trimmed["causal_effect_per_sd"],
            "trimming_shift": round(trimmed["causal_effect_per_sd"] - full["causal_effect_per_sd"], 5),
            "n_full": full["n_obs"],
            "n_trimmed": trimmed["n_obs"],
        })
        print(f"[OVERLAP] {exposure}: share={resid_share:.3f} verdict={verdict}")

    df = pd.DataFrame(rows)
    df.to_csv(ART / "causal" / "overlap_diagnostics.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(df["exposure"], df["residual_variance_share"], color="#2563eb")
    ax.axvline(0.50, color="#16a34a", ls="--", lw=1, label="adequate ≥ 0.50")
    ax.axvline(0.25, color="#dc2626", ls="--", lw=1, label="weak < 0.25")
    ax.set_xlabel("Residual exposure variance share (overlap proxy)")
    ax.set_title("Positivity/overlap diagnostics for continuous exposures")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ART / "figures" / "overlap_diagnostics.png", dpi=180); plt.close(fig)
    return df


def unobserved_confounding_sensitivity(effects, overlap):
    """Appendix S9: how strong must an unmeasured confounder be to nullify each effect?

    Partial-R2 style bound: an omitted confounder must explain enough residual variance in both
    exposure and outcome for the product term to cancel the estimate.
    """
    rows = []
    merged = effects.merge(
        overlap[["exposure", "residual_variance_share", "overlap_verdict"]], on="exposure"
    )
    for _, r in merged.iterrows():
        effect = abs(r["causal_effect_per_sd"])
        se = r["std_error"]
        # Cinelli-Hazlett robustness value: the shared partial R2 an omitted confounder would
        # need with both treatment and outcome to drive the estimate to zero.
        # RV = 0.5 * (sqrt(f^4 + 4 f^2) - f^2), where f^2 = t^2 / df.
        t_stat = effect / (se + 1e-12)
        dof = max(1, int(r["n_obs"]) - len(ast.literal_eval(r["adjustment_set"])) - 2)
        f_sq = (t_stat ** 2) / dof
        rv = float(0.5 * (np.sqrt(f_sq ** 2 + 4 * f_sq) - f_sq))
        rv = min(max(rv, 0.0), 1.0)
        rows.append({
            "exposure": r["exposure"],
            "effect_per_sd": r["causal_effect_per_sd"],
            "t_statistic": round(float(t_stat), 2),
            "robustness_value_to_nullify": round(rv, 4),
            "overlap_verdict": r["overlap_verdict"],
            "interpretation": (
                "robust to weak confounding" if rv >= 0.04
                else "sensitive to modest confounding" if rv >= 0.02
                else "highly sensitive"
            ),
        })
    df = pd.DataFrame(rows).sort_values("robustness_value_to_nullify", ascending=False)
    df.to_csv(ART / "causal" / "unobserved_confounding_sensitivity.csv", index=False)
    return df


def main():
    import networkx as nx
    X_causal = pd.read_parquet(ART / "data" / "causal_design_matrix.parquet")
    stability = pd.read_csv(ART / "causal" / "causal_discovery_stability.csv")
    effects = pd.read_csv(ART / "causal" / "causal_effects_dml.csv")
    graph = nx.read_graphml(ART / "causal" / "consensus_graph.graphml")

    print("[S6] Graph-threshold sensitivity")
    thr_df, rank_pivot = graph_threshold_sensitivity(X_causal, stability)
    print("[S7] Overlap and trimming diagnostics")
    ov_df = overlap_diagnostics(X_causal, graph)
    print("[S9] Unobserved-confounding sensitivity")
    uc_df = unobserved_confounding_sensitivity(effects, ov_df)

    summary = {
        "thresholds_tested": sorted(thr_df["threshold"].unique().tolist()),
        "max_rank_change_across_thresholds": int(rank_pivot["rank_range"].max()),
        "exposures_with_adequate_overlap": ov_df.query("overlap_verdict == 'adequate'")["exposure"].tolist(),
        "exposures_with_weak_overlap": ov_df.query("overlap_verdict == 'weak'")["exposure"].tolist(),
        "max_abs_trimming_shift": float(ov_df["trimming_shift"].abs().max()),
        "most_robust_exposure": uc_df.iloc[0]["exposure"],
        "least_robust_exposure": uc_df.iloc[-1]["exposure"],
    }
    (ART / "reports" / "sensitivity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
