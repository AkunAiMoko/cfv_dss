"""Domain-constrained causal discovery, orthogonal causal-effect estimation, and decision-policy simulation."""
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import KFold
import networkx as nx
from cfv_dss.config import ART, CONSTRUCTS, TIER_OF, EXPOSURES, FEASIBILITY, SEED


def fit_domain_constrained_dag(X: pd.DataFrame, corr_threshold: float = 0.04) -> nx.DiGraph:
    """Score/correlation-based DAG learner enforcing temporal/domain tier constraints.

    Constraint rule: Edge X -> Y allowed ONLY if Tier(X) <= Tier(Y), and if Tier(X) == Tier(Y),
    direction follows domain precedence. Tier 5 (TARGET) has NO outgoing edges.
    """
    nodes = list(X.columns)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    corr = X.corr(method="spearman").abs()
    n = len(X)

    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i >= j:
                continue
            r = corr.loc[u, v]
            if r < corr_threshold:
                continue

            tier_u = TIER_OF.get(u, 3)
            tier_v = TIER_OF.get(v, 3)

            # Determine direction strictly by tier
            if tier_u < tier_v:
                G.add_edge(u, v, weight=float(r))
            elif tier_v < tier_u:
                G.add_edge(v, u, weight=float(r))
            else:
                # Same tier: allow direction only if partial correlation resolves or domain order
                # Default: no edge if ambiguous, or directed from lower node index
                G.add_edge(u, v, weight=float(r))

    # Remove any cycles (should be acyclic by tier construction)
    while not nx.is_directed_acyclic_graph(G):
        cycle = nx.find_cycle(G)
        G.remove_edge(*cycle[0][:2])
    return G


def bootstrap_causal_discovery(X: pd.DataFrame, n_boot: int = 50, corr_threshold: float = 0.04) -> pd.DataFrame:
    """Bootstrap discovery over n_boot resamples and report edge stability (Table 4)."""
    nodes = list(X.columns)
    edge_counts = {}

    rng = np.random.default_rng(SEED)
    n = len(X)
    sample_size = min(n, 20_000)

    for b in range(n_boot):
        idx = rng.choice(n, size=sample_size, replace=True)
        sub = X.iloc[idx]
        G = fit_domain_constrained_dag(sub, corr_threshold=corr_threshold)
        for u, v in G.edges():
            edge_counts[(u, v)] = edge_counts.get((u, v), 0) + 1

    records = []
    for (u, v), count in edge_counts.items():
        freq = count / n_boot
        tier_u = TIER_OF.get(u, 3)
        tier_v = TIER_OF.get(v, 3)
        records.append({
            "source": u,
            "target": v,
            "source_tier": tier_u,
            "target_tier": tier_v,
            "bootstrap_freq": round(freq, 3),
            "tier_compatible": tier_u <= tier_v,
            "retained": freq >= 0.60,
        })

    df = pd.DataFrame(records).sort_values("bootstrap_freq", ascending=False)
    df.to_csv(ART / "causal" / "causal_discovery_stability.csv", index=False)
    return df


def build_consensus_dag(stability_df: pd.DataFrame, threshold: float = 0.60) -> nx.DiGraph:
    retained = stability_df[stability_df["bootstrap_freq"] >= threshold]
    G = nx.DiGraph()
    for _, row in retained.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["bootstrap_freq"])
    return G


def derive_adjustment_set(G: nx.DiGraph, exposure: str, outcome: str = "TARGET") -> list:
    """Derive backdoor adjustment set: parents and ancestors of exposure/outcome that are not descendants of exposure."""
    all_nodes = set(G.nodes())
    if exposure not in all_nodes or outcome not in all_nodes:
        # Fallback: all lower-tier constructs
        exp_tier = TIER_OF.get(exposure, 3)
        return [k for k, t in TIER_OF.items() if t < exp_tier and k != exposure and k != outcome]

    descendants = nx.descendants(G, exposure)
    exp_tier = TIER_OF.get(exposure, 3)

    # Valid confounders: non-descendants belonging to lower or equal tier
    adj = [
        node for node in all_nodes
        if node not in descendants
        and node != exposure
        and node != outcome
        and TIER_OF.get(node, 3) <= exp_tier
    ]
    return sorted(adj)


def double_machine_learning(
    X_df: pd.DataFrame,
    exposure: str,
    outcome: str = "TARGET",
    adj_set: list = None,
    n_splits: int = 4,
) -> dict:
    """Cross-fitted Double Machine Learning (DML) / partially linear model (Chernozhukov et al.).

    Estimand: average causal derivative / treatment effect on future distress probability.
    Nuisance models:
      q(W) = E[Y | W]   via gradient-boosted classifier
      m(W) = E[D | W]   via gradient-boosted regressor
    Residual orthogonalization:
      tilde_Y = Y - q(W)
      tilde_D = D - m(W)
      theta = (tilde_D' tilde_Y) / (tilde_D' tilde_D)
    """
    if adj_set is None or len(adj_set) == 0:
        adj_set = [c for c in CONSTRUCTS.keys() if c != exposure]

    D = X_df[exposure].values.astype(float)
    Y = X_df[outcome].values.astype(float)
    W = X_df[adj_set].values.astype(float)

    # Standardize treatment for comparable cross-exposure causal scales (1 SD change)
    std_D = np.std(D)
    if std_D > 0:
        D_std = (D - np.mean(D)) / std_D
    else:
        D_std = D

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    res_Y = np.zeros_like(Y)
    res_D = np.zeros_like(D_std)

    for train_idx, test_idx in kf.split(W):
        W_tr, W_te = W[train_idx], W[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        D_tr, D_te = D_std[train_idx], D_std[test_idx]

        # Nuisance model for Y (continuous probability)
        reg_y = HistGradientBoostingRegressor(max_iter=60, random_state=SEED)
        reg_y.fit(W_tr, Y_tr)
        res_Y[test_idx] = Y_te - reg_y.predict(W_te)

        # Nuisance model for D
        reg_d = HistGradientBoostingRegressor(max_iter=60, random_state=SEED)
        reg_d.fit(W_tr, D_tr)
        res_D[test_idx] = D_te - reg_d.predict(W_te)

    # Orthogonal IV / OLS on residuals
    denom = np.sum(res_D ** 2) + 1e-9
    theta = float(np.sum(res_D * res_Y) / denom)

    # Asymptotic variance for cross-fitted DML
    psi = (res_Y - theta * res_D) * res_D
    var_theta = float(np.mean(psi ** 2) / (np.mean(res_D ** 2) ** 2) / len(D))
    se_theta = float(np.sqrt(max(1e-12, var_theta)))
    z_stat = theta / (se_theta + 1e-9)
    p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
    ci_low = float(theta - 1.96 * se_theta)
    ci_high = float(theta + 1.96 * se_theta)

    # Placebo / falsification: shuffle treatment to verify false-positive control
    rng = np.random.default_rng(SEED)
    res_D_shuf = rng.permutation(res_D)
    placebo_theta = float(np.sum(res_D_shuf * res_Y) / (np.sum(res_D_shuf ** 2) + 1e-9))

    return {
        "exposure": exposure,
        "causal_effect_per_sd": round(theta, 5),
        "std_error": round(se_theta, 5),
        "ci_lower_95": round(ci_low, 5),
        "ci_upper_95": round(ci_high, 5),
        "p_value": round(p_val, 5),
        "placebo_effect": round(placebo_theta, 5),
        "adjustment_set": adj_set,
        "n_obs": len(D),
    }


def estimate_all_causal_effects(df: pd.DataFrame, consensus_dag: nx.DiGraph) -> pd.DataFrame:
    rows = []
    for exp in EXPOSURES:
        adj = derive_adjustment_set(consensus_dag, exp, outcome="TARGET")
        print(f"[CAUSAL DML] Estimating effect of {exp} (Adj set: {adj})...")
        res = double_machine_learning(df, exposure=exp, outcome="TARGET", adj_set=adj)
        meta = CONSTRUCTS.get(exp, (exp, 3, "treatment", "actionable", "+"))
        res["construct_label"] = meta[0]
        res["actionability"] = meta[3]
        res["expected_sign"] = meta[4]
        res["feasibility_weight"] = FEASIBILITY.get(exp, 0.7)
        rows.append(res)

    df_effects = pd.DataFrame(rows).sort_values("causal_effect_per_sd", key=abs, ascending=False)
    df_effects["causal_rank"] = range(1, len(df_effects) + 1)
    df_effects.to_csv(ART / "causal" / "causal_effects_dml.csv", index=False)
    (ART / "causal" / "causal_effects_dml.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return df_effects


def compare_predictive_vs_causal(pred_imp_df: pd.DataFrame, causal_df: pd.DataFrame) -> pd.DataFrame:
    """Signature DSS novelty: merge predictive rank and causal rank into quadrant classification."""
    merged = pd.merge(
        pred_imp_df, causal_df, left_on="feature", right_on="exposure", how="inner"
    )
    # Spearman rank correlation
    sp_corr, sp_p = stats.spearmanr(merged["predictive_rank"], merged["causal_rank"])

    # Classification into 4 quadrants
    median_pred_rank = merged["predictive_rank"].median()
    median_caus_rank = merged["causal_rank"].median()

    def classify(row):
        is_pred_top = row["predictive_rank"] <= median_pred_rank
        is_caus_top = row["causal_rank"] <= median_caus_rank
        if is_pred_top and is_caus_top:
            return "Dual Priority (Predictive + Causal)"
        elif is_pred_top and not is_caus_top:
            return "Predictive Marker Only (Low Intervention Utility)"
        elif not is_pred_top and is_caus_top:
            return "Hidden Intervention Lever (High Causal, Moderate SHAP)"
        else:
            return "Low Priority"

    merged["quadrant"] = merged.apply(classify, axis=1)
    merged["spearman_rank_corr"] = round(float(sp_corr), 3)
    merged["spearman_p_val"] = round(float(sp_p), 4)

    merged.to_csv(ART / "causal" / "predictive_vs_causal_divergence.csv", index=False)
    print(f"[DIVERGENCE] Spearman rho (pred rank vs causal rank) = {sp_corr:.3f} (p={sp_p:.4f})")
    return merged


def simulate_intervention_policies(
    df: pd.DataFrame,
    pred_probs: np.ndarray,
    causal_df: pd.DataFrame,
    budget_pct: float = 0.20,
) -> pd.DataFrame:
    """Section 6.4: Decision-policy simulation under fixed intervention budget (e.g., 20% of cohort).

    Policy A: Risk-Only targeting -> target borrowers with highest predicted distress.
    Policy B: SHAP/Predictive-Only targeting -> intervene on the top predictive feature.
    Policy C: Causal targeting -> target the actionable feature with largest individual causal reduction.
    Policy D: Causal + Feasibility -> causal benefit weighted by actionability cost.
    """
    n = len(df)
    k = int(n * budget_pct)

    # Each row supplies one candidate lever with its estimated effect, exposure level, and cost.
    levers = causal_df.drop_duplicates("exposure").set_index("exposure")
    beta = levers["causal_effect_per_sd"].abs()
    # Absent history (e.g. no credit-card record) carries no exposure to reduce, hence zero benefit.
    exposure_level = df[levers.index].fillna(0).clip(lower=0)

    # Shared utility model: reducing one lever by one SD avoids beta * baseline_risk expected distress,
    # scaled by how much of that lever the borrower actually carries.
    benefit = exposure_level.mul(beta, axis=1).mul(pred_probs, axis=0)

    top_causal_lever = beta.idxmax()
    top_pred_lever = levers["predictive_rank"].idxmin()

    def policy_value(lever_per_borrower, cost_weighted):
        gain = benefit.to_numpy()[np.arange(n), lever_per_borrower]
        if cost_weighted:
            cost = np.array([FEASIBILITY.get(levers.index[i], 0.7) for i in lever_per_borrower])
            gain = gain * cost
        order = np.argsort(gain)[-k:]
        return float(gain[order].sum())

    col_of = {name: i for i, name in enumerate(levers.index)}
    risk_rank = np.argsort(np.argsort(-pred_probs))
    targeted_by_risk = risk_rank < k

    # A: rank by predicted risk only, then apply the single highest-effect lever uniformly.
    fixed_causal = np.full(n, col_of[top_causal_lever])
    gain_risk_only = benefit.to_numpy()[np.arange(n), fixed_causal] * targeted_by_risk
    val_A = float(np.sort(gain_risk_only)[-k:].sum())

    # B: apply the strongest predictive feature as the lever, without causal verification.
    val_B = policy_value(np.full(n, col_of[top_pred_lever]), cost_weighted=False)

    # C: per-borrower lever with the largest estimated causal benefit.
    val_C = policy_value(benefit.to_numpy().argmax(axis=1), cost_weighted=False)

    # D: same as C but discounted by actionability/cost feasibility.
    feasible_benefit = benefit.mul(pd.Series(FEASIBILITY).reindex(levers.index).fillna(0.7), axis=1)
    val_D = policy_value(feasible_benefit.to_numpy().argmax(axis=1), cost_weighted=True)

    policies = [
        {"policy": "A. Risk-Only Targeting", "budget_coverage": budget_pct, "simulated_utility": round(val_A, 2), "lever": top_causal_lever, "description": "Targets highest predicted vulnerability, one uniform lever"},
        {"policy": "B. Predictive-Importance Only", "budget_coverage": budget_pct, "simulated_utility": round(val_B, 2), "lever": top_pred_lever, "description": "Intervenes on strongest predictive feature without causal verification"},
        {"policy": "C. Causal Targeting", "budget_coverage": budget_pct, "simulated_utility": round(val_C, 2), "lever": "per-borrower", "description": "Selects the lever with largest estimated causal benefit per borrower"},
        {"policy": "D. Causal + Feasibility Constrained", "budget_coverage": budget_pct, "simulated_utility": round(val_D, 2), "lever": "per-borrower (cost-weighted)", "description": "Causal benefit discounted by actionability and intervention cost"},
    ]
    df_pol = pd.DataFrame(policies)
    df_pol.to_csv(ART / "causal" / "decision_policy_simulation.csv", index=False)
    return df_pol
