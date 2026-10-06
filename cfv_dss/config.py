"""Shared paths, construct metadata, and causal tiers for the CFV-DSS pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data Set Home Credit"
ART = ROOT / "cfv_dss" / "artifacts"
for sub in ("data", "models", "metrics", "figures", "causal", "reports"):
    (ART / sub).mkdir(parents=True, exist_ok=True)

SEED = 42

# construct -> (label, tier, causal role, actionability, expected sign on distress)
CONSTRUCTS = {
    "DBP": ("Debt Burden Pressure", 3, "treatment/mediator", "partly actionable", "+"),
    "US": ("Utilization Stress", 3, "treatment/mediator", "actionable", "+"),
    "DS": ("Delinquency Severity", 2, "confounder/diagnostic", "limited", "+"),
    "HAIC": ("Household-Adjusted Income Capacity", 1, "confounder/capacity", "partly actionable", "-"),
    "CE": ("Credit Exposure", 2, "treatment/mediator", "actionable (planning horizon)", "+"),
    "RR": ("Repayment Reliability", 2, "confounder/diagnostic", "limited", "-"),
    "AAP": ("Application Affordability Pressure", 4, "decision variable/treatment", "actionable at origination", "+"),
    "FHD": ("Financial History Depth", 0, "background/confounder", "not actionable", "-"),
    "HB": ("Household Burden", 0, "effect modifier/confounder", "low", "+"),
    "CNP": ("Credit Search / New-Credit Pressure", 3, "treatment/mediator", "actionable", "+"),
    "EMPSTAB": ("Employment Stability", 1, "confounder/capacity", "low", "-"),
    "AGE": ("Applicant Age", 0, "background/confounder", "not actionable", "-"),
    "DEBT_RATIO_BUREAU": ("External Debt-to-Limit Ratio", 2, "treatment/mediator", "actionable", "+"),
}

# Tier 5 = outcome, no outgoing edges. Edges only allowed from lower to >= tier.
OUTCOME = "TARGET"
TIER_OF = {k: v[1] for k, v in CONSTRUCTS.items()}
TIER_OF[OUTCOME] = 5

# exposures carried into causal estimation (actionable or decision-relevant only)
EXPOSURES = ["DBP", "US", "CE", "AAP", "CNP", "DEBT_RATIO_BUREAU"]

# intervention cost weights used by the feasibility-constrained policy (Section 6.4)
FEASIBILITY = {"US": 1.0, "CNP": 1.0, "CE": 0.6, "AAP": 0.8, "DBP": 0.5, "DEBT_RATIO_BUREAU": 0.6}

# Give Me Some Credit construct mapping (construct equivalence, not identical formulas)
GMSC_MAP = {
    "US": "RevolvingUtilizationOfUnsecuredLines",
    "DBP": "DebtRatio",
    "HAIC": "_income_per_head",
    "DS": "_delinq_severity",
    "CE": "NumberOfOpenCreditLinesAndLoans",
    "HB": "NumberOfDependents",
    "AGE": "age",
}
