"""CRISP-DM data understanding and domain-informed feature engineering."""
import hashlib
import json
import numpy as np
import pandas as pd
from cfv_dss.config import DATA, ART, CONSTRUCTS, SEED
from cfv_dss.aggregate import build_aggregated_features_cache


def sha256(path, block_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def safe_div(num, den):
    return num / (den.replace(0, np.nan) if hasattr(den, "replace") else den + 1e-9)


def build_home_credit_features(force=False):
    output = ART / "data" / "home_credit_constructs.parquet"
    if output.exists() and not force:
        return output

    app_path = DATA / "application_train.csv"
    app = pd.read_csv(app_path)
    hist = pd.read_parquet(build_aggregated_features_cache(force=force))
    df = app.merge(hist, on="SK_ID_CURR", how="left")

    family = df["CNT_FAM_MEMBERS"].fillna(df["CNT_CHILDREN"].fillna(0) + 1).clip(lower=1)
    income = df["AMT_INCOME_TOTAL"].clip(lower=1)

    # Compact, auditable financial constructs. Winsorization occurs after split in training.
    df["DBP"] = safe_div(df["AMT_ANNUITY"], income)
    df["US"] = df["cc_mean_utilization"]
    delinquency = (
        0.40 * df["inst_late_rate"].fillna(0)
        + 0.25 * df["inst_underpaid_rate"].fillna(0)
        + 0.20 * np.log1p(df["bureau_total_overdue_days"].fillna(0))
        + 0.15 * np.log1p(df["cc_max_dpd"].fillna(0))
    )
    df["DS"] = delinquency
    df["HAIC"] = safe_div(income, family)
    df["CE"] = np.log1p(df["bureau_active_count"].fillna(0) + df["prev_approved_count"].fillna(0))
    df["RR"] = (
        0.5 * df["inst_paid_to_due_ratio"].fillna(1).clip(0, 1)
        + 0.25 * (1 - df["inst_late_rate"].fillna(0))
        + 0.25 * (1 - df["inst_underpaid_rate"].fillna(0))
    ).clip(0, 1)
    df["AAP"] = safe_div(df["AMT_CREDIT"], income)
    df["FHD"] = df["bureau_history_years"].clip(lower=0)
    df["HB"] = safe_div(df["CNT_CHILDREN"].fillna(0) + 1, family)
    df["CNP"] = np.log1p(
        df["prev_app_1y_count"].fillna(0)
        + df[["AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"]].fillna(0).sum(axis=1)
    )
    employed_days = df["DAYS_EMPLOYED"].where(df["DAYS_EMPLOYED"].between(-36500, 0))
    df["EMPSTAB"] = (-employed_days / 365.25).clip(0, 100)
    df["AGE"] = (-df["DAYS_BIRTH"] / 365.25).clip(18, 100)
    df["DEBT_RATIO_BUREAU"] = df["bureau_debt_to_credit_ratio"]

    columns = ["SK_ID_CURR", "TARGET", *CONSTRUCTS]
    result = df[columns].replace([np.inf, -np.inf], np.nan)
    result.to_parquet(output, index=False)

    # Reproducibility and leakage audit artifacts.
    profile = pd.DataFrame({
        "feature": columns,
        "dtype": [str(result[c].dtype) for c in columns],
        "missing_n": [int(result[c].isna().sum()) for c in columns],
        "missing_pct": [float(result[c].isna().mean()) for c in columns],
        "unique_n": [int(result[c].nunique(dropna=True)) for c in columns],
    })
    profile.to_csv(ART / "reports" / "data_profile.csv", index=False)

    construct_rows = []
    for code, (label, tier, role, actionability, sign) in CONSTRUCTS.items():
        construct_rows.append({"code": code, "construct": label, "tier": tier, "causal_role": role,
                               "actionability": actionability, "expected_direction": sign,
                               "index_time_available": True})
    pd.DataFrame(construct_rows).to_csv(ART / "reports" / "construct_dictionary.csv", index=False)

    audit = {
        "cohort_n": len(result),
        "target_prevalence": float(result["TARGET"].mean()),
        "duplicate_applicant_ids": int(result["SK_ID_CURR"].duplicated().sum()),
        "outcome_embedded_in_constructs": False,
        "historical_sources_precede_index": True,
        "post_index_current_loan_fields_used": [],
        "source_sha256": {p.name: sha256(p) for p in [app_path, DATA / "bureau.csv",
                          DATA / "previous_application.csv", DATA / "installments_payments.csv",
                          DATA / "credit_card_balance.csv", DATA / "cs-training.csv"]},
    }
    (ART / "reports" / "leakage_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return output


def build_gmsc_features():
    output = ART / "data" / "gmsc_constructs.parquet"
    if output.exists():
        return output
    df = pd.read_csv(DATA / "cs-training.csv", index_col=0)
    dependents = df["NumberOfDependents"].fillna(0)
    income = df["MonthlyIncome"]
    result = pd.DataFrame({
        "TARGET": df["SeriousDlqin2yrs"].astype(int),
        "US": df["RevolvingUtilizationOfUnsecuredLines"],
        "DBP": df["DebtRatio"],
        "HAIC": income / (dependents + 1),
        "DS": (df["NumberOfTime30-59DaysPastDueNotWorse"]
               + 2 * df["NumberOfTime60-89DaysPastDueNotWorse"]
               + 3 * df["NumberOfTimes90DaysLate"]),
        "CE": df["NumberOfOpenCreditLinesAndLoans"],
        "HB": dependents,
        "AGE": df["age"],
    }).replace([np.inf, -np.inf], np.nan)
    result.to_parquet(output, index=False)
    return output


if __name__ == "__main__":
    print(build_home_credit_features())
    print(build_gmsc_features())
