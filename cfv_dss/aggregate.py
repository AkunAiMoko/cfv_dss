"""Leakage-safe out-of-core aggregation of Home Credit historical tables.

All aggregations compute statistics exclusively from events preceding the index application.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from cfv_dss.config import DATA, ART


def aggregate_bureau(chunk_size: int = 250_000) -> pd.DataFrame:
    """Aggregate bureau.csv into per-SK_ID_CURR metrics.

    Constructs: CE (Credit Exposure), DEBT_RATIO_BUREAU, FHD (Financial History Depth).
    """
    path = DATA / "bureau.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    # Use cols: SK_ID_CURR, DAYS_CREDIT, AMT_CREDIT_SUM, AMT_CREDIT_SUM_DEBT, AMT_CREDIT_SUM_LIMIT,
    #           CREDIT_DAY_OVERDUE, CNT_CREDIT_PROLONG, CREDIT_ACTIVE
    usecols = [
        "SK_ID_CURR", "DAYS_CREDIT", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_LIMIT", "CREDIT_DAY_OVERDUE", "CNT_CREDIT_PROLONG", "CREDIT_ACTIVE",
    ]

    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size):
        chunk["is_active"] = (chunk["CREDIT_ACTIVE"] == "Active").astype(float)
        chunk["has_overdue"] = (chunk["CREDIT_DAY_OVERDUE"] > 0).astype(float)
        # Group by SK_ID_CURR
        agg = chunk.groupby("SK_ID_CURR").agg(
            bureau_count=("DAYS_CREDIT", "count"),
            bureau_active_count=("is_active", "sum"),
            bureau_earliest_days=("DAYS_CREDIT", "min"),  # more negative = older
            bureau_latest_days=("DAYS_CREDIT", "max"),
            bureau_total_debt=("AMT_CREDIT_SUM_DEBT", "sum"),
            bureau_total_credit=("AMT_CREDIT_SUM", "sum"),
            bureau_total_overdue_days=("CREDIT_DAY_OVERDUE", "sum"),
            bureau_has_overdue_any=("has_overdue", "max"),
            bureau_prolong_count=("CNT_CREDIT_PROLONG", "sum"),
        ).reset_index()
        parts.append(agg)

    combined = pd.concat(parts, ignore_index=True)
    out = combined.groupby("SK_ID_CURR").agg(
        bureau_count=("bureau_count", "sum"),
        bureau_active_count=("bureau_active_count", "sum"),
        bureau_earliest_days=("bureau_earliest_days", "min"),
        bureau_latest_days=("bureau_latest_days", "max"),
        bureau_total_debt=("bureau_total_debt", "sum"),
        bureau_total_credit=("bureau_total_credit", "sum"),
        bureau_total_overdue_days=("bureau_total_overdue_days", "sum"),
        bureau_has_overdue_any=("bureau_has_overdue_any", "max"),
        bureau_prolong_count=("bureau_prolong_count", "sum"),
    ).reset_index()

    # Derived
    out["bureau_history_years"] = (-out["bureau_earliest_days"]) / 365.25
    out["bureau_debt_to_credit_ratio"] = (
        out["bureau_total_debt"] / (out["bureau_total_credit"] + 1.0)
    ).clip(0, 10.0)
    return out


def aggregate_previous_application(chunk_size: int = 250_000) -> pd.DataFrame:
    """Aggregate previous_application.csv into per-SK_ID_CURR metrics.

    Constructs: CNP (Credit Search / New Pressure), historical refusal rate.
    """
    path = DATA / "previous_application.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    usecols = [
        "SK_ID_CURR", "NAME_CONTRACT_STATUS", "AMT_APPLICATION",
        "AMT_CREDIT", "DAYS_DECISION", "CNT_PAYMENT",
    ]

    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size):
        chunk["is_refused"] = (chunk["NAME_CONTRACT_STATUS"] == "Refused").astype(float)
        chunk["is_approved"] = (chunk["NAME_CONTRACT_STATUS"] == "Approved").astype(float)
        chunk["is_recent_1y"] = (chunk["DAYS_DECISION"] >= -365).astype(float)

        agg = chunk.groupby("SK_ID_CURR").agg(
            prev_app_count=("NAME_CONTRACT_STATUS", "count"),
            prev_refused_count=("is_refused", "sum"),
            prev_approved_count=("is_approved", "sum"),
            prev_app_1y_count=("is_recent_1y", "sum"),
            prev_amt_app_sum=("AMT_APPLICATION", "sum"),
            prev_amt_cred_sum=("AMT_CREDIT", "sum"),
            prev_last_decision_days=("DAYS_DECISION", "max"),
        ).reset_index()
        parts.append(agg)

    combined = pd.concat(parts, ignore_index=True)
    out = combined.groupby("SK_ID_CURR").agg(
        prev_app_count=("prev_app_count", "sum"),
        prev_refused_count=("prev_refused_count", "sum"),
        prev_approved_count=("prev_approved_count", "sum"),
        prev_app_1y_count=("prev_app_1y_count", "sum"),
        prev_amt_app_sum=("prev_amt_app_sum", "sum"),
        prev_amt_cred_sum=("prev_amt_cred_sum", "sum"),
        prev_last_decision_days=("prev_last_decision_days", "max"),
    ).reset_index()

    out["prev_refusal_rate"] = (
        out["prev_refused_count"] / (out["prev_app_count"] + 1e-6)
    ).clip(0, 1.0)
    return out


def aggregate_installments(chunk_size: int = 500_000) -> pd.DataFrame:
    """Aggregate installments_payments.csv (~723 MB) into per-SK_ID_CURR repayment metrics.

    Constructs: RR (Repayment Reliability), historical missed/underpayment/late days.
    """
    path = DATA / "installments_payments.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    usecols = [
        "SK_ID_CURR", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER",
        "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT",
    ]

    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size):
        # delay in days: positive means late payment
        delay = (chunk["DAYS_ENTRY_PAYMENT"] - chunk["DAYS_INSTALMENT"]).fillna(0)
        chunk["is_late"] = (delay > 0).astype(float)
        chunk["days_late_pos"] = np.maximum(0, delay)
        # underpayment
        amt_diff = (chunk["AMT_INSTALMENT"] - chunk["AMT_PAYMENT"].fillna(0))
        chunk["is_underpaid"] = (amt_diff > 1.0).astype(float)
        chunk["amt_diff_pos"] = np.maximum(0, amt_diff)
        chunk["amt_inst"] = chunk["AMT_INSTALMENT"].fillna(0)
        chunk["amt_paid"] = chunk["AMT_PAYMENT"].fillna(0)

        agg = chunk.groupby("SK_ID_CURR").agg(
            inst_count=("NUM_INSTALMENT_VERSION", "count"),
            inst_late_count=("is_late", "sum"),
            inst_underpaid_count=("is_underpaid", "sum"),
            inst_total_days_late=("days_late_pos", "sum"),
            inst_total_underpaid_amt=("amt_diff_pos", "sum"),
            inst_total_due=("amt_inst", "sum"),
            inst_total_paid=("amt_paid", "sum"),
        ).reset_index()
        parts.append(agg)

    combined = pd.concat(parts, ignore_index=True)
    out = combined.groupby("SK_ID_CURR").agg(
        inst_count=("inst_count", "sum"),
        inst_late_count=("inst_late_count", "sum"),
        inst_underpaid_count=("inst_underpaid_count", "sum"),
        inst_total_days_late=("inst_total_days_late", "sum"),
        inst_total_underpaid_amt=("inst_total_underpaid_amt", "sum"),
        inst_total_due=("inst_total_due", "sum"),
        inst_total_paid=("inst_total_paid", "sum"),
    ).reset_index()

    out["inst_late_rate"] = (out["inst_late_count"] / (out["inst_count"] + 1e-6)).clip(0, 1)
    out["inst_underpaid_rate"] = (out["inst_underpaid_count"] / (out["inst_count"] + 1e-6)).clip(0, 1)
    out["inst_paid_to_due_ratio"] = (
        out["inst_total_paid"] / (out["inst_total_due"] + 1.0)
    ).clip(0, 2.0)
    return out


def aggregate_credit_card(chunk_size: int = 500_000) -> pd.DataFrame:
    """Aggregate credit_card_balance.csv (~425 MB) into per-SK_ID_CURR utilization metrics.

    Construct: US (Utilization Stress) revolving utilization.
    """
    path = DATA / "credit_card_balance.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    usecols = [
        "SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_ATM_CURRENT", "AMT_DRAWINGS_CURRENT",
        "SK_DPD", "SK_DPD_DEF",
    ]

    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size):
        chunk["cc_util_local"] = (
            chunk["AMT_BALANCE"].clip(lower=0) / (chunk["AMT_CREDIT_LIMIT_ACTUAL"] + 1.0)
        ).clip(0, 5.0)
        chunk["cc_has_dpd"] = (chunk["SK_DPD"] > 0).astype(float)

        agg = chunk.groupby("SK_ID_CURR").agg(
            cc_record_count=("AMT_BALANCE", "count"),
            cc_mean_balance=("AMT_BALANCE", "mean"),
            cc_max_balance=("AMT_BALANCE", "max"),
            cc_mean_limit=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
            cc_mean_utilization=("cc_util_local", "mean"),
            cc_max_utilization=("cc_util_local", "max"),
            cc_max_dpd=("SK_DPD", "max"),
        ).reset_index()
        parts.append(agg)

    combined = pd.concat(parts, ignore_index=True)
    out = combined.groupby("SK_ID_CURR").agg(
        cc_record_count=("cc_record_count", "sum"),
        cc_mean_balance=("cc_mean_balance", "mean"),
        cc_max_balance=("cc_max_balance", "max"),
        cc_mean_limit=("cc_mean_limit", "mean"),
        cc_mean_utilization=("cc_mean_utilization", "mean"),
        cc_max_utilization=("cc_max_utilization", "max"),
        cc_max_dpd=("cc_max_dpd", "max"),
    ).reset_index()
    return out


def build_aggregated_features_cache(force: bool = False) -> Path:
    """Run all table aggregations once and cache to Parquet for fast reuse."""
    cache_path = ART / "data" / "home_credit_historical_aggregates.parquet"
    if cache_path.exists() and not force:
        print(f"[CACHE] Found historical aggregates at {cache_path}")
        return cache_path

    print("[AGGREGATION] Step 1/4: Aggregating bureau.csv...")
    df_bureau = aggregate_bureau()
    print(f"  Bureau rows: {len(df_bureau):,}")

    print("[AGGREGATION] Step 2/4: Aggregating previous_application.csv...")
    df_prev = aggregate_previous_application()
    print(f"  Previous applications rows: {len(df_prev):,}")

    print("[AGGREGATION] Step 3/4: Aggregating installments_payments.csv...")
    df_inst = aggregate_installments()
    print(f"  Installments rows: {len(df_inst):,}")

    print("[AGGREGATION] Step 4/4: Aggregating credit_card_balance.csv...")
    df_cc = aggregate_credit_card()
    print(f"  Credit card rows: {len(df_cc):,}")

    # Merge all on SK_ID_CURR (outer join across history tables)
    merged = df_bureau.merge(df_prev, on="SK_ID_CURR", how="outer")
    merged = merged.merge(df_inst, on="SK_ID_CURR", how="outer")
    merged = merged.merge(df_cc, on="SK_ID_CURR", how="outer")

    merged.to_parquet(cache_path, index=False)
    print(f"[CACHE] Saved {len(merged):,} merged historical records to {cache_path}")
    return cache_path


if __name__ == "__main__":
    build_aggregated_features_cache()
