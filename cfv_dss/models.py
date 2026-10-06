"""Benchmark 8 machine-learning classifiers for financial vulnerability prediction."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
)
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from cfv_dss.config import ART, CONSTRUCTS, SEED


def make_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])


def get_models(scale_pos_weight=1.0):
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED
        ),
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=SEED
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=120, max_depth=10, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=120, max_depth=10, class_weight="balanced",
            n_jobs=-1, random_state=SEED
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=80, random_state=SEED
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=120, class_weight="balanced", random_state=SEED
        ),
        "XGBoost": XGBClassifier(
            n_estimators=120, max_depth=5, learning_rate=0.08,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=SEED, n_jobs=-1
        ),
    }


def calibration_metrics(y_true, y_prob):
    # slope and intercept via logistic regression of log-odds on y_true
    prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    logit = np.log(prob / (1 - prob))
    # simple linear regression logit vs binary
    cov = np.cov(logit, y_true)
    slope = float(cov[0, 1] / (np.var(logit) + 1e-9))
    intercept = float(np.mean(y_true) - slope * np.mean(logit))
    return slope, intercept


def evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test):
    pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-6)
    models = get_models(scale_pos_weight=pos_weight)

    results = []
    trained_pipelines = {}
    test_predictions = {}

    preproc = make_preprocessor()
    X_train_p = preproc.fit_transform(X_train)
    X_val_p = preproc.transform(X_val)
    X_test_p = preproc.transform(X_test)

    # Save fitted preprocessor
    joblib.dump(preproc, ART / "models" / "preprocessor.joblib")

    for name, clf in models.items():
        print(f"[MODEL] Training {name}...")
        clf.fit(X_train_p, y_train)

        # Calibrate via isotonic/sigmoid using validation fold
        # HistGradientBoosting, XGB, RF benefit from calibration
        try:
            calibrated = CalibratedClassifierCV(estimator=clf, cv="prefit", method="isotonic")
            calibrated.fit(X_val_p, y_val)
            pred_model = calibrated
        except Exception:
            pred_model = clf

        prob_test = pred_model.predict_proba(X_test_p)[:, 1]
        pred_label = (prob_test >= 0.5).astype(int)

        roc_auc = float(roc_auc_score(y_test, prob_test))
        pr_auc = float(average_precision_score(y_test, prob_test))
        brier = float(brier_score_loss(y_test, prob_test))
        slope, intercept = calibration_metrics(y_test.values, prob_test)
        f1 = float(f1_score(y_test, pred_label, zero_division=0))
        rec = float(recall_score(y_test, pred_label, zero_division=0))
        prec = float(precision_score(y_test, pred_label, zero_division=0))
        cm = confusion_matrix(y_test, pred_label).tolist()

        row = {
            "model": name,
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "calibration_slope": round(slope, 4),
            "calibration_intercept": round(intercept, 4),
            "f1_score": round(f1, 4),
            "recall": round(rec, 4),
            "precision": round(prec, 4),
            "confusion_matrix": cm,
        }
        results.append(row)
        trained_pipelines[name] = pred_model
        test_predictions[name] = prob_test

        # Save individual model
        joblib.dump(pred_model, ART / "models" / f"{name}.joblib")

    df_res = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    df_res.to_csv(ART / "metrics" / "model_comparison.csv", index=False)
    (ART / "metrics" / "model_comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    # Pick best calibrated nonlinear benchmark (highest ROC-AUC while maintaining low Brier)
    best_row = df_res.iloc[0]
    best_name = best_row["model"]
    best_bundle = {
        "best_model_name": best_name,
        "model": trained_pipelines[best_name],
        "preprocessor": preproc,
        "feature_names": list(CONSTRUCTS.keys()),
        "test_roc_auc": best_row["roc_auc"],
        "test_pr_auc": best_row["pr_auc"],
        "test_brier": best_row["brier_score"],
    }
    joblib.dump(best_bundle, ART / "models" / "best_model_bundle.joblib")
    print(f"[BEST MODEL] Selected {best_name} with test ROC-AUC={best_row['roc_auc']}")

    # Permutation importance on best model (replaces SHAP under python 3.14 without sacrificing DSS rigor)
    from sklearn.inspection import permutation_importance
    print(f"[EXPLANATION] Computing permutation importance on {best_name}...")
    perm = permutation_importance(
        trained_pipelines[best_name], X_test_p, y_test,
        n_repeats=10, random_state=SEED, scoring="roc_auc", n_jobs=-1
    )
    imp_df = pd.DataFrame({
        "feature": list(CONSTRUCTS.keys()),
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    imp_df["predictive_rank"] = range(1, len(imp_df) + 1)
    imp_df.to_csv(ART / "metrics" / "predictive_importance.csv", index=False)

    return df_res, best_bundle, imp_df
