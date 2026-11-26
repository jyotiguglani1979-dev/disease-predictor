"""
src/train_all.py   (Step 5: train all modules)
----------------------------------------------
Run everything:        python -m src.train_all
Run one module only:   python -m src.train_all A      (or B, or C)

WHAT EACH MODULE DOES AND WHY
  A  UCI 920 patients  : "do I have heart disease now?"
       - tested by LEAVING ONE HOSPITAL OUT (train on 3, test on the 4th).
         A random split would look better but would be dishonest, because
         the 4 hospitals are very different (disease rate 36% .. 94%).
       - small, shallow model, because 920 rows is little data.
  B  BRFSS 438k people : "heart attack / angina from lifestyle + history"
       - big data -> boosted trees on ALL rows, 60/20/20 split.
  C  Framingham 4.2k   : "heart disease in the next 10 years"
       - logistic regression vs boosted trees, best one wins on validation.

RULES USED EVERYWHERE (to keep results honest)
  1. Missing values are NOT filled before splitting (no leakage).
  2. The model is chosen on the VALIDATION set, never on the test set.
  3. Probabilities are calibrated, so "30% risk" really means about 30%.
  4. The alert threshold is chosen on validation (favours catching cases).
  5. The test set is used ONCE, at the end.
"""
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from config import MODELS, RANDOM_STATE, Tables
from src.db import load

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Feature lists
# ----------------------------------------------------------------------------
# A: vessels (66% missing) and thal (53% missing) are dropped: too empty, and
#    normal users will not have these test results.
A_FEATURES = ["age", "sex", "chest_pain", "resting_bp", "cholesterol",
              "fasting_sugar", "rest_ecg", "max_hr", "exercise_angina",
              "oldpeak", "st_slope"]
A_CATEGORICAL = ["chest_pain", "rest_ecg", "st_slope"]

# B: alcohol dropped (people change it AFTER a diagnosis = reverse causality).
#    BMI is kept, height and weight dropped (BMI already contains them).
B_FEATURES = ["sex", "age_group", "bmi", "sleep_hours", "physical_activity",
              "stroke", "copd", "kidney_disease", "arthritis", "depression",
              "diabetes", "smoker", "general_health"]
B_TARGETS = {"heart_attack": "heart_attack_model", "chd": "angina_chd_model"}

results: dict = {}   # collected for models/metrics.json


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def make_xgb(depth, n_estimators=800, lr=0.05, early=True, **extra):
    return XGBClassifier(
        n_estimators=n_estimators, learning_rate=lr, max_depth=depth,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_lambda=5.0, tree_method="hist", eval_metric="logloss",
        early_stopping_rounds=30 if early else None,
        random_state=RANDOM_STATE, n_jobs=-1, **extra)


def make_lr(numeric, categorical=()):
    """Logistic regression with its own filling + scaling (done inside, so no leakage)."""
    num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    cat = make_pipeline(SimpleImputer(strategy="most_frequent"),
                        OneHotEncoder(handle_unknown="ignore"))
    parts = [("num", num, list(numeric))]
    if categorical:
        parts.append(("cat", cat, list(categorical)))
    return Pipeline([("prep", ColumnTransformer(parts)),
                     ("lr", LogisticRegression(max_iter=3000, C=0.5))])


def scores(y, p):
    return {"auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p)}


def pick_threshold(y, p, beta=2.0):
    """Threshold that maximises F-beta (beta=2 -> catching cases matters more)."""
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    f = (1 + beta**2) * prec * rec / np.maximum(beta**2 * prec + rec, 1e-12)
    return float(thr[int(np.nanargmax(f))])


def at_threshold(y, p, thr):
    pred = p >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    return {"threshold": round(thr, 4),
            "recall": tp / max(int((y == 1).sum()), 1),
            "precision": tp / max(int(pred.sum()), 1),
            "flagged_share": float(pred.mean())}


def fmt(d):
    return "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())


def save_bundle(name, bundle):
    path = MODELS / f"{name}.joblib"
    joblib.dump(bundle, path)
    print(f"  saved -> {path}")


# ----------------------------------------------------------------------------
# Generic trainer for B and C  (train / validation / test)
# ----------------------------------------------------------------------------
def train_split(df, features, target, name, depths, lr, calibration):
    print(f"\n--- {name}: target='{target}' ---")
    df = df.dropna(subset=[target])
    X, y = df[features], df[target].astype(int)
    print(f"  rows: {len(df):,}   positive rate: {y.mean():.2%}   features: {len(features)}")

    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=RANDOM_STATE)
    print(f"  train {len(X_tr):,} | validation {len(X_va):,} | test {len(X_te):,}")

    # 1) candidates (all judged on the VALIDATION set)
    cands = {"logistic_regression": make_lr(features).fit(X_tr, y_tr)}
    for d in depths:
        m = make_xgb(d, lr=lr)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        cands[f"xgboost_depth{d}"] = m
    val = {k: scores(y_va, m.predict_proba(X_va)[:, 1]) for k, m in cands.items()}
    print("  validation scores:")
    for k, s in val.items():
        print(f"    {k:22s} {fmt(s)}")
    best = max(val, key=lambda k: val[k]["pr_auc"])
    print(f"  -> chosen (best validation PR-AUC): {best}")

    # 2) calibrate the chosen model on the validation set
    model = CalibratedClassifierCV(FrozenEstimator(cands[best]), method=calibration)
    model.fit(X_va, y_va)
    p_va = model.predict_proba(X_va)[:, 1]
    thr = pick_threshold(y_va, p_va)

    # 3) FINAL test (used once)
    p_te = model.predict_proba(X_te)[:, 1]
    raw_te = cands[best].predict_proba(X_te)[:, 1]
    base_te = cands["logistic_regression"].predict_proba(X_te)[:, 1]
    final = scores(y_te, p_te)
    print("  TEST results:")
    print(f"    chosen + calibrated : {fmt(final)}")
    print(f"    before calibration  : brier={brier_score_loss(y_te, raw_te):.4f}")
    print(f"    logistic baseline   : auc={roc_auc_score(y_te, base_te):.3f}")
    alert = at_threshold(y_te.to_numpy(), p_te, thr)
    print(f"    alert rule          : {fmt(alert)}")

    save_bundle(name, {"model": model, "features": features, "target": target,
                       "threshold": thr, "chosen": best, "test": final, "alert": alert})
    results[name] = {"chosen": best, "test": final, "alert": alert}


# ----------------------------------------------------------------------------
# Module B and C
# ----------------------------------------------------------------------------
def run_B():
    print("\n" + "=" * 70 + "\nMODULE B - BRFSS lifestyle screening (large data)\n" + "=" * 70)
    df = load(Tables.BRFSS)
    feats = [c for c in B_FEATURES if c in df.columns]
    for target, name in B_TARGETS.items():
        train_split(df, feats, target, name, depths=(4, 6), lr=0.08, calibration="isotonic")


def run_C():
    print("\n" + "=" * 70 + "\nMODULE C - Framingham 10-year risk\n" + "=" * 70)
    df = load(Tables.FRAMINGHAM)
    feats = [c for c in df.columns if c != "target"]
    # small data -> shallow trees, slow learning, Platt scaling (isotonic overfits small data)
    train_split(df, feats, "target", "risk_10year_model", depths=(2, 3), lr=0.03,
                calibration="sigmoid")


# ----------------------------------------------------------------------------
# Module A  (leave-one-hospital-out)
# ----------------------------------------------------------------------------
def run_A():
    print("\n" + "=" * 70 + "\nMODULE A - UCI clinical, tested hospital by hospital\n" + "=" * 70)
    df = load(Tables.UCI)
    X, y, site = df[A_FEATURES], df["target"].astype(int), df["site"]
    numeric = [c for c in A_FEATURES if c not in A_CATEGORICAL]

    models = {
        "logistic_regression": lambda: make_lr(numeric, A_CATEGORICAL),
        "xgboost_depth2": lambda: make_xgb(2, n_estimators=250, lr=0.04, early=False),
    }

    print("  Test 1: leave one hospital out (AUC on the hospital the model never saw)")
    rows = {k: {} for k in models}
    for s in sorted(site.unique()):
        te = site == s
        for k, mk in models.items():
            m = mk().fit(X[~te], y[~te])
            rows[k][s] = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
    table = pd.DataFrame(rows).round(3)
    table.loc["MEAN"] = table.mean().round(3)
    print(table.to_string())

    print("\n  Test 2: ordinary random 5-fold (looks better, but less honest)")
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = {}
    for k, mk in models.items():
        p = cross_val_predict(mk(), X, y, cv=skf, method="predict_proba")[:, 1]
        cv_auc[k] = roc_auc_score(y, p)
    print("   ", {k: round(v, 3) for k, v in cv_auc.items()})

    best = table.loc["MEAN"].idxmax()
    print(f"\n  -> chosen (best average across hospitals): {best}")

    final = CalibratedClassifierCV(models[best](), method="sigmoid", cv=5)
    final.fit(X, y)
    save_bundle("clinical_model", {
        "model": final, "features": A_FEATURES, "target": "target", "chosen": best,
        "leave_one_hospital_out_auc": table.loc["MEAN", best],
        "note": "Trained on all 4 hospitals. Expect lower accuracy on a new hospital."})
    results["clinical_model"] = {"chosen": best,
                                 "leave_one_hospital_out_auc": float(table.loc["MEAN", best]),
                                 "random_cv_auc": cv_auc[best]}


# ----------------------------------------------------------------------------
def main(which=None):
    which = [w.upper() for w in (which or ["A", "B", "C"])]
    runners = {"A": run_A, "B": run_B, "C": run_C}
    for w in which:
        if w not in runners:
            print(f"Unknown module '{w}'. Use A, B or C.")
            return
        runners[w]()
    out = MODELS / "metrics.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(results)
    out.write_text(json.dumps(old, indent=2, default=float))
    print(f"\nAll done. Summary saved -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
