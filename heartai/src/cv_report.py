"""
src/cv_report.py   (Step 6a: more reliable testing for the SMALL datasets)
--------------------------------------------------------------------------
Why: Module C has only ~850 test rows (about 130 positives), so one split can
swing the AUC by +-0.03. Here we test 25 times (5-fold x 5 repeats) and report
mean +- spread. We also compare with a trivial "age only" model, to prove the
full model adds value.

For Module C this script also RE-TRAINS the final model on ALL 4,238 rows
(the earlier model used only 60% of the rows) and saves it again.

Run:  python -m src.cv_report
"""
import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict)

from config import RANDOM_STATE, Tables
from src.db import load
from src.train_all import (A_CATEGORICAL, A_FEATURES, at_threshold, fmt, make_lr,
                           make_xgb, pick_threshold, save_bundle)

warnings.filterwarnings("ignore")


def repeated_cv(make_model, X, y, n_repeats=5):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=RANDOM_STATE)
    aucs, aps = [], []
    for tr, te in cv.split(X, y):
        m = make_model().fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], p))
        aps.append(average_precision_score(y.iloc[te], p))
    return np.array(aucs), np.array(aps)


def show(name, aucs, aps):
    print(f"    {name:22s} AUC {aucs.mean():.3f} +- {aucs.std():.3f}   "
          f"PR-AUC {aps.mean():.3f} +- {aps.std():.3f}")


def run_A():
    print("\n" + "=" * 70 + "\nMODULE A - repeated cross-validation (random folds)\n" + "=" * 70)
    df = load(Tables.UCI)
    X, y = df[A_FEATURES], df["target"].astype(int)
    numeric = [c for c in A_FEATURES if c not in A_CATEGORICAL]
    models = {
        "logistic_regression": lambda: make_lr(numeric, A_CATEGORICAL),
        "xgboost_depth2": lambda: make_xgb(2, n_estimators=250, lr=0.04, early=False),
    }
    print("  (hospital-wise AUC from Step 5 is the honest number; this shows stability)")
    for k, mk in models.items():
        show(k, *repeated_cv(mk, X, y))


def run_C():
    print("\n" + "=" * 70 + "\nMODULE C - repeated cross-validation + final model on ALL data\n" + "=" * 70)
    df = load(Tables.FRAMINGHAM)
    feats = [c for c in df.columns if c != "target"]
    X, y = df[feats], df["target"].astype(int)
    print(f"  rows: {len(df):,}   positive rate: {y.mean():.2%}   (5-fold x 5 repeats = 25 tests)")

    models = {
        "age_only_baseline": lambda: make_lr(["age"]),
        "logistic_regression": lambda: make_lr(feats),
        "xgboost_depth2": lambda: make_xgb(2, n_estimators=300, lr=0.03, early=False),
    }
    stats = {}
    for k, mk in models.items():
        a, p = repeated_cv(mk, X, y)
        stats[k] = (a, p)
        show(k, a, p)

    best = max((k for k in models if k != "age_only_baseline"), key=lambda k: stats[k][0].mean())
    print(f"\n  -> chosen (best mean AUC): {best}")

    # honest out-of-fold probabilities of the calibrated final model -> threshold + alert rule
    def final_model():
        return CalibratedClassifierCV(models[best](), method="sigmoid", cv=5)

    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(final_model(), X, y, cv=skf, method="predict_proba")[:, 1]
    thr = pick_threshold(y, oof)
    alert = at_threshold(y.to_numpy(), oof, thr)
    print(f"  out-of-fold: AUC={roc_auc_score(y, oof):.3f}  PR-AUC={average_precision_score(y, oof):.3f}  "
          f"brier={brier_score_loss(y, oof):.4f}")
    print(f"  alert rule : {fmt(alert)}")

    model = final_model().fit(X, y)       # final model uses ALL rows
    save_bundle("risk_10year_model", {
        "model": model, "features": feats, "target": "target", "threshold": thr,
        "chosen": best, "cv_auc_mean": float(stats[best][0].mean()),
        "cv_auc_std": float(stats[best][0].std()), "alert": alert})


if __name__ == "__main__":
    run_A()
    run_C()
    print("\nDone.")
