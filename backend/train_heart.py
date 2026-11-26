"""Train and evaluate a separate Framingham 10-year CHD risk model.

Provide a properly licensed CSV as db/framingham.csv with baseline predictors and a
binary TENYEARCHD outcome. No target is inferred or fabricated when the file is absent.
"""
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from backend.app.config import HEART_DATASET_PATH, HEART_MODEL_PATH

FEATURES = ["age", "sex_male", "current_smoker", "systolic_bp", "bp_treatment", "total_cholesterol", "hdl", "diabetes"]
ALIASES = {
    "age": ["age", "AGE"], "sex_male": ["sex_male", "male", "SEX_MALE", "SEX"],
    "current_smoker": ["current_smoker", "cursmoke", "CURSMOKE"],
    "systolic_bp": ["systolic_bp", "sysbp", "SYSBP"], "bp_treatment": ["bp_treatment", "bpmeds", "BPMEDS"],
    "total_cholesterol": ["total_cholesterol", "totchol", "TOTCHOL"], "hdl": ["hdl", "HDLC", "HDL"],
    "diabetes": ["diabetes", "DIABETES", "diabetes_mellitus"]}

def main():
    if not HEART_DATASET_PATH.exists(): raise SystemExit(f"No Framingham CSV found at {HEART_DATASET_PATH}. Obtain eligible data and place it there first.")
    df = pd.read_csv(HEART_DATASET_PATH)
    lower = {str(c).lower(): c for c in df.columns}
    def col_for(feature):
        for alias in ALIASES[feature]:
            if alias.lower() in lower: return lower[alias.lower()]
        raise ValueError(f"Missing required predictor '{feature}'. Accepted names: {ALIASES[feature]}")
    if "tenyearchd" not in lower: raise ValueError("Dataset must include the prospective binary TENYEARCHD outcome; prevalent heart-disease labels are not a substitute for 10-year risk.")
    y = pd.to_numeric(df[lower["tenyearchd"]], errors="coerce")
    x = pd.DataFrame({feature: pd.to_numeric(df[col_for(feature)], errors="coerce") for feature in FEATURES})
    # Framingham teaching data codes SEX as 1=men, 2=women; model input is a male indicator.
    if lower.get("sex") and col_for("sex_male") == lower["sex"]:
        x["sex_male"] = x["sex_male"].map({1: 1, 2: 0})
    valid = y.isin([0, 1]) & x.notna().any(axis=1)
    x, y, df = x.loc[valid], y.loc[valid].astype(int), df.loc[valid]
    if y.nunique() != 2 or y.value_counts().min() < 10: raise ValueError("Need both target classes with at least 10 rows each.")
    if "randid" in lower:
        splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=2026)
        train_ix, test_ix = next(splitter.split(x, y, groups=df[lower["randid"]]))
        x_train, x_test, y_train, y_test = x.iloc[train_ix], x.iloc[test_ix], y.iloc[train_ix], y.iloc[test_ix]
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=2026, stratify=y)
    base = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler()),
                     ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=2026))])
    model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 1]; pred = (prob >= .5).astype(int)
    report = {"dataset": str(HEART_DATASET_PATH), "target": "TENYEARCHD: prospective 10-year coronary heart disease event",
              "rows_used": int(len(x)), "train_rows": int(len(x_train)), "test_rows": int(len(x_test)), "patient_group_split": "RANDID" in lower,
              "metrics": {"accuracy": float(accuracy_score(y_test, pred)), "precision": float(precision_score(y_test, pred, zero_division=0)),
                          "recall": float(recall_score(y_test, pred, zero_division=0)), "f1": float(f1_score(y_test, pred, zero_division=0)),
                          "roc_auc": float(roc_auc_score(y_test, prob)), "brier_score": float(brier_score_loss(y_test, prob)),
                          "confusion_matrix": confusion_matrix(y_test, pred).tolist()},
              "features": FEATURES, "caveat": "Population research estimate, not a diagnosis or assessment of an ongoing heart attack. Validate external calibration and target population before any clinical use."}
    HEART_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "report": report}, HEART_MODEL_PATH)
    HEART_MODEL_PATH.with_name("heart_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
