"""Small reproducible benchmark on generated Synthea records.

This benchmark is for software validation only. Synthea records are synthetic,
not real patients, and must not be presented as real-world clinical evidence.
"""
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from backend.app.config import DB_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DB_DIR / "synthea")
    parser.add_argument("--top-conditions", type=int, default=20)
    args = parser.parse_args()

    patients_path = args.data_dir / "patients.csv"
    conditions_path = args.data_dir / "conditions.csv"
    observations_path = args.data_dir / "observations.csv"
    if not all(path.exists() for path in (patients_path, conditions_path, observations_path)):
        raise FileNotFoundError("Expected Synthea patients.csv, conditions.csv, and observations.csv")

    patients = pd.read_csv(patients_path, usecols=["Id", "BIRTHDATE", "GENDER"])
    conditions = pd.read_csv(conditions_path)
    observations = pd.read_csv(observations_path)

    disorders = conditions[conditions["DESCRIPTION"].str.contains("disorder", case=False, na=False)]
    counts = disorders["DESCRIPTION"].value_counts()
    selected = counts[counts >= 20].head(args.top_conditions).index
    disorders = disorders[disorders["DESCRIPTION"].isin(selected)]

    observation_dates = pd.to_datetime(observations["DATE"], utc=True, errors="coerce")
    observations = observations.assign(_date=observation_dates)
    observations = observations[
        observations["TYPE"].eq("numeric") & observations["VALUE"].apply(
            lambda value: isinstance(value, (int, float)) or str(value).replace(".", "", 1).replace("-", "", 1).isdigit()
        )
    ].copy()
    observations["VALUE"] = pd.to_numeric(observations["VALUE"], errors="coerce")

    latest = (
        observations.sort_values("_date")
        .groupby(["PATIENT", "CODE"], as_index=False)
        .tail(1)
        .pivot(index="PATIENT", columns="CODE", values="VALUE")
    )
    latest.columns = [f"obs_{column}" for column in latest.columns]

    patients = patients.set_index("Id").copy()
    patients["AGE"] = (
        pd.Timestamp(date.today().isoformat()).tz_localize(None)
        - pd.to_datetime(patients["BIRTHDATE"], errors="coerce")
    ).dt.days / 365.25

    labels = disorders.groupby("PATIENT")["DESCRIPTION"].first()
    frame = patients.join(latest, how="inner").join(labels.rename("CONDITION"), how="inner")
    frame = frame.dropna(subset=["AGE", "CONDITION"])
    frame = frame.select_dtypes(include=["number", "object"]).fillna(0)

    feature_columns = ["AGE", "GENDER"] + [column for column in frame.columns if column.startswith("obs_")]
    X = frame[[column for column in feature_columns if column in frame.columns]]
    y = frame["CONDITION"].astype(str)
    if y.nunique() < 2 or len(frame) < 100:
        raise RuntimeError("Not enough synthetic records or condition classes for a benchmark")

    X_train, X_test, y_train, y_test = train_split(X, y, test_size=0.25, random_state=2026, stratify=y)

    numeric = [column for column in X.columns if column != "GENDER"]
    categorical = ["GENDER"] if "GENDER" in X.columns else []
    preprocess = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=2026,
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = list(model.named_steps["classifier"].classes_)
    top_5 = np.argsort(probabilities, axis=1)[:, -5:]
    top_5_recall = float(np.mean([
        label in {classes[index] for index in row_indices}
        for label, row_indices in zip(y_test, top_5)
    ]))

    report = {
        "dataset": "MITRE Synthea synthetic patient population",
        "data_directory": str(args.data_dir),
        "purpose": "software benchmark only; no real-world clinical validity",
        "synthetic_population_seed": 20261121,
        "rows": int(len(frame)),
        "condition_classes": int(y.nunique()),
        "top_1_accuracy": float(accuracy_score(y_test, predicted)),
        "top_5_recall": top_5_recall,
        "macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
        "classification_report": classification_report(y_test, predicted, output_dict=True, zero_division=0),
    }
    output = args.data_dir / "synthea_benchmark.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report saved to {output}")


def train_split(X, y, test_size, random_state, stratify):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


if __name__ == "__main__":
    main()
