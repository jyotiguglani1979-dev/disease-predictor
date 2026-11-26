"""Train the binary Indian symptom disease model (honest evaluation version).

Usage (from the project root):  python -m backend.train_indian

What changed compared with the first version
--------------------------------------------
1. The CSV has only ~145 unique symptom patterns repeated thousands of times.
   Random train/test splitting put the same pattern on both sides, which gave a
   fake 99% score. Now we remove duplicates and split by pattern, so every test
   pattern is one the model has never seen.
2. Real users type 2-4 symptoms, not a full record. Training now also includes
   every smaller subset (2..n symptoms) of each pattern, and the evaluation
   reports accuracy with all symptoms, any 3, and any 2 symptoms.
3. Logistic regression gives real probabilities (the old LinearSVC did not).
4. Reports top-1, top-3 and top-5 accuracy, which is what the app shows.
"""
import itertools
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from backend.app.config import DB_DIR, INDIAN_MODEL_PATH

SEED = 2026
MIN_SYMPTOMS = 2
C_VALUE = 1.0
REPEATS = 3
FOLDS = 5


def make_model():
    return LogisticRegression(C=C_VALUE, max_iter=3000, class_weight="balanced")


def load_unique(path):
    frame = pd.read_csv(path)
    if "disease" not in frame.columns:
        raise RuntimeError("indian_disease_binary.csv must have a 'disease' column")
    symptoms = [c for c in frame.columns if c != "disease"]
    labels = frame["disease"].astype(str).str.strip().to_numpy()
    matrix = frame[symptoms].to_numpy(dtype=np.uint8)
    seen, rows, ys, groups, pattern_ids = set(), [], [], [], {}
    for row, label in zip(matrix, labels):
        pattern = row.tobytes()
        if (pattern, label) in seen:
            continue
        seen.add((pattern, label))
        rows.append(row)
        ys.append(label)
        groups.append(pattern_ids.setdefault(pattern, len(pattern_ids)))
    return symptoms, np.array(rows), np.array(ys), np.array(groups), len(frame)


def subsets(row, size):
    on = np.flatnonzero(row)
    if len(on) < size:
        return []
    out = []
    for combo in itertools.combinations(on, size):
        vector = np.zeros_like(row)
        vector[list(combo)] = 1
        out.append(vector)
    return out


def augment(rows, labels):
    """Add every smaller subset (MIN_SYMPTOMS..n) of each pattern."""
    extra_x, extra_y = [], []
    for row, label in zip(rows, labels):
        for size in range(MIN_SYMPTOMS, int(row.sum()) + 1):
            for vector in subsets(row, size):
                extra_x.append(vector)
                extra_y.append(label)
    return np.vstack([rows, np.array(extra_x)]), np.concatenate([labels, np.array(extra_y)])


def top_k(model, x, truth, ks=(1, 3, 5)):
    order = np.argsort(-model.predict_proba(x), axis=1)
    classes = np.asarray(model.classes_)
    return {k: float(np.mean([t in classes[o[:k]] for t, o in zip(truth, order)])) for k in ks}


def test_view(rows, labels, size):
    if size is None:
        return rows, labels
    xs, ys = [], []
    for row, label in zip(rows, labels):
        for vector in subsets(row, size):
            xs.append(vector)
            ys.append(label)
    return np.array(xs), np.array(ys)


def evaluate(rows, labels, groups):
    views = {"all_symptoms": None, "any_3_symptoms": 3, "any_2_symptoms": 2}
    scores = {name: {1: [], 3: [], 5: []} for name in views}
    for repeat in range(REPEATS):
        splitter = GroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + repeat)
        for train_idx, test_idx in splitter.split(rows, labels, groups=groups):
            x_train, y_train = augment(rows[train_idx], labels[train_idx])
            model = make_model().fit(x_train, y_train)
            for name, size in views.items():
                x_test, y_test = test_view(rows[test_idx], labels[test_idx], size)
                if len(y_test) == 0:
                    continue
                result = top_k(model, x_test, y_test)
                for k in (1, 3, 5):
                    scores[name][k].append(result[k])
    return {
        name: {f"top_{k}": round(float(np.mean(values)), 4) for k, values in by_k.items()}
        for name, by_k in scores.items()
    }


def main():
    warnings.filterwarnings("ignore")
    data_path = DB_DIR / "indian_disease_binary.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")
    symptoms, rows, labels, groups, raw_rows = load_unique(data_path)
    if len(set(labels)) < 2 or len(rows) < 50:
        raise RuntimeError("Indian binary dataset is too small for a meaningful evaluation")

    print(f"Raw rows: {raw_rows}; unique symptom patterns: {len(set(groups.tolist()))}; "
          f"unique (pattern, disease) rows: {len(rows)}", flush=True)
    print("Evaluating on symptom patterns the model has never seen...", flush=True)
    honest = evaluate(rows, labels, groups)
    for name, values in honest.items():
        print(f"  {name:15s} top-1 {values['top_1']*100:5.1f}%  top-3 {values['top_3']*100:5.1f}%  top-5 {values['top_5']*100:5.1f}%")

    x_all, y_all = augment(rows, labels)
    model = make_model().fit(x_all, y_all)
    report = {
        "dataset": "indian_disease_binary.csv",
        "purpose": "software benchmark only; compact pattern data, not clinical validation",
        "raw_rows": int(raw_rows),
        "unique_symptom_patterns": int(len(set(groups.tolist()))),
        "training_rows_after_dedup_and_subsets": int(len(y_all)),
        "conditions": int(len(set(labels))),
        "symptom_columns": int(len(symptoms)),
        "model": f"LogisticRegression(C={C_VALUE}, class_weight=balanced) trained on every 2..n symptom subset",
        "evaluation": f"{REPEATS} x {FOLDS}-fold CV split by symptom pattern (no pattern is in both train and test)",
        "unseen_pattern_accuracy": honest,
        "note": ("The earlier 99.1% score came from identical rows being in both train and test. "
                 "These numbers are on unseen patterns, but the dataset itself is still a small "
                 "pattern set, not real patient data."),
    }
    INDIAN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "symptoms": symptoms, "conditions": list(model.classes_), "report": report},
                INDIAN_MODEL_PATH)
    report_path = INDIAN_MODEL_PATH.with_name("indian_evaluation.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {INDIAN_MODEL_PATH} and {report_path}")


if __name__ == "__main__":
    main()
