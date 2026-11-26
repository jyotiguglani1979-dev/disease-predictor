"""Reproducible DDXPlus benchmark and artifact builder.

Usage: python -m backend.train_ddxplus [--train-sample 50000 --valid-sample 12000 --test-sample 16000]
All sets are sampled deterministically from their official split files; patients are never
pooled across splits. DIFFERENTIAL_DIAGNOSIS and INITIAL_EVIDENCE are excluded from features.
"""
import argparse, ast, csv, json, random, time
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from backend.app.config import DB_DIR, MODEL_PATH, ROOT
from backend.app.clinical import EVIDENCE_BY_CODE

SEED = 2026
BINARY_CODES = [code for code, info in EVIDENCE_BY_CODE.items() if info.get("data_type") == "B"]

def row_features(row, rng=None):
    d = {"age": float(row["AGE"]), "sex:" + row["SEX"]: 1.0}
    evidence = set(ast.literal_eval(row["EVIDENCES"]))
    keep_rate = rng.uniform(0.2, 0.85) if rng is not None else 1.0
    for code in BINARY_CODES:
        if rng is None or rng.random() < keep_rate:
            d[("evidence:" if code in evidence else "negative:") + code] = 1.0
    for item in sorted(evidence):
        if item in BINARY_CODES: continue
        if rng is not None and rng.random() >= keep_rate: continue
        d["evidence:" + item] = 1.0
    return d

def reservoir(path: Path, limit: int, seed: int):
    rng = random.Random(seed); sample = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if len(sample) < limit:
                sample.append(row)
            else:
                j = rng.randrange(i + 1)
                if j < limit: sample[j] = row
    return sample

def arrays(rows, vectorizer=None, mask_seed=None):
    rng = random.Random(mask_seed) if mask_seed is not None else None
    dictionaries = [row_features(r, rng) for r in rows]
    y = np.asarray([r["PATHOLOGY"] for r in rows])
    if vectorizer is None:
        vectorizer = DictVectorizer(sparse=True)
        x = vectorizer.fit_transform(dictionaries)
        if x.shape[1] < 2**31 and x.nnz < 2**31:
            x.indices = x.indices.astype(np.int32, copy=False); x.indptr = x.indptr.astype(np.int32, copy=False)
        return x, y, vectorizer
    x = vectorizer.transform(dictionaries)
    if x.shape[1] < 2**31 and x.nnz < 2**31:
        x.indices = x.indices.astype(np.int32, copy=False); x.indptr = x.indptr.astype(np.int32, copy=False)
    return x, y

def metrics(model, x, y):
    pred = model.predict(x)
    labels = sorted(set(y) | set(pred))
    top_k = {}
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x)
        classes = list(model.classes_)
    elif hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(x))
        classes = list(model.classes_)
        if raw.ndim == 1:
            raw = np.column_stack((-raw, raw))
        scores = raw
    else:
        scores = None
        classes = []
    if scores is not None:
        for k in (1, 5):
            count = min(k, len(classes))
            top_indices = np.argsort(scores, axis=1)[:, -count:]
            top_labels = [[classes[index] for index in row] for row in top_indices]
            top_k[f"top_{k}_accuracy"] = float(sum(label in top for label, top in zip(y.tolist(), top_labels)) / len(y))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        **top_k,
        "confusion_labels": labels,
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "classification_report": classification_report(y, pred, output_dict=True, zero_division=0),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-sample", type=int, default=50000)
    p.add_argument("--valid-sample", type=int, default=12000)
    p.add_argument("--test-sample", type=int, default=16000)
    a = p.parse_args(); started = time.time()
    paths = {"train": DB_DIR / "release_train_patients", "validation": DB_DIR / "release_validate_patients",
             "test": DB_DIR / "release_test_patients"}
    # The user's supplied test split has a trailing " 1" in its filename; preserve it in place.
    if not paths["test"].exists() and (DB_DIR / "release_test_patients 1").exists():
        paths["test"] = DB_DIR / "release_test_patients 1"
    for name, path in paths.items():
        if not path.exists(): raise FileNotFoundError(f"Expected DDXPlus {name} split: {path}")
    print("Sampling official splits (patient rows never cross between splits)...", flush=True)
    train_rows = reservoir(paths["train"], a.train_sample, SEED)
    valid_rows = reservoir(paths["validation"], a.valid_sample, SEED + 1)
    test_rows = reservoir(paths["test"], a.test_sample, SEED + 2)
    _, y_train, vec = arrays(train_rows)
    # Symptom assessments begin with only a subset of each patient's findings. Train against
    # deterministic random evidence masks to reduce the full-record/partial-conversation gap.
    x_train, y_train = arrays(train_rows, vec, mask_seed=SEED + 3)
    x_val_full, y_val = arrays(valid_rows, vec)
    x_val, _ = arrays(valid_rows, vec, mask_seed=SEED + 4)
    x_test_full, y_test = arrays(test_rows, vec)
    x_test_partial, _ = arrays(test_rows, vec, mask_seed=SEED + 5)
    candidates = {
        "decision_tree": DecisionTreeClassifier(max_depth=30, min_samples_leaf=3, class_weight="balanced", random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=24, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
    }
    for c_value in (0.1, 0.25, 0.5, 1.0):
        candidates[f"linear_svm_C_{c_value:g}"] = LinearSVC(C=c_value, class_weight="balanced", random_state=SEED, dual="auto")
    results = {}
    for name, model in candidates.items():
        fit_x, fit_y = x_train, y_train
        val_x = x_val
        val_full_x = x_val_full
        eval_y = y_val
        print(f"Fitting {name} on {len(fit_y):,} rows...", flush=True)
        model.fit(fit_x, fit_y)
        results[name] = {"validation_rows": int(len(eval_y)), "validation_partial_evidence": metrics(model, val_x, eval_y),
                         "validation_full_evidence": metrics(model, val_full_x, eval_y), "model": model}
        print(f"{name}: partial-evidence validation macro-F1={results[name]['validation_partial_evidence']['f1_macro']:.4f}", flush=True)
    winner = max(results, key=lambda name: results[name]["validation_partial_evidence"]["f1_macro"])
    model = results[winner]["model"]
    test_x = x_test_partial
    test_full_x = x_test_full
    test_metrics_partial = metrics(model, test_x, y_test)
    test_metrics_full = metrics(model, test_full_x, y_test)
    compact = {name: {"rows": value["validation_rows"], "partial_evidence": value["validation_partial_evidence"], "full_evidence": value["validation_full_evidence"]} for name, value in results.items()}
    report = {
        "dataset": "DDXPlus English release, local workspace copy",
        "split_files": {k: str(v.relative_to(ROOT)) for k, v in paths.items()},
        "split_sampling": {"method": "deterministic reservoir sample from each official split", "seed": SEED,
                           "train_rows": len(train_rows), "validation_rows": len(valid_rows), "test_rows": len(test_rows)},
        "sampled_class_distribution": {"train": dict(Counter(y_train.tolist())), "validation": dict(Counter(y_val.tolist())), "test": dict(Counter(y_test.tolist()))},
        "feature_policy": "AGE, SEX, observed EVIDENCES, and explicit binary absences from each complete synthetic patient record; PATHOLOGY target; DIFFERENTIAL_DIAGNOSIS and INITIAL_EVIDENCE excluded",
        "selection_metric": "validation macro-F1 on random partial-evidence inputs",
        "selected_model": winner,
        "evidence_masking": "Training and partial-evidence evaluation independently reveal each binary yes/no state and each observed categorical value at a per-patient random rate from 20% to 85%; unasked answers remain unknown at inference. No patient rows are shared across official splits.",
        "validation_metrics": compact,
        "held_out_test_metrics": {"partial_evidence": test_metrics_partial, "full_evidence": test_metrics_full},
        "elapsed_seconds": round(time.time() - started, 1),
        "score_note": "Classifier outputs are model scores, not clinically calibrated probabilities. DDXPlus is synthetic and is not validated for clinical decisions.",
    }
    artifact = {"model": model, "vectorizer": vec, "model_name": winner, "conditions": sorted(set(y_train)), "trained_rows": len(train_rows), "report": report}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    report_path = MODEL_PATH.with_name("ddxplus_evaluation.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected {winner}; held-out partial-evidence test macro-F1={test_metrics_partial['f1_macro']:.4f}", flush=True)
    print(f"Saved {MODEL_PATH} and {report_path}", flush=True)

if __name__ == "__main__": main()
