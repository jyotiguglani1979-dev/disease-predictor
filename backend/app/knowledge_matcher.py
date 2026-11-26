"""Symptom association lookup over the three supplied CSVs; not a predictor."""
import re
import math
from functools import lru_cache
from pathlib import Path
import pandas as pd
from .config import DB_DIR

FILES = ["Disease and symptoms dataset.csv", "improved_disease_dataset.csv", "symtoms_df.csv"]
# An initial audit found that this file assigns many unrelated binary findings
# to most diagnoses with similar near-random rates; retain it on disk/status,
# but do not let its labels create misleading matches until its provenance is verified.
EXCLUDED_FILES = {"improved_disease_dataset.csv"}

def norm(value):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split())

def _read(path):
    return pd.read_csv(path)

@lru_cache(maxsize=1)
def load_associations():
    associations = {}
    for filename in FILES:
        if filename in EXCLUDED_FILES:
            continue
        path = DB_DIR / filename
        if not path.exists():
            continue
        sample = pd.read_csv(path, nrows=2)
        disease_col = next((c for c in sample.columns if norm(c) in {"disease", "diseases", "prognosis"}), None)
        if not disease_col:
            continue
        symptom_cols = [c for c in sample.columns if c != disease_col and not norm(c).startswith("unnamed")]
        # Read in bounded chunks; the largest supplied file is about 182 MB.
        for frame in pd.read_csv(path, chunksize=20000):
            wide = all(pd.api.types.is_numeric_dtype(frame[c]) for c in symptom_cols)
            if wide:
                for disease, group in frame.groupby(disease_col, sort=False):
                    label = str(disease).strip()
                    if not label or label.lower() == "nan": continue
                    active = group[symptom_cols].fillna(0).sum(axis=0)
                    symptoms = {norm(col): int(value) for col, value in active.items() if value > 0}
                    if symptoms:
                        key = (filename, norm(label))
                        item = associations.setdefault(key, {"disease": label, "symptom_counts": {}, "rows": 0, "source": filename})
                        item["rows"] += len(group)
                        for term, count in symptoms.items(): item["symptom_counts"][term] = item["symptom_counts"].get(term, 0) + count
            else:
                for disease, group in frame.groupby(disease_col, sort=False):
                    label = str(disease).strip()
                    if not label or label.lower() == "nan": continue
                    terms = [norm(value) for value in group[symptom_cols].to_numpy().ravel()
                             if pd.notna(value) and norm(value) not in {"", "nan", "none"}]
                    counts = pd.Series(terms, dtype="object").value_counts().to_dict()
                    if counts:
                        key = (filename, norm(label))
                        item = associations.setdefault(key, {"disease": label, "symptom_counts": {}, "rows": 0, "source": filename})
                        item["rows"] += len(group)
                        for term, count in counts.items(): item["symptom_counts"][term] = item["symptom_counts"].get(term, 0) + int(count)
    return list(associations.values())

_REPORTED_ALIASES = {
    "body ache": "muscle pain", "body aches": "muscle pain", "bodyache": "muscle pain",
    "bodyaches": "muscle pain", "sore muscles": "muscle pain", "aching muscles": "muscle pain",
    "myalgia": "muscle pain", "myalgias": "muscle pain",
    "head pain": "headache", "pain in my head": "headache", "my head hurts": "headache",
}
_NEGATION = re.compile(r"(?:\bno\b|\bnot\b|\bnever\b|\bwithout\b|don t|doesn t|didn t|do not)(?:\s+\w+){0,4}\s*$")

@lru_cache(maxsize=1)
def _reported_vocabulary():
    """Build the CSV vocabulary once; assessments reuse this small in-memory index."""
    vocabulary = {term for row in load_associations() for term in row["symptom_counts"]}
    vocabulary.update(_REPORTED_ALIASES)
    return tuple(sorted((term for term in vocabulary if term), key=lambda term: (-len(term), term)))

@lru_cache(maxsize=4096)
def _reported_pattern(term):
    return re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])")

def extract_reported_symptoms(text):
    """Extract affirmative CSV symptom terms from free text, without treating them as DDX evidence."""
    folded = " " + norm(text) + " "
    found = set()
    for term in _reported_vocabulary():
        if not term or f" {term} " not in folded:
            continue
        states = []
        for match in _reported_pattern(term).finditer(folded):
            context = folded[max(0, match.start() - 48):match.start()]
            context = re.split(r"\b(?:but|however|except)\b", context)[-1]
            states.append(not bool(_NEGATION.search(context)))
        if states and all(states):
            found.add(_REPORTED_ALIASES.get(term, term))
    return sorted(found)

def match_conditions(symptoms, limit=6):
    query = set()
    for symptom in symptoms:
        term = norm(symptom)
        if not term: continue
        query.add(term)
    # Ignore generic words during fuzzy matching; otherwise "body aches" can
    # match unrelated findings just because their DDXPlus question says "body".
    generic = {"pain", "ache", "aches", "body", "muscle", "symptom", "symptoms", "have", "had", "your", "been", "feel", "does", "with", "from", "this", "that", "often", "related", "severe", "diffuse", "widespread", "generalized", "vague", "annoying", "areas", "area", "one", "both", "more", "anywhere", "currently", "ever", "noticed", "change", "consultation"}
    if not query: return []
    aggregate = {}
    for row in load_associations():
        # Match a specific term like "fever" against variants such as "high fever".
        overlap = set()
        for term in row["symptom_counts"]:
            row_tokens = set(term.split()) - generic
            matched = any(term == phrase or (term not in generic and len(phrase) >= 5 and (f" {term} " in f" {phrase} " or f" {phrase} " in f" {term} "))
                          for phrase in query)
            if not matched and row_tokens:
                matched = any(len(token) >= 5 and token in phrase.split() for token in row_tokens for phrase in query)
            if matched:
                overlap.add(term)
        if not overlap: continue
        key = norm(row["disease"])
        item = aggregate.setdefault(key, {"condition": row["disease"], "matched_symptoms": set(), "sources": set(), "specificity": {}})
        item["matched_symptoms"].update(overlap)
        for term in overlap:
            frequency = row["symptom_counts"][term] / max(row["rows"], 1)
            item["specificity"][term] = max(item["specificity"].get(term, 0), -math.log(max(frequency, 0.001)))
        item["sources"].add(row["source"])
    ranked = sorted(aggregate.values(), key=lambda x: (-len(x["matched_symptoms"]), -sum(x["specificity"].values()), x["condition"].casefold()))
    matches = [{"condition": x["condition"], "matched_symptoms": sorted(x["matched_symptoms"]),
                "sources": sorted(x["sources"]), "label": "Knowledge-base match — not an ML prediction"}
               for x in ranked]

    if {"fever", "high fever"} & query:
        typhoid_terms = {"headache", "head pain", "stomach pain", "abdominal pain",
                         "weakness", "fatigue", "nausea", "vomiting", "diarrhea",
                         "constipation", "muscle pain"}
        overlap = sorted(typhoid_terms & query)
        if len(overlap) >= 3:
            matches.insert(0, {
                "condition": "Typhoid fever",
                "matched_symptoms": overlap,
                "sources": ["WHO: Typhoid fact sheet", "NCDC India: National Treatment Guidelines — Enteric Fever"],
                "label": "Reference-guided outside-model consideration",
            })

    unique = {}
    for item in matches:
        key = norm(item["condition"])
        if key not in unique:
            unique[key] = item
    return list(unique.values())[:limit]

def dataset_status():
    return {name: {"present": (DB_DIR / name).exists(),
                   "used_for_matching": name not in EXCLUDED_FILES}
            for name in FILES}

def association_count():
    return len(load_associations())
