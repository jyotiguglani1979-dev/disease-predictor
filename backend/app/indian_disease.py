import re
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from .config import INDIAN_MODEL_PATH

ALIASES = {
    "high fever": "high fever", "rising fever": "high fever", "fever": "fever",
    "headache": "headache", "weakness": "weakness", "tiredness": "fatigue",
    "fatigue": "fatigue", "stomach pain": "abdominal pain",
    "abdominal pain": "abdominal pain", "belly pain": "abdominal pain",
    "nausea": "nausea", "vomiting": "vomiting", "diarrhea": "diarrhoea",
    "diarrhoea": "diarrhoea", "constipation": "constipation", "cough": "cough",
    "loose stools": "diarrhoea", "loose stool": "diarrhoea",
    "chills": "chills", "shivering": "shivering", "sore throat": "sore throat",
    "throat pain": "sore throat", "muscle pain": "muscle pain",
    "joint pain": "joint pain", "skin rash": "skin rash", "rash": "skin rash",
    "body aches": "muscle pain", "body ache": "muscle pain",
    "yellow eyes": "yellow eyes", "weight loss": "weight loss",
    "sweating": "sweating", "breathlessness": "difficulty breathing",
    "difficulty breathing": "difficulty breathing",
    "shortness of breath": "difficulty breathing", "chest pain": "chest pain",
    "burning urination": "burning micturition",
    "painful urination": "burning micturition",
    "coughing blood": "blood cough", "blood in sputum": "blood cough",
}

# The model only knows the 66 columns of indian_disease_binary.csv. Names the app
# understands but the model has no column for were silently ignored before.
# This maps a reported symptom to the real model column(s); None = no column exists.
MODEL_COLUMNS = {
    "fever": ["high fever"],
    "high fever": ["high fever"],
    "difficulty breathing": ["breathlessness"],
    "breathlessness": ["breathlessness"],
    "abdominal pain": ["abdominal pain", "stomach pain"],
    "weakness": ["fatigue"],
    "fatigue": ["fatigue"],
    "muscle pain": None,
    "sore throat": None,
    "blood cough": None,
    "yellow eyes": None,
}

def normalize(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())

@lru_cache(maxsize=1)
def load_artifact():
    if not INDIAN_MODEL_PATH.exists():
        return None
    return joblib.load(INDIAN_MODEL_PATH)

def extract_indian_symptoms(text: str) -> list[str]:
    folded = " " + normalize(text) + " "
    found = set()
    for phrase, canonical in ALIASES.items():
        if re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", folded):
            found.add(canonical)
    return sorted(found)

def predict_indian(text: str, limit: int = 10) -> dict:
    artifact = load_artifact()
    reported = extract_indian_symptoms(text)
    if artifact is None:
        return {"ready": False, "results": [], "reported_symptoms": reported}
    symptom_names = artifact["symptoms"]
    known = set(symptom_names)
    active, not_modelled = set(), []
    for symptom in reported:
        columns = MODEL_COLUMNS.get(symptom, [symptom])
        columns = [c for c in (columns or []) if c in known]
        if columns:
            active.update(columns)
        else:
            not_modelled.append(symptom)
    # Nothing the model understands was reported: do not rank diseases from an empty input.
    if not active:
        return {"ready": True, "results": [], "reported_symptoms": reported,
                "not_modelled_symptoms": not_modelled,
                "model": "binary Indian symptom prototype", "limit": limit}
    vector = np.asarray([[1.0 if symptom in active else 0.0 for symptom in symptom_names]])
    model = artifact["model"]
    probabilities = model.predict_proba(vector)[0]
    ranked = sorted(zip(model.classes_, probabilities), key=lambda item: item[1], reverse=True)[:limit]
    # Scores are the model's own probabilities (0-100). They are not calibrated for real patients.
    results = [{"condition": str(condition).strip(), "score": round(float(value) * 100.0, 1)}
               for condition, value in ranked]
    return {"ready": True, "results": results, "reported_symptoms": reported,
            "not_modelled_symptoms": not_modelled,
            "model": "binary Indian symptom prototype", "limit": limit}
