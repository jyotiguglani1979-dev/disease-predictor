"""
src/prep_brfss.py   (Module B: large lifestyle screening, 200k+ people)
------------------------------------------------------------------------
Reads the Kaggle "Indicators of Heart Disease (2022 update)" file, which is
the CDC BRFSS 2022 survey. It has two separate answers we need:
    HadHeartAttack  -> target `heart_attack`
    HadAngina       -> target `chd`  (angina / coronary heart disease)

Ideas taken from CardioSense:
  - ordinal encoding with meaning (health, age) instead of random numbers
  - diabetes: "pre-diabetes" and "only during pregnancy" are NOT full diabetes
  - leave out behaviours that people change AFTER a diagnosis (reverse
    causality). Alcohol is kept in the table but flagged; decide in training.

Important: this data is self-reported, US only. Say so in your report.

Run:  python -m src.prep_brfss
"""
import re
import sys

import numpy as np
import pandas as pd

from config import RAW, BRFSS_FILE, Tables
from src.db import save

REQUIRED = ["HadHeartAttack", "HadAngina", "Sex", "AgeCategory", "BMI"]

BINARY = {   # original column -> new name  (Yes/No -> 1/0)
    "PhysicalActivities": "physical_activity",
    "AlcoholDrinkers": "alcohol",          # reverse-causality risk
    "HadStroke": "stroke",
    "HadCOPD": "copd",
    "HadKidneyDisease": "kidney_disease",
    "HadArthritis": "arthritis",
    "HadDepressiveDisorder": "depression",
}
NUMERIC = {
    "BMI": "bmi",
    "HeightInMeters": "height_m",
    "WeightInKilograms": "weight_kg",
    "SleepHours": "sleep_hours",
}


def yes_no(s: pd.Series) -> pd.Series:
    t = s.astype("string").str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[t.str.startswith("yes", na=False)] = 1.0
    out[t.str.startswith("no", na=False)] = 0.0
    return out


def diabetes(s: pd.Series) -> pd.Series:
    t = s.astype("string").str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[t.str.startswith("no", na=False)] = 0.0
    out[t.str.contains("pre-diabetes|borderline|pregnancy", na=False)] = 1.0
    out[t.str.startswith("yes", na=False) & ~t.str.contains("pregnancy", na=False)] = 2.0
    return out


def smoker(s: pd.Series) -> pd.Series:
    t = s.astype("string").str.lower()
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[t.str.contains("never", na=False)] = 0.0
    out[t.str.contains("former", na=False)] = 1.0
    out[t.str.contains("current", na=False)] = 2.0
    return out


def general_health(s: pd.Series) -> pd.Series:
    m = {"poor": 0, "fair": 1, "good": 2, "very good": 3, "excellent": 4}
    return s.astype("string").str.strip().str.lower().map(m).astype("float")


def age_lower_bound(s: pd.Series) -> pd.Series:
    """'Age 18 to 24' -> 18,  'Age 80 or older' -> 80"""
    return s.astype("string").str.extract(r"(\d+)", expand=False).astype("float")


def main() -> bool:
    print("=" * 60)
    print("Module B - BRFSS 2022 lifestyle screening")
    print("=" * 60)

    path = RAW / BRFSS_FILE
    if not path.exists():
        print(f"  MISSING FILE: {path}  (see README, Step 4)")
        return False

    raw = pd.read_csv(path, low_memory=False)
    print(f"  raw rows: {len(raw):,}")

    absent = [c for c in REQUIRED if c not in raw.columns]
    if absent:
        print("  Columns not found:", absent)
        print("  Columns in your file:", list(raw.columns))
        return False

    df = pd.DataFrame(index=raw.index)
    df["heart_attack"] = yes_no(raw["HadHeartAttack"])
    df["chd"] = yes_no(raw["HadAngina"])
    df["sex"] = raw["Sex"].astype("string").str.lower().map({"male": 1, "female": 0}).astype("float")
    df["age_lower"] = age_lower_bound(raw["AgeCategory"])
    ranks = {v: i for i, v in enumerate(sorted(df["age_lower"].dropna().unique()))}
    df["age_group"] = df["age_lower"].map(ranks)

    for src, dst in NUMERIC.items():
        if src in raw.columns:
            df[dst] = pd.to_numeric(raw[src], errors="coerce")
    for src, dst in BINARY.items():
        if src in raw.columns:
            df[dst] = yes_no(raw[src])
    if "HadDiabetes" in raw.columns:
        df["diabetes"] = diabetes(raw["HadDiabetes"])
    if "SmokerStatus" in raw.columns:
        df["smoker"] = smoker(raw["SmokerStatus"])
    if "GeneralHealth" in raw.columns:
        df["general_health"] = general_health(raw["GeneralHealth"])

    # Both targets must be known
    before = len(df)
    df = df.dropna(subset=["heart_attack", "chd"]).copy()
    df[["heart_attack", "chd"]] = df[["heart_attack", "chd"]].astype(int)
    print(f"  dropped {before - len(df):,} rows with unknown target")
    print(f"  rows kept: {len(df):,}")
    print(f"  heart attack rate: {df['heart_attack'].mean():.1%}")
    print(f"  angina/CHD rate:   {df['chd'].mean():.1%}")
    print("\n  % missing per column:")
    print((df.isna().mean() * 100).round(1).to_string())

    save(df, Tables.BRFSS)
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
