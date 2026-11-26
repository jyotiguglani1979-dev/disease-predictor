"""
src/prep_framingham.py   (Module C: 10-year heart disease risk)
----------------------------------------------------------------
Framingham follow-up data: each person was measured once, then followed
for 10 years. Target = did they get coronary heart disease within 10 years.
This is the only dataset here that is truly about the FUTURE.

Missing values are kept (filled later inside training).

Run:  python -m src.prep_framingham
"""
import sys

import pandas as pd

from config import RAW, FRAMINGHAM_FILE, Tables
from src.db import save

RENAME = {
    "male": "sex",                      # 1 = male
    "currentSmoker": "smoker",
    "cigsPerDay": "cigs_per_day",
    "BPMeds": "bp_meds",
    "prevalentStroke": "prev_stroke",
    "prevalentHyp": "prev_hypertension",
    "totChol": "total_chol",
    "sysBP": "sys_bp",
    "diaBP": "dia_bp",
    "heartRate": "heart_rate",
    "TenYearCHD": "target",
}
REQUIRED = list(RENAME) + ["age", "diabetes", "BMI", "glucose"]


def main() -> bool:
    print("=" * 60)
    print("Module C - Framingham (10-year risk)")
    print("=" * 60)

    path = RAW / FRAMINGHAM_FILE
    if not path.exists():
        print(f"  MISSING FILE: {path}  (see README, Step 4)")
        return False

    df = pd.read_csv(path)
    absent = [c for c in REQUIRED if c not in df.columns]
    if absent:
        print("  Columns not found:", absent)
        print("  Columns in your file:", list(df.columns))
        return False

    df = df.rename(columns=RENAME).rename(columns={"BMI": "bmi"})
    df = df.drop(columns=[c for c in ["education"] if c in df.columns])

    print(f"  rows: {len(df):,}")
    print(f"  10-year heart disease rate: {df['target'].mean():.1%}")
    print("\n  % missing per column (only those with missing):")
    miss = (df.isna().mean() * 100).round(1)
    print(miss[miss > 0].to_string())

    save(df, Tables.FRAMINGHAM)
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
