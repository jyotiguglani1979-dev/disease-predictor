"""
src/prep_uci.py   (Module A: current heart disease, clinical data)
-------------------------------------------------------------------
Combines the 4 UCI hospitals (Cleveland, Hungary, Switzerland, VA Long Beach)
= 920 patients into one clean table.

Design ideas taken from CardioPredict:
  - raw -> clean table in SQLite
  - cholesterol == 0 is impossible, so treat it as missing
But we do NOT fill missing values here. Filling is done later inside model
training (so test data never leaks into the fill values).
We also keep a `site` column so we can later test: train on 3 hospitals,
test on the 4th. That is a much more honest test than a random split.

Run:  python -m src.prep_uci
"""
import sys

import numpy as np
import pandas as pd

from config import RAW, UCI_FILES, Tables
from src.db import save

COLUMNS = [
    "age", "sex", "chest_pain", "resting_bp", "cholesterol", "fasting_sugar",
    "rest_ecg", "max_hr", "exercise_angina", "oldpeak", "st_slope",
    "vessels", "thal", "num",
]


def main() -> bool:
    print("=" * 60)
    print("Module A - UCI Heart Disease (4 hospitals)")
    print("=" * 60)

    frames, missing = [], []
    for site, fname in UCI_FILES.items():
        path = RAW / "uci" / fname
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
        df["site"] = site
        frames.append(df)

    if missing:
        print("  MISSING FILES (see README, Step 4):")
        for m in missing:
            print("   -", m)
        return False

    df = pd.concat(frames, ignore_index=True)

    # Impossible zeros -> missing
    df.loc[df["resting_bp"] == 0, "resting_bp"] = np.nan
    df.loc[df["cholesterol"] == 0, "cholesterol"] = np.nan

    # num = 0 (no disease) ... 4 (severe). We use: any disease = 1
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")

    print(f"  total patients: {len(df):,}   (expected 920)")
    print("\n  patients and disease rate per hospital:")
    print(df.groupby("site")["target"].agg(["count", "mean"]).round(3).to_string())
    print("\n  % missing per column:")
    print((df.isna().mean() * 100).round(1).to_string())

    save(df, Tables.UCI)
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
