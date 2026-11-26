"""
config.py
---------
One place for all paths, file names and table names.
Change a file name here if your downloaded file is named differently.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
DB_PATH = PROCESSED / "heartai.db"

for _p in (RAW, RAW / "uci", PROCESSED, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

# --- Raw file names (put these inside data/raw/ as shown in README) ---
UCI_FILES = {                       # data/raw/uci/
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va_long_beach": "processed.va.data",
}
FRAMINGHAM_FILE = "framingham.csv"          # data/raw/
BRFSS_FILE = "heart_2022_with_nans.csv"     # data/raw/

RANDOM_STATE = 42


class Tables:
    UCI = "clinical_uci"            # Module A: current heart disease (clinical)
    BRFSS = "screening_brfss"       # Module B: large lifestyle screening
    FRAMINGHAM = "risk_framingham"  # Module C: 10-year risk
