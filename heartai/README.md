# HeartAI - setup guide (Phase 1: folder + data)

Inspired by two open projects: **CardioPredict** (Aamir-J) for the structure
(database -> clean -> model -> SHAP -> dashboard) and **CardioSense**
(Surieyaa, MIT) for the methods (calibration, reverse-causality check).
All code here is newly written. Credit both repos in your report.

## The 3 modules

| Module | Question | Data | Rows |
|---|---|---|---|
| A | Heart disease right now (clinical) | UCI Heart Disease, 4 hospitals | 920 |
| B | Heart attack + angina/CHD from lifestyle (large) | BRFSS 2022 (Kaggle) | ~240k |
| C | Heart attack/CHD in next 10 years | Framingham | ~4.2k |

## Step 1 - Install tools (one time)
1. Python 3.11 or 3.12: https://www.python.org/downloads/  (tick **Add Python to PATH**)
2. VS Code: https://code.visualstudio.com/

## Step 2 - Put the project folder on your computer
Unzip `heartai.zip`. You get a folder `heartai` like this:

```
heartai/
  config.py
  requirements.txt
  src/            (prep_uci.py, prep_brfss.py, prep_framingham.py, prep_all.py, db.py)
  data/raw/       <- you put downloads here
  data/raw/uci/   <- UCI files go here
  data/processed/ (auto-created database goes here)
  models/         (trained models go here later)
```

## Step 3 - Install libraries
Open the `heartai` folder in VS Code -> Terminal -> New Terminal, then:

```
python -m venv venv
venv\Scripts\activate          (Windows)      source venv/bin/activate   (Mac/Linux)
pip install -r requirements.txt
```

## Step 4 - Download the 3 datasets (free)

**A) UCI Heart Disease** -> https://archive.ics.uci.edu/dataset/45/heart+disease
Click **Download**, unzip, and copy these 4 files into `data/raw/uci/`:
- `processed.cleveland.data`
- `processed.hungarian.data`
- `processed.switzerland.data`
- `processed.va.data`

**B) BRFSS 2022 (Kaggle, free account needed)**
https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
Download, unzip, and copy **`heart_2022_with_nans.csv`** into `data/raw/`.

**C) Framingham (Kaggle)**
https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression
Download, unzip, and copy **`framingham.csv`** into `data/raw/`.

(If a file has a slightly different name, change it in `config.py`.)

## Step 5 - Clean everything into one database
```
python -m src.prep_all
```
You should see 3 lines saying `OK` and 3 tables created:
`clinical_uci`, `screening_brfss`, `risk_framingham`.
Expected numbers: UCI 920 rows, BRFSS roughly 240,000 rows, Framingham 4,240 rows.

If you see `Columns not found`, copy that message to Claude. The Kaggle files
sometimes change column names.

## Step 6 - Tell me it worked
Send me the printed output of Step 5. Next phases:
- Phase 2: feature engineering + train models (XGBoost, calibration, hospital-wise test)
- Phase 3: SHAP explanations + risk calculator dashboard (Streamlit)
- Phase 4: combine into one "Heart Health" result + safety text (call 108)

## Safety
Research project, not a medical device. Never tell a user a heart attack will
happen at a certain time. Chest pain or breathlessness: show "call 108" first.
