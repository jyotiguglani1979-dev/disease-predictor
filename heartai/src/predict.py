"""
src/predict.py   (the combined Heart Health result)
---------------------------------------------------
    from src.predict import predict_heart_health
    result = predict_heart_health({"age": 55, "sex": "male", ...})

One input form -> up to 4 results:
    disease_now             (Module A, needs clinical test values)
    heart_attack_screening  (Module B, lifestyle + history)
    angina_chd_screening    (Module B)
    ten_year_risk           (Module C, needs BP + cholesterol)

Every result gives:
    probability + level (Low / Moderate / High) + "about 6 in 10,000 people like you"
    habits       : what-if numbers, e.g. "a non-smoker like you: 0.04% (yours 0.06%)"
    best_case    : the same person with healthy habits, and how many times higher the risk is
    raising_risk / lowering_risk : technical SHAP view (compared with the average person in the data)

WHY THIS VERSION WAS CHANGED (Step 7b)
  1. Module B used isotonic calibration. Isotonic is a STEP function: for young people many
     different profiles all land on the same step (even exactly 0.0%), so a smoker and a
     non-smoker aged 20 looked identical. Calibration gave no accuracy gain for Module B
     (Brier 0.0458 before vs 0.0460 after), so Module B now shows the model's own probability.
  2. SHAP bars are measured against the AVERAGE adult in the data (about 50 years old). For a
     20-year-old that exaggerates some bars and hides others, and normal people cannot read them.
     The main view now compares the person with THE SAME PERSON WITH HEALTHY HABITS instead.

STEP 8 CHANGES (evidence from benchmarks on real data)
  * New clinical cross-check: ACC/AHA Pooled Cohort Equations (needs HDL). On simulated adults the
    Framingham model ranks people like the PCE (rank agreement 0.93) but its numbers run 2 to 3
    times higher, because the Framingham-era cohort had much more heart disease than people today.
  * Framingham risk bands were rescaled (15% / 37% instead of 5% / 20%) so they match the clinical
    bands for 85% of simulated adults (27% before). When HDL is given, the banner uses the PCE.
  * Age guard: the 10-year model needs age 30 to 79 (study had ages 32 to 70).
  * Impossible inputs are ignored with a warning; blood-pressure crisis gives a hard-stop alert.

First time only:   python -m src.predict --build     (saves SHAP background data)
Try it:            python -m src.predict --demo
"""
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
import shap

from config import MODELS, Tables
from src.clinical import clean_inputs, pce_out_of_range, pce_risk, safety_alerts
from src.db import load

warnings.filterwarnings("ignore")

EMERGENCY = ("If you have chest pain or pressure, pain spreading to the arm, jaw or back, "
             "sudden breathlessness, cold sweat or fainting RIGHT NOW: stop and call 108 "
             "(or 112) immediately. Do not wait for this tool.")
DISCLAIMER = ("Research project, not a medical device. It gives estimated risk, never a "
              "diagnosis and never a time of a heart attack. Please see a doctor for advice.")

FRIENDLY = {
    "age": "Age", "age_group": "Age", "sex": "Sex (male)", "bmi": "BMI",
    "sleep_hours": "Sleep hours", "physical_activity": "Physically active",
    "stroke": "Past stroke", "copd": "Lung disease (COPD)", "kidney_disease": "Kidney disease",
    "arthritis": "Arthritis", "depression": "Depression", "diabetes": "Diabetes",
    "smoker": "Smoking", "general_health": "Self-rated health",
    "chest_pain": "Chest pain type", "resting_bp": "Resting blood pressure",
    "cholesterol": "Cholesterol", "fasting_sugar": "High fasting sugar",
    "rest_ecg": "Resting ECG", "max_hr": "Max heart rate", "exercise_angina": "Chest pain on exercise",
    "oldpeak": "ST depression (oldpeak)", "st_slope": "ST slope",
    "cigs_per_day": "Cigarettes per day", "bp_meds": "BP medicine",
    "prev_stroke": "Past stroke", "prev_hypertension": "High blood pressure history",
    "total_chol": "Cholesterol", "sys_bp": "Systolic BP", "dia_bp": "Diastolic BP",
    "heart_rate": "Resting heart rate", "glucose": "Blood sugar",
}

# short names used in the overall banner ("... is raising it: smoking, extra weight")
HABIT_WORDS = {"smoking": "smoking", "activity": "not being physically active",
               "weight": "extra weight", "sleep": "sleep", "bp": "blood pressure",
               "cholesterol": "cholesterol"}


# ----------------------------------------------------------------------------
# Input handling: ONE form for the user, mapped to each module's columns
# ----------------------------------------------------------------------------
def num(inp, key):
    v = inp.get(key)
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return np.nan
    if key == "sex" and isinstance(v, str):
        return {"male": 1.0, "m": 1.0, "female": 0.0, "f": 0.0}.get(v.strip().lower(), np.nan)
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def age_to_group(age):
    """Same 13 age groups as BRFSS: 18-24, 25-29, ..., 75-79, 80+"""
    if np.isnan(age):
        return np.nan
    if age < 25:
        return 0.0
    if age >= 80:
        return 12.0
    return float(1 + int((age - 25) // 5))


def build_A(i):
    gluc = num(i, "glucose")
    fs = num(i, "fasting_sugar")
    if np.isnan(fs) and not np.isnan(gluc):
        fs = float(gluc > 120)
    return {"age": num(i, "age"), "sex": num(i, "sex"), "chest_pain": num(i, "chest_pain"),
            "resting_bp": num(i, "sys_bp"), "cholesterol": num(i, "total_chol"),
            "fasting_sugar": fs, "rest_ecg": num(i, "rest_ecg"), "max_hr": num(i, "max_hr"),
            "exercise_angina": num(i, "exercise_angina"), "oldpeak": num(i, "oldpeak"),
            "st_slope": num(i, "st_slope")}


def build_B(i):
    return {"sex": num(i, "sex"), "age_group": age_to_group(num(i, "age")), "bmi": num(i, "bmi"),
            "sleep_hours": num(i, "sleep_hours"), "physical_activity": num(i, "physical_activity"),
            "stroke": num(i, "stroke"), "copd": num(i, "copd"),
            "kidney_disease": num(i, "kidney_disease"), "arthritis": num(i, "arthritis"),
            "depression": num(i, "depression"), "diabetes": num(i, "diabetes_status"),
            "smoker": num(i, "smoking_status"), "general_health": num(i, "general_health")}


def _hypertension(i):
    """High-blood-pressure history: ticked by the person, OR readings of 140/90 or more,
    OR taking BP medicine (this is how the Framingham column is defined)."""
    ticked, meds = num(i, "hypertension"), num(i, "bp_meds")
    sbp, dbp = num(i, "sys_bp"), num(i, "dia_bp")
    if ticked == 1 or meds == 1 or (not np.isnan(sbp) and sbp >= 140) or (not np.isnan(dbp) and dbp >= 90):
        return 1.0
    return 0.0 if not np.isnan(ticked) else np.nan


def build_C(i):
    smk, dia = num(i, "smoking_status"), num(i, "diabetes_status")
    cigs = num(i, "cigs_per_day")
    if smk in (0.0, 1.0):                      # never / former smoker -> 0 a day
        cigs = 0.0
    elif smk == 2.0 and (np.isnan(cigs) or cigs <= 0):
        cigs = 10.0                            # current smoker who gave no number: assume 10 a day
    return {"sex": num(i, "sex"), "age": num(i, "age"),
            "smoker": np.nan if np.isnan(smk) else float(smk == 2), "cigs_per_day": cigs,
            "bp_meds": num(i, "bp_meds"), "prev_stroke": num(i, "stroke"),
            "prev_hypertension": _hypertension(i),
            "diabetes": np.nan if np.isnan(dia) else float(dia == 2),
            "total_chol": num(i, "total_chol"), "sys_bp": num(i, "sys_bp"),
            "dia_bp": num(i, "dia_bp"), "bmi": num(i, "bmi"),
            "heart_rate": num(i, "resting_heart_rate"), "glucose": num(i, "glucose")}


# ----------------------------------------------------------------------------
# "What if" levers: things a person can change.  Each lever is tested by changing
# ONLY that thing in the person's own row and asking the model again.
# We never tell someone a change helps unless the model really says it lowers the risk.
# (Former smokers get no smoking lever: in survey data people often quit AFTER a heart
#  problem, so "former smoker" can look worse than "current smoker".)
# ----------------------------------------------------------------------------
def _val(r, key):
    v = r.get(key, np.nan)
    return np.nan if v is None else v


def levers_none(row):
    return []


def levers_B(row):
    r, L = row.iloc[0], []
    if _val(r, "smoker") == 2:
        L.append(dict(key="smoking", title="Smoking", compare="A non-smoker like you",
                      changes={"smoker": 0.0}))
    if _val(r, "physical_activity") == 0:
        L.append(dict(key="activity", title="Physical activity", compare="Someone like you who is active",
                      changes={"physical_activity": 1.0}))
    if _val(r, "bmi") >= 25:
        L.append(dict(key="weight", title="Weight", compare="Someone like you at a healthy weight (BMI 22)",
                      changes={"bmi": 22.0}))
    sl = _val(r, "sleep_hours")
    if sl < 6 or sl > 9:
        L.append(dict(key="sleep", title="Sleep", compare="Someone like you sleeping 7 hours",
                      changes={"sleep_hours": 7.0}))
    return L


def levers_C(row):
    r, L = row.iloc[0], []
    if _val(r, "smoker") == 1:
        L.append(dict(key="smoking", title="Smoking", compare="A non-smoker like you",
                      changes={"smoker": 0.0, "cigs_per_day": 0.0}))
    sbp, dbp = _val(r, "sys_bp"), _val(r, "dia_bp")
    if sbp > 120 or dbp > 80:
        L.append(dict(key="bp", title="Blood pressure", compare="Someone like you with blood pressure 120/80",
                      changes={"sys_bp": min(sbp, 120.0) if not np.isnan(sbp) else 120.0,
                               "dia_bp": min(dbp, 80.0) if not np.isnan(dbp) else 80.0}))
    if _val(r, "total_chol") > 200:
        L.append(dict(key="cholesterol", title="Cholesterol", compare="Someone like you with cholesterol 190",
                      changes={"total_chol": 190.0}))
    if _val(r, "bmi") >= 25:
        L.append(dict(key="weight", title="Weight", compare="Someone like you at a healthy weight (BMI 22)",
                      changes={"bmi": 22.0}))
    return L



# ----------------------------------------------------------------------------
# Clinical cross-check: ACC/AHA Pooled Cohort Equations (needs HDL; age 40-79)
# ----------------------------------------------------------------------------
PCE_TITLE = "10-year heart attack / stroke risk (clinical calculator)"
PCE_PHRASE = "have a heart attack, stroke or die of heart disease within 10 years"
PCE_NOTE = ("ACC/AHA Pooled Cohort Equations (2013), the calculator doctors use. Built on US "
            "data. Guidelines say South Asian ancestry can mean higher risk than this number.")
PCE_ODD_NOTE = (" Caution: for a very high total/HDL cholesterol ratio (6 or more) or age 78 and over, "
                "the published equation is less reliable (small odd effects such as risk dipping "
                "slightly with age), so treat the number as rough.")
PCE_BANDS = (0.05, 0.20)                       # under 5% / 5-20% / 20% and above


def _pce_scorer(df):
    return np.array([pce_risk(r.age, r.male, r.sbp, r.tc, r.hdl, r.smoker, r.diabetes, r.treated)
                     for r in df.itertuples()])


def run_pce(inp):
    """Returns module dict (status done/skipped) and its level or None."""
    need = ["age", "sex", "sys_bp", "total_chol", "hdl_chol"]
    missing = [k for k in need if np.isnan(num(inp, k))]
    if missing:
        return {"title": PCE_TITLE, "status": "skipped", "needs_more_inputs": missing}, None
    age, sbp, tc, hdl = (num(inp, k) for k in ("age", "sys_bp", "total_chol", "hdl_chol"))
    bad = pce_out_of_range(age, sbp, tc, hdl)
    if bad:
        return {"title": PCE_TITLE, "status": "skipped", "needs_more_inputs": [], "reason": bad}, None
    smk, dia = num(inp, "smoking_status"), num(inp, "diabetes_status")
    row = pd.DataFrame([{"age": age, "male": num(inp, "sex"), "sbp": sbp, "tc": tc, "hdl": hdl,
                         "smoker": 0.0 if np.isnan(smk) else float(smk == 2),
                         "diabetes": 0.0 if np.isnan(dia) else float(dia == 2),
                         "treated": bool(num(inp, "bp_meds") == 1)}])
    p = float(_pce_scorer(row)[0])
    levers = []
    if row.at[0, "smoker"] == 1:
        levers.append(dict(key="smoking", title="Smoking", compare="A non-smoker like you",
                           changes={"smoker": 0.0}))
    if sbp > 120:
        levers.append(dict(key="bp", title="Blood pressure", compare="Someone like you with systolic pressure 120",
                           changes={"sbp": 120.0}))
    if tc > 200:
        levers.append(dict(key="cholesterol", title="Cholesterol", compare="Someone like you with cholesterol 190",
                           changes={"tc": 190.0}))
    habits, best = what_if(_pce_scorer, row, p, levers)
    lvl = level_of(p, None, PCE_BANDS)
    return {"title": PCE_TITLE, "status": "done", "probability": p, "level": lvl,
            "frequency": frequency_text(p), "phrase": PCE_PHRASE, "habits": habits, "best_case": best,
            "raising_risk": [], "lowering_risk": [], "technical": False,
            "note": PCE_NOTE + (PCE_ODD_NOTE if (tc / hdl >= 6 or age >= 78) else "")}, lvl


MODULES = {
    "disease_now": dict(
        file="clinical_model", table=Tables.UCI, build=build_A, levers=levers_none, display="calibrated",
        title="Heart disease check (clinical)",
        phrase="have heart disease (among people already sent for heart tests)",
        needs=["age", "sex", "chest_pain", "sys_bp", "total_chol", "max_hr", "exercise_angina", "oldpeak"],
        note="Based on hospital patients who were already referred for heart tests, so it is "
             "not a general-population chance."),
    "heart_attack_screening": dict(
        file="heart_attack_model", table=Tables.BRFSS, build=build_B, levers=levers_B, display="raw",
        title="Heart attack screening (lifestyle)",
        phrase="have had a heart attack",
        needs=["age", "sex", "bmi"],
        note="Chance that a person with this profile has had a heart attack (US survey data). "
             "Screening only, not a prediction of the future."),
    "angina_chd_screening": dict(
        file="angina_chd_model", table=Tables.BRFSS, build=build_B, levers=levers_B, display="raw",
        title="Angina / coronary heart disease screening (lifestyle)",
        phrase="have had angina or coronary heart disease",
        needs=["age", "sex", "bmi"],
        note="Chance that a person with this profile has angina or coronary heart disease "
             "(US survey data). Screening only."),
    "ten_year_risk": dict(
        file="risk_10year_model", table=Tables.FRAMINGHAM, build=build_C, levers=levers_C, display="calibrated",
        title="10-year heart disease risk",
        phrase="get coronary heart disease within 10 years",
        needs=["age", "sex", "sys_bp", "total_chol"],
        # Clinical groups are <5% / 5-20% / 20%+ but they are built for MODERN risk scales. The
        # Framingham cohort runs about 2.4x higher, so the cut-offs are rank-matched to the
        # clinical PCE on 20,000 simulated adults (agreement 85% vs 27% with 5/20).
        bands=(0.15, 0.37), age_range=(30, 79), soft_age=(40, 70),
        note="Chance of coronary heart disease within 10 years (US Framingham study, older "
             "cohort). Numbers from this study usually run higher than modern calculators."),
}


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def make_scorer(bundle, display):
    """Returns f(DataFrame) -> array of probabilities.
    display='raw' uses the model's own probability, from inside the calibration wrapper."""
    if display == "raw":
        try:
            base = [cc.estimator for cc in bundle["model"].calibrated_classifiers_]
            if base:
                return lambda X: np.mean([b.predict_proba(X)[:, 1] for b in base], axis=0)
        except Exception:
            pass                           # not a calibrated wrapper: fall through
    return lambda X: bundle["model"].predict_proba(X)[:, 1]


def what_if(score, row, p_now, levers):
    """Test each lever alone, then all together. Keeps only levers that really lower the risk."""
    kept = []
    for lv in levers:
        r2 = row.copy()
        for k, v in lv["changes"].items():
            r2[k] = v
        after = float(score(r2)[0])
        ratio = after / p_now if p_now > 0 else 1.0
        if ratio < 0.97:
            kept.append({**lv, "after": after, "ratio": ratio, "times_higher": 1.0 / max(ratio, 1e-9)})
    best = None
    if kept:
        r2 = row.copy()
        for lv in kept:
            for k, v in lv["changes"].items():
                r2[k] = v
        after = float(score(r2)[0])
        ratio = after / p_now if p_now > 0 else 1.0
        best = {"after": after, "ratio": ratio, "times_higher": 1.0 / max(ratio, 1e-9)}
    kept.sort(key=lambda d: d["ratio"])
    return kept, best


def level_of(p, thr, bands=None):
    """bands = (low_limit, high_limit). Without bands: module A uses 30% / 60%,
    modules B use half of the alert threshold / the alert threshold."""
    if bands is None:
        bands = (0.30, 0.60) if thr is None else (thr / 2, thr)
    lo, hi = bands
    return "Low" if p < lo else ("Moderate" if p < hi else "High")


# ----------------------------------------------------------------------------
# SHAP background data (needed once) and technical explanation
# ----------------------------------------------------------------------------
def build_background(n=100):
    out = {}
    for key, spec in MODULES.items():
        bundle = joblib.load(MODELS / f"{spec['file']}.joblib")
        df = load(spec["table"])[bundle["features"]].dropna()
        out[spec["file"]] = df.sample(min(n, len(df)), random_state=42).reset_index(drop=True)
    joblib.dump(out, MODELS / "background.joblib")
    print("saved SHAP background ->", MODELS / "background.joblib")


def _background():
    path = MODELS / "background.joblib"
    if not path.exists():
        raise FileNotFoundError("Run:  python -m src.predict --build   (one time)")
    return joblib.load(path)


def explain(score, feats, row, background, shown=None, top_up=3, top_down=2):
    """SHAP: how many percentage points each input moved the risk, compared with the
    AVERAGE person in the training data (not with someone of your own age)."""
    shown = shown or {}

    def f(d):
        return score(pd.DataFrame(d, columns=feats))

    masker = shap.maskers.Independent(background, max_samples=len(background))
    ex = shap.Explainer(f, masker, algorithm="permutation")
    sv = ex(row, max_evals=6 * len(feats) + 1).values[0]
    items = []
    for name, val, eff in zip(feats, row.iloc[0].tolist(), sv):
        val = shown.get(name, val)
        items.append({"factor": FRIENDLY.get(name, name),
                      "your_value": None if pd.isna(val) else round(float(val), 2),
                      "effect_points": round(float(eff) * 100, 2)})
    items.sort(key=lambda d: d["effect_points"], reverse=True)
    raising = [d for d in items if d["effect_points"] > 0.05][:top_up]
    lowering = [d for d in items[::-1] if d["effect_points"] < -0.05][:top_down]
    return raising, lowering


# ----------------------------------------------------------------------------
# The main function
# ----------------------------------------------------------------------------
def predict_heart_health(inp: dict) -> dict:
    inp, warnings_list = clean_inputs(inp)
    result = {"modules": {}, "disclaimer": DISCLAIMER, "input_warnings": warnings_list,
              "alerts": safety_alerts(inp)}
    if inp.get("symptoms_now"):
        result["emergency"] = EMERGENCY
    background = _background()
    levels, flags = [], {}
    if any(a["severity"] == "urgent" for a in result["alerts"]):
        levels.append("High")                       # blood-pressure crisis always counts

    pce_mod, pce_level = run_pce(inp)               # clinical calculator first (it decides the banner)

    for key, spec in MODULES.items():
        missing = [k for k in spec["needs"] if np.isnan(num(inp, k))]
        if missing:
            result["modules"][key] = {"title": spec["title"], "status": "skipped",
                                      "needs_more_inputs": missing}
            continue
        age_now = num(inp, "age")
        if "age_range" in spec and not (spec["age_range"][0] <= age_now <= spec["age_range"][1]):
            lo_a, hi_a = spec["age_range"]
            result["modules"][key] = {
                "title": spec["title"], "status": "skipped", "needs_more_inputs": [],
                "reason": (f"10-year risk calculators are built for adults aged {lo_a} to {hi_a} "
                           "(the study had nobody younger than 32). For your age, the lifestyle "
                           "screening results are more useful.")}
            continue
        bundle = joblib.load(MODELS / f"{spec['file']}.joblib")
        feats = bundle["features"]
        row = pd.DataFrame([spec["build"](inp)])[feats]
        score = make_scorer(bundle, spec["display"])

        p = float(score(row)[0])
        lvl = level_of(p, bundle.get("threshold"), spec.get("bands"))
        habits, best = what_if(score, row, p, spec["levers"](row))
        up, down = explain(score, feats, row, background[spec["file"]],
                           {"age_group": num(inp, "age")})      # show age in years

        note = spec["note"]
        if "soft_age" in spec and not (spec["soft_age"][0] <= age_now <= spec["soft_age"][1]):
            note += (f" Your age is at the edge of the ages the study covered well "
                     f"({spec['soft_age'][0]} to {spec['soft_age'][1]}), so treat it as rough.")
        banner_note = None
        if key == "ten_year_risk" and pce_level is not None:
            banner_note = ("Shown for comparison. The overall result uses the clinical calculator "
                           "below because you gave your HDL cholesterol.")
        else:
            levels.append(lvl)
        for h in habits:                       # habits that raise risk by about 18% or more
            if h["ratio"] <= 0.85:
                flags[h["key"]] = max(flags.get(h["key"], 0.0), h["times_higher"])

        result["modules"][key] = {
            "title": spec["title"], "status": "done", "probability": p, "level": lvl,
            "frequency": frequency_text(p), "phrase": spec["phrase"], "habits": habits, "best_case": best,
            "raising_risk": up, "lowering_risk": down, "note": note, "banner_note": banner_note}

    result["modules"]["ten_year_ascvd_pce"] = pce_mod
    if pce_level is not None:
        levels.append(pce_level)
        for h in pce_mod["habits"]:
            if h["ratio"] <= 0.85:
                flags[h["key"]] = max(flags.get(h["key"], 0.0), h["times_higher"])

    result["habit_flags"] = [{"key": k, "word": HABIT_WORDS.get(k, k), "times_higher": v}
                             for k, v in sorted(flags.items(), key=lambda kv: -kv[1])]
    if "High" in levels:
        result["heart_health_level"] = "high"
        result["heart_health"] = "Needs attention: please arrange a check-up with a doctor."
    elif "Moderate" in levels:
        result["heart_health_level"] = "moderate"
        result["heart_health"] = "Keep an eye on it: some risk factors are present."
    elif levels and result["habit_flags"]:
        words = ", ".join(f["word"] for f in result["habit_flags"])
        result["heart_health_level"] = "watch"
        result["heart_health"] = (f"Low risk today, but your habits are pushing it up: {words}. "
                                  "Changing them early matters most.")
    elif levels:
        result["heart_health_level"] = "good"
        result["heart_health"] = "Good: no major risk signals from the information given."
    else:
        result["heart_health_level"] = "unknown"
        result["heart_health"] = "Not enough information to judge."
    return result


# ----------------------------------------------------------------------------
# Plain-text formatting (used by the demo; the app has its own layout)
# ----------------------------------------------------------------------------
def pct(p: float) -> str:
    if p >= 0.10:
        return f"{p:.0%}"
    if p >= 0.01:
        return f"{p:.1%}"
    return f"{p:.2%}"


def frequency_text(p: float) -> str:
    """'86 in 1,000' / '6 in 10,000' / '4 in 100,000': the unit is chosen so the count is at
    least 3, so a tiny difference (6 vs 4 in 10,000) is still visible to a normal reader."""
    for unit in (1_000, 10_000, 100_000):
        n = p * unit
        if n >= 3 or unit == 100_000:
            return f"{max(round(n), 1):,} in {unit:,}"


def print_result(r):
    if "emergency" in r:
        print("\n!!! " + r["emergency"])
    for w in r.get("input_warnings", []):
        print("\n[input warning]", w)
    for a in r.get("alerts", []):
        print(f"\n!!! [{a['severity']}] {a['text']}")
    for m in r["modules"].values():
        print(f"\n== {m['title']}")
        if m["status"] == "skipped":
            print("   skipped:", m.get("reason") or ("needs " + ", ".join(m["needs_more_inputs"])))
            continue
        if m.get("banner_note"):
            print("   (" + m["banner_note"] + ")")
        print(f"   {pct(m['probability'])}  ->  {m['level']}   "
              f"(about {m['frequency']} people with your answers {m['phrase']})")
        if m["best_case"]:
            b = m["best_case"]
            print(f"   same person with healthy habits: {pct(b['after'])}  "
                  f"-> your habits make it {m['probability'] / b['after']:.1f}x higher")
            for h in m["habits"]:
                print(f"     * {h['title']}: {h['compare'][0].lower() + h['compare'][1:]}: {pct(h['after'])} "
                      f"(yours {pct(m['probability'])}, {h['times_higher']:.1f}x higher)")
        for d in m["raising_risk"]:
            print(f"   [technical] + {d['factor']} ({d['your_value']}): +{d['effect_points']} points vs average adult")
        for d in m["lowering_risk"]:
            print(f"   [technical] - {d['factor']} ({d['your_value']}): {d['effect_points']} points vs average adult")
    print(f"\nHEART HEALTH [{r['heart_health_level']}]:", r["heart_health"])
    print(r["disclaimer"])


DEMO_PEOPLE = {
    "52-year-old male, healthy numbers (with HDL)": dict(
        age=52, sex="male", bmi=24, smoking_status=0, diabetes_status=0, physical_activity=1,
        sleep_hours=7, general_health=3, stroke=0, copd=0, kidney_disease=0, arthritis=0, depression=0,
        hypertension=0, bp_meds=0, sys_bp=118, dia_bp=76, total_chol=185, hdl_chol=55, glucose=88,
        resting_heart_rate=66),
    "58-year-old, blood-pressure crisis reading": dict(
        age=58, sex="female", bmi=29, smoking_status=0, diabetes_status=0, physical_activity=0,
        sleep_hours=6, general_health=2, stroke=0, copd=0, kidney_disease=0, arthritis=0, depression=0,
        hypertension=1, bp_meds=0, sys_bp=190, dia_bp=122, total_chol=230, hdl_chol=48, glucose=100,
        resting_heart_rate=80),
    "Age 20 male, NON-smoker": dict(
        age=20, sex="male", bmi=24.2, smoking_status=0, diabetes_status=0, physical_activity=1,
        sleep_hours=7, general_health=2, stroke=0, copd=0, kidney_disease=0, arthritis=0, depression=0),
    "Age 20 male, CURRENT smoker": dict(
        age=20, sex="male", bmi=24.2, smoking_status=2, diabetes_status=0, physical_activity=1,
        sleep_hours=7, general_health=2, stroke=0, copd=0, kidney_disease=0, arthritis=0, depression=0),
    "62-year-old male, smoker, diabetic, with clinical tests": dict(
        age=62, sex="male", bmi=31, smoking_status=2, cigs_per_day=15, diabetes_status=2,
        physical_activity=0, sleep_hours=5, general_health=1, stroke=0, copd=0, kidney_disease=0,
        arthritis=1, depression=0, hypertension=1, bp_meds=1, sys_bp=158, dia_bp=96, total_chol=262,
        glucose=148, hdl_chol=38, resting_heart_rate=84, chest_pain=4, rest_ecg=1, max_hr=118,
        exercise_angina=1, oldpeak=2.1, st_slope=2),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="save SHAP background data (one time)")
    ap.add_argument("--demo", action="store_true", help="run example people")
    a = ap.parse_args()
    if a.build:
        build_background()
    if a.demo:
        for name, person in DEMO_PEOPLE.items():
            print("\n" + "#" * 70 + f"\n# {name}\n" + "#" * 70)
            print_result(predict_heart_health(person))
