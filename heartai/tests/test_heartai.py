"""
Run from the heartai folder:   python -m pytest tests -q      (pip install pytest, one time)
Checks that the safety rules and the clinical calculator behave as expected.
"""
import numpy as np
import pytest

from src.clinical import clean_inputs, pce_risk, safety_alerts
from src.predict import MODELS, level_of, predict_heart_health

HAVE_MODELS = all((MODELS / f"{n}.joblib").exists() for n in
                  ["clinical_model", "heart_attack_model", "angina_chd_model",
                   "risk_10year_model", "background"])
needs_models = pytest.mark.skipif(not HAVE_MODELS, reason="train the models first")

BASE = dict(age=55, sex="male", bmi=26, smoking_status=0, diabetes_status=0, physical_activity=1,
            sleep_hours=7, general_health=2, stroke=0, copd=0, kidney_disease=0, arthritis=0,
            depression=0)


# ---- clinical calculator: published example values (ACC/AHA 2013, age 55, SBP 120, TC 213, HDL 50)
def test_pce_white_female_reference():
    assert abs(pce_risk(55, 0, 120, 213, 50, 0, 0, False) * 100 - 2.1) < 0.1


def test_pce_white_male_reference():
    assert abs(pce_risk(55, 1, 120, 213, 50, 0, 0, False) * 100 - 5.4) < 0.1


def test_pce_smoking_and_bp_raise_risk():
    base = pce_risk(55, 1, 120, 213, 50, 0, 0, False)
    assert pce_risk(55, 1, 120, 213, 50, 1, 0, False) > base
    assert pce_risk(55, 1, 160, 213, 50, 0, 0, False) > base


# ---- input cleaning and safety
def test_impossible_values_are_ignored():
    out, warns = clean_inputs({"age": 40, "sys_bp": 900, "total_chol": 190})
    assert "sys_bp" not in out and out["total_chol"] == 190 and len(warns) == 1


def test_diastolic_must_be_lower_than_systolic():
    out, warns = clean_inputs({"sys_bp": 90, "dia_bp": 120})
    assert "sys_bp" not in out and "dia_bp" not in out and warns


def test_bp_crisis_alert():
    assert safety_alerts({"sys_bp": 190, "dia_bp": 100})[0]["severity"] == "urgent"
    assert safety_alerts({"sys_bp": 120, "dia_bp": 80}) == []


def test_level_bands():
    assert level_of(0.03, None, (0.05, 0.20)) == "Low"
    assert level_of(0.10, None, (0.05, 0.20)) == "Moderate"
    assert level_of(0.25, None, (0.05, 0.20)) == "High"


# ---- full predictor (needs trained models)
@needs_models
def test_young_person_gets_no_ten_year_result():
    r = predict_heart_health({**BASE, "age": 22, "sys_bp": 118, "dia_bp": 76, "total_chol": 170, "hdl_chol": 55})
    assert r["modules"]["ten_year_risk"]["status"] == "skipped"
    assert r["modules"]["ten_year_ascvd_pce"]["status"] == "skipped"


@needs_models
def test_smoker_never_looks_safer_than_non_smoker():
    a = predict_heart_health({**BASE, "age": 20})["modules"]["heart_attack_screening"]["probability"]
    b = predict_heart_health({**BASE, "age": 20, "smoking_status": 2})["modules"]["heart_attack_screening"]["probability"]
    assert b >= a


@needs_models
def test_bp_crisis_forces_high_banner():
    r = predict_heart_health({**BASE, "sys_bp": 195, "dia_bp": 125, "total_chol": 190, "hdl_chol": 50})
    assert r["heart_health_level"] == "high" and r["alerts"]


@needs_models
def test_clinical_calculator_decides_banner_when_hdl_given():
    r = predict_heart_health({**BASE, "age": 52, "sys_bp": 118, "dia_bp": 76, "total_chol": 185, "hdl_chol": 55})
    assert r["modules"]["ten_year_ascvd_pce"]["status"] == "done"
    assert r["modules"]["ten_year_risk"]["banner_note"]
    assert r["heart_health_level"] in ("good", "watch")


@needs_models
def test_symptoms_show_emergency_text():
    assert "emergency" in predict_heart_health({**BASE, "symptoms_now": True})


# ---- direction checks: these also catch a broken dataset or a column mix-up on YOUR machine
def _p(module, **kw):
    r = predict_heart_health({**BASE, **kw})["modules"][module]
    return r["probability"]


CLIN = dict(sys_bp=125, dia_bp=80, total_chol=200, hdl_chol=50)


@needs_models
def test_screening_risk_goes_up_with_age_and_diabetes():
    assert _p("heart_attack_screening", age=65) > _p("heart_attack_screening", age=40)
    assert _p("heart_attack_screening", diabetes_status=2) > _p("heart_attack_screening", diabetes_status=0)


@needs_models
def test_ten_year_risk_goes_up_with_age_bp_and_smoking():
    base = _p("ten_year_risk", age=50, **CLIN)
    assert _p("ten_year_risk", age=62, **CLIN) > base
    assert _p("ten_year_risk", age=50, **{**CLIN, "sys_bp": 165}) > base
    assert _p("ten_year_risk", age=50, smoking_status=2, cigs_per_day=15, **CLIN) > base


@needs_models
def test_high_reading_counts_as_hypertension_even_if_not_ticked():
    from src.predict import _hypertension
    assert _hypertension({"sys_bp": 165, "dia_bp": 95}) == 1.0
    assert _hypertension({"sys_bp": 118, "dia_bp": 76, "hypertension": 0}) == 0.0


@needs_models
def test_unusual_cholesterol_pattern_gets_a_caution():
    odd = predict_heart_health({**BASE, "age": 45, "sex": "female", "sys_bp": 130, "dia_bp": 80,
                                "total_chol": 300, "hdl_chol": 35})["modules"]["ten_year_ascvd_pce"]
    normal = predict_heart_health({**BASE, "age": 45, "sys_bp": 120, "dia_bp": 80,
                                   "total_chol": 190, "hdl_chol": 55})["modules"]["ten_year_ascvd_pce"]
    assert "Caution" in odd["note"] and "Caution" not in normal["note"]
