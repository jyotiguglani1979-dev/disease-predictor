"""
app.py   (the HeartAI web app)
------------------------------
Run from the heartai folder:   streamlit run app.py
"""
import pandas as pd
import streamlit as st

from config import MODELS
from src.predict import DISCLAIMER, EMERGENCY, FRIENDLY, pct, predict_heart_health

st.set_page_config(page_title="HeartAI", page_icon="❤️", layout="wide")

COLOR = {"Low": "green", "Moderate": "orange", "High": "red"}
HEALTH_MAP = {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4}
SMOKE_MAP = {"Never smoked": 0, "Former smoker": 1, "Current smoker": 2}
DIABETES_MAP = {"No": 0, "Pre-diabetes / borderline": 1, "Yes": 2}
CHEST_PAIN = {"1 - Typical angina": 1, "2 - Atypical angina": 2,
              "3 - Non-anginal pain": 3, "4 - No chest pain (asymptomatic)": 4}
ECG = {"0 - Normal": 0, "1 - ST-T wave abnormality": 1, "2 - Left ventricular hypertrophy": 2}
SLOPE = {"1 - Upsloping": 1, "2 - Flat": 2, "3 - Downsloping": 3}
HISTORY = {"Stroke": "stroke", "Lung disease (COPD)": "copd", "Kidney disease": "kidney_disease",
           "Arthritis": "arthritis", "Depression": "depression",
           "High blood pressure": "hypertension", "Taking blood pressure medicine": "bp_meds"}
UNLOCK = {"chest_pain": "chest pain type", "max_hr": "max heart rate", "oldpeak": "ST depression",
          "exercise_angina": "chest pain on exercise", "sys_bp": "systolic BP",
          "total_chol": "cholesterol", "age": "age", "sex": "sex", "bmi": "height and weight"}


# ----------------------------------------------------------------------------
# Header + safety
# ----------------------------------------------------------------------------
st.title("❤️ HeartAI: Heart Health Check")
st.caption("A college research project. It estimates risk. It does not diagnose.")

needed = ["clinical_model", "heart_attack_model", "angina_chd_model",
          "risk_10year_model", "background"]
missing_files = [n for n in needed if not (MODELS / f"{n}.joblib").exists()]
if missing_files:
    st.error("Models not found: " + ", ".join(missing_files))
    st.code("python -m src.train_all\npython -m src.cv_report\npython -m src.predict --build")
    st.stop()

symptoms = st.checkbox(
    "🚨 I have chest pain or pressure, pain in the arm/jaw/back, sudden breathlessness, "
    "cold sweat or fainting RIGHT NOW", key="symptoms")
if symptoms:
    st.error("**" + EMERGENCY + "**")

# ----------------------------------------------------------------------------
# Form
# ----------------------------------------------------------------------------
with st.form("heart_form"):
    st.subheader("1. Basics")
    c1, c2, c3, c4 = st.columns(4)
    age = c1.number_input("Age (years)", 18, 100, 45, key="age")
    sex = c2.radio("Sex", ["Male", "Female"], horizontal=True, key="sex")
    height = c3.number_input("Height (cm)", 100, 230, 170, key="height")
    weight = c4.number_input("Weight (kg)", 30, 250, 70, key="weight")

    st.subheader("2. Lifestyle and health history")
    c1, c2, c3 = st.columns(3)
    smoking = c1.selectbox("Smoking", list(SMOKE_MAP), key="smoking")
    cigs = c1.number_input("Cigarettes per day (only if you smoke)", 0, 80, 10, key="cigs")
    diabetes = c2.selectbox("Diabetes", list(DIABETES_MAP), key="diabetes")
    active = c2.radio("Physically active (exercise in the last month)", ["Yes", "No"],
                      horizontal=True, key="active")
    general = c3.selectbox("Your general health", list(HEALTH_MAP), index=2, key="general")
    sleep = c3.slider("Sleep (hours per night)", 3, 12, 7, key="sleep")
    history = st.multiselect("Have you ever been told you have / do you take:", list(HISTORY),
                             key="history")

    st.subheader("3. Measurements (optional, needed for 10-year risk)")
    has_meas = st.checkbox("I know my blood pressure and cholesterol", key="has_meas")
    c1, c2, c3, c4, c5 = st.columns(5)
    sys_bp = c1.number_input("Systolic BP (top)", 80, 240, 120, key="sys_bp")
    dia_bp = c2.number_input("Diastolic BP (bottom)", 40, 140, 80, key="dia_bp")
    chol = c3.number_input("Total cholesterol (mg/dL)", 100, 400, 190, key="chol")
    gluc = c4.number_input("Blood sugar (mg/dL)", 60, 400, 95, key="gluc")
    rhr = c5.number_input("Resting heart rate", 40, 140, 72, key="rhr")

    st.subheader("4. Hospital test results (optional; together with section 3 it unlocks "
                 "the clinical heart disease check)")
    has_clin = st.checkbox("I have stress-test / ECG results", key="has_clin")
    c1, c2, c3 = st.columns(3)
    cp = c1.selectbox("Chest pain type", list(CHEST_PAIN), key="cp")
    ecg = c1.selectbox("Resting ECG", list(ECG), key="ecg")
    max_hr = c2.number_input("Max heart rate in stress test", 60, 220, 150, key="max_hr")
    exang = c2.radio("Chest pain during exercise", ["No", "Yes"], horizontal=True, key="exang")
    oldpeak = c3.number_input("ST depression (oldpeak)", 0.0, 7.0, 0.0, 0.1, key="oldpeak")
    slope = c3.selectbox("ST slope", list(SLOPE), key="slope")

    submitted = st.form_submit_button("Check my heart health", type="primary")


def build_inputs() -> dict:
    current_smoker = SMOKE_MAP[smoking] == 2
    inp = {
        "age": age, "sex": sex.lower(), "bmi": round(weight / (height / 100) ** 2, 1),
        "smoking_status": SMOKE_MAP[smoking],
        "cigs_per_day": cigs if current_smoker else 0,
        "diabetes_status": DIABETES_MAP[diabetes], "physical_activity": int(active == "Yes"),
        "sleep_hours": sleep, "general_health": HEALTH_MAP[general],
        "symptoms_now": symptoms,
    }
    for label, key in HISTORY.items():
        inp[key] = int(label in history)
    if has_meas:
        inp.update(sys_bp=sys_bp, dia_bp=dia_bp, total_chol=chol, glucose=gluc,
                   resting_heart_rate=rhr)
    if has_clin:
        inp.update(chest_pain=CHEST_PAIN[cp], rest_ecg=ECG[ecg], max_hr=max_hr,
                   exercise_angina=int(exang == "Yes"), oldpeak=oldpeak, st_slope=SLOPE[slope])
        # the clinical model also uses blood pressure and cholesterol
    return inp


# ----------------------------------------------------------------------------
# Result cards: plain words and numbers first, technical SHAP view collapsed
# ----------------------------------------------------------------------------
def technical_view(m: dict) -> None:
    rows = m["raising_risk"] + m["lowering_risk"]
    with st.expander("Technical view: what moved this number (for the report)"):
        if not rows:
            st.caption("No single input stands out.")
            return
        for d in m["raising_risk"]:
            st.markdown(f"- **{d['factor']}** ({d['your_value']}) pushes it **up** by "
                        f"{d['effect_points']:.2f} percentage points")
        for d in m["lowering_risk"]:
            st.markdown(f"- **{d['factor']}** ({d['your_value']}) pushes it **down** by "
                        f"{abs(d['effect_points']):.2f} percentage points")
        df = pd.DataFrame(rows).sort_values("effect_points")
        df["factor"] = df["factor"] + " (" + df["your_value"].astype(str) + ")"
        st.bar_chart(df.set_index("factor")["effect_points"], horizontal=True,
                     height=36 * len(df) + 60)
        st.caption("These numbers compare you with the AVERAGE adult in the data (about 50 "
                   "years old). That is why a young person sees a big minus for age. For "
                   "your own age, use the comparison with healthy habits above.")


def show_card(key: str, m: dict) -> None:
    p = m["probability"]
    st.markdown(f"#### {m['title']}")
    st.markdown(f"### {pct(p)}  :{COLOR[m['level']]}[{m['level']}]")
    st.markdown(f"About **{m['frequency']}** people with your answers {m['phrase']}.")

    best = m["best_case"]
    if best:
        chart = pd.DataFrame({"Estimated risk (%)": [p * 100, best["after"] * 100]},
                             index=["You", "Same person, healthy habits"])
        st.bar_chart(chart, horizontal=True, height=150)
        st.markdown(f"With healthy habits, the same person would be at **{pct(best['after'])}**. "
                    f"Your habits make your risk about **{p / best['after']:.1f} times higher**.")
        st.markdown("**What you can change** (risk if only that one thing were different):")
        for h in m["habits"]:
            st.markdown(f"- **{h['title']}:** {h['compare'][0].lower() + h['compare'][1:]} is at **{pct(h['after'])}** "
                        f"(you: {pct(p)}, {h['times_higher']:.1f} times higher)")
    elif key == "disease_now":
        st.markdown("**Main reasons (from your test results):**")
        for d in m["raising_risk"]:
            st.markdown(f"- **{d['factor']}** ({d['your_value']}) raises the estimate "
                        f"by about {d['effect_points']:.0f} points")
        for d in m["lowering_risk"]:
            st.markdown(f"- **{d['factor']}** ({d['your_value']}) lowers the estimate "
                        f"by about {abs(d['effect_points']):.0f} points")
    else:
        st.caption("None of the habits you entered clearly raises this estimate.")

    technical_view(m)
    st.caption(m["note"])


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
if submitted:
    with st.spinner("Calculating..."):
        result = predict_heart_health(build_inputs())

    level = result["heart_health_level"]
    text = "### Heart health: " + result["heart_health"]
    if level == "high":
        st.error(text)
    elif level in ("moderate", "watch"):
        st.warning(text)
    elif level == "good":
        st.success(text)
    else:
        st.info(text)

    done = {k: m for k, m in result["modules"].items() if m["status"] == "done"}
    skipped = {k: m for k, m in result["modules"].items() if m["status"] == "skipped"}

    cols = st.columns(2)
    for i, (key, m) in enumerate(done.items()):
        with cols[i % 2].container(border=True):
            show_card(key, m)

    for m in skipped.values():
        need = ", ".join(UNLOCK.get(k, FRIENDLY.get(k, k)) for k in m["needs_more_inputs"])
        st.info(f"**{m['title']}** was skipped. Missing: {need}. "
                "Fill the optional sections above to unlock it.")

    if level in ("high", "moderate"):
        st.markdown("**Sensible next steps:** book a check-up and ask for blood pressure, "
                    "cholesterol and blood sugar tests; do not smoke; stay active; "
                    "ask your doctor before starting heavy exercise.")
    elif level == "watch":
        st.markdown("**Easy wins:** start with the habits listed in the cards above. "
                    "Your risk is low today, and it is far easier to change habits now than later.")
    st.caption(DISCLAIMER)

# ----------------------------------------------------------------------------
# About
# ----------------------------------------------------------------------------
with st.expander("About this tool: how to read the numbers, data, accuracy and limits"):
    st.markdown("""
**How to read the numbers**
- *"6 in 10,000 people"* means: out of 10,000 people with exactly your answers, about 6 are
  expected to have this condition. Young people have very small numbers, but habits still
  show up as *"1.5 times higher"*.
- *"Healthy habits"* means the same person, same age and history, but not smoking, active,
  healthy weight, normal sleep, and (for 10-year risk) blood pressure 120/80 and cholesterol
  under 200. Each change is tested alone and all together using the model.

**Four results, three models**
- *Heart disease check (clinical)*: UCI Heart Disease, 920 hospital patients from 4 hospitals.
  Tested on hospitals it never saw: AUC about 0.81.
- *Heart attack and angina/CHD screening*: US BRFSS 2022 survey, about 438,000 people.
  Test AUC about 0.83 and 0.84. These estimate whether a person like you has had one, not the future.
- *10-year risk*: Framingham, about 4,200 people. Cross-validated AUC about 0.73 (modest).

**Limits**
- US data and self-reported answers. Not validated on Indian patients.
- Risk estimates are averages for people with similar answers, not a diagnosis.
- The clinical check reflects patients already referred to hospital tests.

**Credits:** design ideas from the open projects *CardioPredict* (Aamir-J) and *CardioSense* (Surieyaa).
""")
