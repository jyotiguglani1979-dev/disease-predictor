import json, re, uuid
import inspect as pyinspect
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import inspect, text
import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from .clinical import EVIDENCE_BY_CODE, CONDITIONS, evidence_info, parse_free_text, emergency_warning, condition_label
from .coverage import detect_unmodelled_mentions
from .indian_disease import predict_indian, extract_indian_symptoms
from .combine import combine_rankings
from .config import DB_DIR, MODEL_PATH, INDIAN_MODEL_PATH, CONSENT_REQUIRED, HEALTH_REGION
from .database import Base, SessionLocal, engine
from .models import AssessmentRecord, UserAccount
from .schemas import StartAssessment, AnswerRequest, ParseRequest, ChatRequest, AuthRequest, ExplanationRequest
from .auth import current_user, make_token, password_hash, valid_email, verify_password
from .llm import extract_symptoms, gemini_text, explain_rankings
from .retrieval import retriever, SOURCES

Base.metadata.create_all(bind=engine)
# Additive migration for local databases made by earlier app revisions.
if "assessments" in inspect(engine).get_table_names():
    columns = {col["name"] for col in inspect(engine).get_columns("assessments")}
    with engine.begin() as conn:
        if "answers_json" not in columns: conn.execute(text("ALTER TABLE assessments ADD COLUMN answers_json TEXT NOT NULL DEFAULT '{}'"))
        if "user_id" not in columns: conn.execute(text("ALTER TABLE assessments ADD COLUMN user_id VARCHAR(36)"))
app = FastAPI(title="Health Assessment API", version="1.0.0", description="Educational symptom assessment and cardiovascular risk research tool.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    # Vite is commonly opened with the laptop's LAN IP on this college project.
    # Keep this limited to private IPv4 ranges and the Vite development port.
    allow_origin_regex=r"^http://(?:(?:localhost)|(?:127(?:\.\d{1,3}){3})|(?:10(?:\.\d{1,3}){3})|(?:192\.168(?:\.\d{1,3}){2})|(?:172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})):5173$",
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
_artifact = None
BINARY_EVIDENCE_CODES = [code for code, item in EVIDENCE_BY_CODE.items() if item.get("data_type") == "B"]
MAX_FOLLOWUPS = 5
COMBINED_NOTE = ("Combined ranking: an equal mix of the DDXPlus model and the Indian-symptom model. "
                 "A disease both models know gets credit from both. The score is a mix, not a probability.")

def _current_reported_evidence(codes, text):
    """Do not let current headache language answer history-only evidence items."""
    local_history = set(parse_free_text(text))
    return [code for code in codes if code not in {"E_25", "E_99"} or code in local_history]

def db_session():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def model_artifact():
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists(): raise HTTPException(503, "The DDXPlus model has not been trained. Run: python -m backend.train_ddxplus")
        _artifact = joblib.load(MODEL_PATH)
    return _artifact

def normalize_tokens(tokens: list[str]) -> list[str]:
    clean = []
    for token in tokens:
        pieces = token.split("_@_")
        if pieces[0] not in EVIDENCE_BY_CODE: continue
        if len(pieces) == 1:
            if EVIDENCE_BY_CODE[pieces[0]].get("data_type") == "B": clean.append(token)
        elif len(pieces) == 2 and str(pieces[1]) in {str(v) for v in EVIDENCE_BY_CODE[pieces[0]].get("possible-values", [])}:
            clean.append(token)
    return list(dict.fromkeys(clean))

def features(age: int, sex: str, evidence: list[str], negative_codes: list[str] | None = None) -> dict:
    sex_value = "M" if sex.strip().lower() in {"male", "m"} else "F" if sex.strip().lower() in {"female", "f"} else None
    result = {"age": float(age)}
    if sex_value: result["sex:" + sex_value] = 1.0
    result.update({"evidence:" + x: 1.0 for x in evidence})
    result.update({"negative:" + code: 1.0 for code in (negative_codes or [])})
    return result

def score_distribution(model, x) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        values = np.asarray(model.predict_proba(x)[0], dtype=float)
        return values / max(float(values.sum()), 1e-12)
    values = np.asarray(model.decision_function(x)[0], dtype=float)
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()

def indian_knowledge_matches(indian: dict) -> list[dict]:
    if not indian.get("ready"):
        return []
    reported = indian.get("reported_symptoms", [])
    return [
        {
            "condition": item["condition"],
            "matched_symptoms": reported,
            "sources": ["indian_disease_binary.csv"],
            "label": "Binary Indian disease model",
        }
        for item in indian.get("results", [])
    ]

def assess(age: int, sex: str, evidence: list[str], answers: dict | None = None, reported_symptoms: list[str] | None = None, reported_text: str = "") -> dict:
    reported_symptoms = reported_symptoms or []
    indian = predict_indian(reported_text)
    if not evidence:
        return {"model": "DDXPlus", "score_label": "No DDXPlus score available",
                "results": [], "knowledge_matches": indian_knowledge_matches(indian),
                "indian_results": indian,
                "combined": combine_rankings([], indian.get("results", []), limit=5),
                "combined_note": COMBINED_NOTE,
                "reported_symptoms": reported_symptoms, "coverage_only": True,
                "reported_text": reported_text,
                "explanation": "The symptoms you described are present in the supplied symptom datasets, but they do not map to a supported DDXPlus finding. No model ranking is shown because that would be misleading. The related dataset entries below are symptom overlaps, not predictions or diagnoses. Add more symptoms or discuss them with a clinician.",
                "evidence_count": 0}
    artifact = model_artifact()
    answers = answers or {}
    negatives = [code for code, response in answers.items() if response == "no"]
    x = artifact["vectorizer"].transform([features(age, sex, evidence, negatives)])
    model = artifact["model"]
    normalized = score_distribution(model, x)
    classes = list(model.classes_)
    ranked = sorted(zip(classes, normalized), key=lambda row: row[1], reverse=True)[:10]
    symptom_terms = [EVIDENCE_BY_CODE[t.split("_@_")[0]].get("question_en", "") for t in evidence if t.split("_@_")[0] in EVIDENCE_BY_CODE]
    symptom_terms.extend(info.get("question_en", "") for code, info in EVIDENCE_BY_CODE.items() if (answers or {}).get(code) == "yes")
    model_candidates = [{"condition": condition_label(name), "score": round(float(score) * 100, 1)} for name, score in ranked]
    return {"model": artifact["model_name"], "score_label": "Relative model score (not a calibrated probability)",
            "results": model_candidates[:5], "model_candidates": model_candidates,
            "knowledge_matches": indian_knowledge_matches(indian),
            "indian_results": indian,
            "combined": combine_rankings(model_candidates, indian.get("results", []), limit=5),
            "combined_note": COMBINED_NOTE,
            "reported_symptoms": reported_symptoms,
            "reported_text": reported_text,
            "explanation": "Educational ranking; unanswered findings are unknown, not absent. It is not calibrated for India and does not diagnose or rule out a condition.",
            "evidence_count": len(evidence) + len(negatives)}

def next_question(age: int, sex: str, evidence: list[str], answered: list[str] | None = None, answers: dict | None = None, max_answers: int = MAX_FOLLOWUPS, reported_text: str = "") -> dict | None:
    if len(answered or []) >= max_answers:
        return None
    artifact = model_artifact(); model = artifact["model"]; vec: DictVectorizer = artifact["vectorizer"]
    used = {x.split("_@_")[0] for x in evidence}
    importance = getattr(model, "feature_importances_", None)
    if importance is None and hasattr(model, "coef_"):
        importance = np.abs(model.coef_).mean(axis=0)
    values = {}
    if importance is not None:
        for name, value in zip(vec.feature_names_, importance):
            if name.startswith("evidence:"):
                code = name[len("evidence:"):].split("_@_")[0]
                values[code] = max(values.get(code, 0.0), float(value))
    used.update(answered or [])
    indian = predict_indian(reported_text)
    if indian.get("ready") and indian["results"]:
        top_indian = indian["results"][0]["condition"].casefold()
        indian_priority = {
            "typhoid": ["E_94", "E_51", "E_211", "E_53", "E_162"],
            "malaria": ["E_94", "E_51", "E_211", "E_175", "E_144"],
            "dengue": ["E_129", "E_51", "E_94", "E_220", "E_144"],
            "pneumonia": ["E_77", "E_66", "E_220", "E_45", "E_201"],
            "tuberculosis": ["E_45", "E_162", "E_66", "E_77", "E_201"],
            "gastroenteritis": ["E_51", "E_211", "E_53", "E_162", "E_91"],
            "common cold": ["E_97", "E_181", "E_201", "E_94", "E_175"],
        }
        priority = [code for code in indian_priority.get(top_indian, []) if code not in used]
        if priority:
            return evidence_info(priority[0])
    presentation_priority = []
    if "E_129" in evidence and "E_91" in evidence:
        presentation_priority = ["E_94", "E_53", "E_51", "E_148", "E_66"]
    elif "E_51" in evidence and "E_148" in evidence and "E_53" in evidence:
        presentation_priority = ["E_211", "E_53", "E_51", "E_162", "E_94"]
    elif "E_51" in evidence and "E_148" in evidence:
        presentation_priority = ["E_211", "E_53", "E_51", "E_91", "E_66"]
    elif "E_94" in evidence and "E_144" in evidence and "E_91" in evidence:
        presentation_priority = ["E_94", "E_144", "E_175", "E_51", "E_53"]

    presentation_priority = [code for code in presentation_priority if code not in used]
    if presentation_priority:
        return evidence_info(presentation_priority[0])

    candidates = [code for code in EVIDENCE_BY_CODE
                  if code not in used and evidence_info(code) and not EVIDENCE_BY_CODE[code].get("is_antecedent", False)]
    candidates.sort(key=lambda c: values.get(c, 0.0), reverse=True)
    # Prioritize high-value differential questions for common respiratory cases.
    priority_codes = {
        "Pneumonia": ["E_77", "E_66", "E_220", "E_45", "E_201"],
        "Tuberculosis": ["E_45", "E_162", "E_66", "E_77", "E_201"],
        "Influenza": ["E_94", "E_144", "E_175", "E_97", "E_201"],
        "Bronchitis": ["E_77", "E_214", "E_66", "E_220", "E_201"],
        "URTI": ["E_97", "E_181", "E_201", "E_94", "E_175"],
        "Viral pharyngitis": ["E_97", "E_181", "E_201", "E_94", "E_91"],
        "Acute otitis media": ["E_91", "E_201", "E_181", "E_175", "E_97"],
    }
    negatives = [code for code, response in (answers or {}).items() if response == "no"]
    baseline = score_distribution(model, vec.transform([features(age, sex, evidence, negatives)]))
    top_index = int(np.argmax(baseline))
    top_name = condition_label(model.classes_[top_index])
    priority = [code for code in priority_codes.get(top_name, []) if code not in used]
    if priority:
        priority_set = set(priority)
        candidates = priority + [code for code in candidates if code not in priority_set]
        return evidence_info(priority[0])
    if top_name in priority_codes:
        return None
    negatives = [code for code, response in (answers or {}).items() if response == "no"]
    x_base = vec.transform([features(age, sex, evidence, negatives)])
    baseline = score_distribution(model, x_base)
    best, best_value = None, -1.0
    for code in candidates[:20]:
        info = EVIDENCE_BY_CODE[code]
        possible = [code] if info.get("data_type") == "B" else [f"{code}_@_{v}" for v in info.get("possible-values", [])[:10]]
        if not possible: continue
        trial_rows = [features(age, sex, evidence + [token], negatives) for token in possible]
        trial_x = vec.transform(trial_rows)
        distributions = model.predict_proba(trial_x) if hasattr(model, "predict_proba") else model.decision_function(trial_x)
        if not hasattr(model, "predict_proba"):
            shifted = distributions - distributions.max(axis=1, keepdims=True)
            distributions = np.exp(shifted); distributions /= distributions.sum(axis=1, keepdims=True)
        # Expected Jensen-Shannon divergence across the candidate's possible answers.
        divergence = 0.0
        for candidate_distribution in distributions:
            midpoint = 0.5 * (baseline + candidate_distribution)
            divergence += 0.5 * np.sum(baseline * np.log((baseline + 1e-12) / (midpoint + 1e-12)))
            divergence += 0.5 * np.sum(candidate_distribution * np.log((candidate_distribution + 1e-12) / (midpoint + 1e-12)))
        divergence /= len(distributions)
        adaptive_score = float(divergence) * (0.5 + values.get(code, 0.0))
        if adaptive_score > best_value:
            best, best_value = code, adaptive_score
    return evidence_info(best) if best else (evidence_info(candidates[0]) if candidates else None)

def serialize(record: AssessmentRecord):
    return {"id": record.id, "module": record.module, "created_at": record.created_at.isoformat(), "age": record.age,
            "sex": record.sex, "evidence": json.loads(record.evidence_json), "answers": json.loads(record.answers_json), "result": json.loads(record.result_json)}

def prepare_question(question: dict | None) -> dict | None:
    # Preserve the validated DDXPlus question wording and avoid a slow model call
    # on every assessment step. Gemini is used for the separate review instead.
    return question

@app.get("/api/health")
def health():
    return {"status": "ok", "symptom_model_ready": MODEL_PATH.exists(),
            "gemini_configured": bool(__import__('os').getenv('GEMINI_API_KEY')),
            "indian_model_ready": INDIAN_MODEL_PATH.exists(),
            "rag_engine": retriever.kind}

@app.post("/api/session/anonymous")
def anonymous_session():
    session_id = str(uuid.uuid4())
    return {"token": make_token(session_id, anonymous=True), "email": "Guest"}

@app.post("/api/auth/register")
def register(body: AuthRequest, db: Session = Depends(db_session)):
    email = body.email.strip().lower()
    if not valid_email(email): raise HTTPException(400, "Enter a valid email address")
    if db.query(UserAccount).filter(UserAccount.email == email).first(): raise HTTPException(409, "An account with this email already exists")
    user = UserAccount(id=str(uuid.uuid4()), email=email, password_hash=password_hash(body.password))
    db.add(user); db.commit()
    return {"token": make_token(user.id), "email": user.email}

@app.post("/api/auth/login")
def login(body: AuthRequest, db: Session = Depends(db_session)):
    email = body.email.strip().lower()
    user = db.query(UserAccount).filter(UserAccount.email == email).first()
    if not user or not verify_password(body.password, user.password_hash): raise HTTPException(401, "Email or password is incorrect")
    return {"token": make_token(user.id), "email": user.email}

@app.get("/api/auth/me")
def me(user: UserAccount = Depends(current_user)):
    if user is None: raise HTTPException(401, "A signed session is required")
    return {"id": user.id, "email": user.email}

@app.get("/api/symptoms")
def symptoms():
    return [evidence_info(code) for code in EVIDENCE_BY_CODE]

@app.get("/api/knowledge/status")
def knowledge_status():
    return {"datasets": {"indian_disease_binary.csv": {"used_for_matching": True}},
            "association_rows": None,
            "indian_model_ready": INDIAN_MODEL_PATH.exists(),
            "message": "The app uses separate DDXPlus and 30-disease binary Indian symptom models. Neither is clinically validated."}

@app.post("/api/assessment/parse")
def parse_symptoms(body: ParseRequest):
    urgent = emergency_warning(body.text)
    local = [code for code in parse_free_text(body.text) if code in BINARY_EVIDENCE_CODES]
    gemini_codes = extract_symptoms(body.text, BINARY_EVIDENCE_CODES) if body.allow_gemini else None
    gemini = _current_reported_evidence(gemini_codes, body.text) if gemini_codes is not None else None
    combined = normalize_tokens(local + (gemini or []))
    source = "Gemini + local phrase matching" if gemini is not None and local else "Gemini" if gemini is not None else "local phrase matching"
    return {"evidence": combined, "source": source,
            "indian_results": predict_indian(body.text, limit=10),
            "unmodelled_mentions": detect_unmodelled_mentions(body.text),
            "coverage_note": "These explicitly mentioned symptoms or concerns are outside the DDXPlus model and do not affect its scores.",
            "emergency_warning": urgent, "review_required": True}

@app.post("/api/assessment/start")
def start_assessment(body: StartAssessment, db: Session = Depends(db_session), user: UserAccount = Depends(current_user)):
    if user is None: raise HTTPException(401, "Create a local guest session before starting an assessment")
    if CONSENT_REQUIRED and not body.consent: raise HTTPException(400, "Please provide consent before saving this assessment.")
    selected_details = []
    for token in body.evidence:
        parts = token.split("_@_")
        info = EVIDENCE_BY_CODE.get(parts[0], {})
        selected_details.append(info.get("question_en", ""))
        if len(parts) == 2:
            meaning = info.get("value_meaning", {}).get(str(parts[1]), {})
            if isinstance(meaning, dict): selected_details.append(str(meaning.get("en", "")))
    selected_questions = " ".join(selected_details)
    urgent = emergency_warning(body.free_text + " " + selected_questions)
    if urgent: return {"emergency_warning": urgent, "assessment": None}
    selected_evidence = normalize_tokens(body.evidence)
    if len(selected_evidence) != len(set(body.evidence)):
        raise HTTPException(400, "One or more selected evidence codes or values are invalid")
    # The UI already asks the user to review and select parsed findings. Do not
    # send the same free text to Gemini a second time when starting the assessment.
    parsed_codes = (extract_symptoms(body.free_text, BINARY_EVIDENCE_CODES)
                    if body.free_text and body.allow_gemini and not selected_evidence else None)
    parsed = _current_reported_evidence(parsed_codes, body.free_text) if parsed_codes is not None else None
    local = [code for code in parse_free_text(body.free_text) if code in BINARY_EVIDENCE_CODES]
    evidence = normalize_tokens(selected_evidence + local + (parsed or []))
    reported_symptoms = extract_indian_symptoms(body.free_text)
    if not evidence and not reported_symptoms:
        raise HTTPException(400, "Describe at least one symptom in your own words before starting")
    result = assess(body.age, body.sex, evidence, reported_symptoms=reported_symptoms,
                    reported_text=body.free_text)
    record = AssessmentRecord(id=str(uuid.uuid4()), user_id=user.id if user else None, module="symptom", age=body.age, sex=body.sex,
                             evidence_json=json.dumps(evidence), result_json=json.dumps(result), consent=int(body.consent))
    db.add(record); db.commit(); db.refresh(record)
    question = prepare_question(next_question(body.age, body.sex, evidence,
                                              reported_text=body.free_text)) if evidence else None
    return {"emergency_warning": None, "assessment": serialize(record), "next_question": question}

@app.post("/api/assessment/{assessment_id}/answer")
def answer(assessment_id: str, body: AnswerRequest, db: Session = Depends(db_session), user: UserAccount = Depends(current_user)):
    if user is None: raise HTTPException(401, "A local session is required")
    record = db.get(AssessmentRecord, assessment_id)
    if record is None or record.module != "symptom" or record.user_id != (user.id if user else None): raise HTTPException(404, "Assessment not found")
    previous_answers = json.loads(record.answers_json)
    existing_evidence = json.loads(record.evidence_json)
    answered_codes_before = [k for k, v in previous_answers.items() if v in {"yes", "no", "unsure"}]
    previous_result = json.loads(record.result_json)
    reported_text = previous_result.get("reported_text", "")
    expected_question = next_question(record.age, record.sex, existing_evidence,
                                      answered_codes_before, previous_answers,
                                      reported_text=reported_text)
    answer_code = body.code or (body.evidence.split("_@_")[0] if body.evidence else None)
    if expected_question is None or answer_code != expected_question["code"]:
        raise HTTPException(400, "Answer the currently suggested follow-up question")
    if body.evidence:
        if body.response != "yes": raise HTTPException(400, "A selected positive finding must use response='yes'")
        token = normalize_tokens([body.evidence])
        if not token: raise HTTPException(400, "Choose a valid symptom or evidence value")
        token_code = token[0].split("_@_")[0]
        if body.code and body.code != token_code: raise HTTPException(400, "Answer code must match the selected evidence")
        code = token_code
        evidence = normalize_tokens(existing_evidence + token)
        previous_answers[code] = "yes"
    else:
        evidence = existing_evidence
        if body.response == "yes": raise HTTPException(400, "Select an evidence item for a yes response")
        if body.code not in EVIDENCE_BY_CODE: raise HTTPException(400, "Choose the question being answered")
        previous_answers[body.code] = body.response
    urgent_text = " ".join(filter(None, [
        expected_question.get("question", ""),
        body.evidence or "",
        body.response if body.response == "yes" else "",
    ]))
    urgent = emergency_warning(urgent_text)
    if urgent:
        return {"emergency_warning": urgent, "assessment": None,
                "next_question": None, "followup_count": len(previous_answers),
                "followup_limit": MAX_FOLLOWUPS}
    record.evidence_json = json.dumps(evidence); record.answers_json = json.dumps(previous_answers)
    answered_codes = [k for k, v in previous_answers.items() if v in {"yes", "no", "unsure"}]
    result = assess(record.age, record.sex, evidence, previous_answers,
                    previous_result.get("reported_symptoms", []),
                    reported_text)
    record.result_json = json.dumps(result)
    db.commit(); db.refresh(record)
    question = prepare_question(next_question(record.age, record.sex, evidence,
                                              answered_codes, previous_answers,
                                              reported_text=reported_text))
    return {"assessment": serialize(record), "next_question": question,
            "followup_count": len(answered_codes), "followup_limit": MAX_FOLLOWUPS}

@app.post("/api/assessment/{assessment_id}/explain")
def explain_assessment(assessment_id: str, body: ExplanationRequest, db: Session = Depends(db_session), user: UserAccount = Depends(current_user)):
    if not body.allow_gemini: raise HTTPException(400, "Explicit Gemini sharing consent is required for an AI explanation")
    if user is None: raise HTTPException(401, "A local session is required")
    record = db.get(AssessmentRecord, assessment_id)
    if record is None or record.module != "symptom" or record.user_id != (user.id if user else None): raise HTTPException(404, "Assessment not found")
    evidence = json.loads(record.evidence_json)
    labels = [evidence_info(token.split("_@_")[0])["question"] for token in evidence if evidence_info(token.split("_@_")[0])]
    result = json.loads(record.result_json)
    sources = retriever.search(f"{HEALTH_REGION} health context {' '.join(labels)} {body.free_text}", k=8)
    model_conditions = list(dict.fromkeys(
        [*CONDITIONS.keys(), *(condition_label(name) for name in CONDITIONS)]
    ))
    answer_labels = []
    answers = json.loads(record.answers_json)
    for code, response in answers.items():
        info = evidence_info(code)
        if not info:
            continue
        answer_text = {"yes": "Yes", "no": "No", "unsure": "Not sure"}.get(response)
        if answer_text:
            answer_labels.append(f"Follow-up: {info['question']} — {answer_text}")
    review_args = (result.get("model_candidates", result["results"]), labels + answer_labels,
                   model_conditions, sources, body.free_text, HEALTH_REGION)
    knowledge_matches = list(result.get("knowledge_matches", []))
    combined_text = " ".join([body.free_text, " ".join(labels), " ".join(answer_labels)]).lower()
    outside_candidates = [
        ("Typhoid fever", "typhoid", "enteric fever"),
        ("Dengue", "dengue", "breakbone fever"),
        ("Malaria", "malaria", "plasmodium"),
    ]
    for condition, *terms in outside_candidates:
        if any(term in combined_text for term in terms):
            knowledge_matches.append({
                "condition": condition,
                "matched_symptoms": [term for term in terms if term in combined_text],
                "sources": ["trusted medical reference"],
                "label": "Reference-guided outside-model consideration",
            })
    if "knowledge_matches" in pyinspect.signature(explain_rankings).parameters:
        review = explain_rankings(*review_args, knowledge_matches=knowledge_matches)
    else:  # Keep endpoint stubs and downstream integrations using the earlier six-argument API compatible.
        review = explain_rankings(*review_args)
    if not review: raise HTTPException(503, "Gemini is temporarily unavailable or timed out. Your symptom assessment and ML ranking are unchanged; you can retry later.")
    return {"explanation": "",
            "final_shortlist": review.get("final_shortlist", []),
            "reviewed_shortlist": review.get("reviewed_shortlist", []),
            "outside_model_considerations": review["outside_model_considerations"],
            "source": "Google Gemini with retrieved trusted references",
            "disclaimer": "The final list combines DDXPlus and the Indian binary model for educational review only. It does not diagnose or assign probabilities. Original model scores remain unchanged."}

@app.get("/api/history")
def history(db: Session = Depends(db_session), user: UserAccount = Depends(current_user)):
    if user is None: raise HTTPException(401, "A local session is required")
    return [serialize(x) for x in db.query(AssessmentRecord).filter(AssessmentRecord.user_id == (user.id if user else None)).order_by(AssessmentRecord.created_at.desc()).limit(100).all()]

@app.delete("/api/history")
def clear_history(db: Session = Depends(db_session), user: UserAccount = Depends(current_user)):
    if user is None: raise HTTPException(401, "A local session is required")
    deleted = db.query(AssessmentRecord).filter(AssessmentRecord.user_id == (user.id if user else None)).delete(synchronize_session=False)
    db.commit()
    return {"deleted": deleted}

@app.post("/api/assistant")
def assistant(body: ChatRequest):
    urgent = emergency_warning(body.message)
    if urgent: return {"answer": urgent, "sources": [], "provider": "Emergency safety check"}
    sources = retriever.search(body.message)
    snippets = "\n".join(f"- {s['title']}: {s['summary']} Source: {s['url']}" for s in sources)
    answer_text = None
    used_gemini = False
    if body.allow_gemini:
        prompt = (
            "You are a helpful general-purpose assistant in an educational college project. Answer the user's question "
            "directly and naturally, including greetings and topics not covered by the supplied references. Do not refuse "
            "only because retrieval found no matching source. Use supplied references when they are relevant, and never "
            "invent citations or claim a reference says something it does not.\n"
            "For health questions, provide general educational information only: do not diagnose, prescribe medicines, "
            "recommend dosages, rule out serious illness, or imply certainty. Encourage a clinician for personal medical "
            "decisions. If the question describes possible emergency symptoms, advise contacting local emergency services "
            "immediately. For non-health questions, answer normally.\n"
            "Relevant retrieved references (may be empty):\n" + (snippets or "None found") +
            "\nUser's question:\n" + body.message
        )
        answer_text = gemini_text(prompt)
        used_gemini = bool(answer_text)
    if not answer_text:
        if sources:
            answer_text = "Gemini is unavailable or wasn't enabled. Here is information from the app's references: " + " ".join(s["summary"] for s in sources[:2])
        elif not body.allow_gemini:
            answer_text = "I couldn't find a built-in reference for that question. Enable Google Gemini below to get a general answer, or ask about a topic covered by the app's references."
        else:
            answer_text = "Gemini couldn't answer right now, and I couldn't find a relevant built-in reference. Please try again later."
    return {"answer": answer_text,
            "sources": [{"title": s["title"], "url": s["url"], "publisher": s["publisher"]} for s in sources],
            "provider": "Google Gemini" if used_gemini else "Built-in references"}

@app.get("/api/about")
def about():
    return {"disclaimer": "For education only. This application is not a medical device and does not diagnose or replace a licensed clinician. DDXPlus is synthetic and the Indian binary model is a 30-condition pattern prototype; both require clinical validation.",
            "sources": SOURCES}
