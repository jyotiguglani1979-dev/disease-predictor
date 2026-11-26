import json, os, tempfile, uuid
from pathlib import Path

# Isolate test records from the application's configured database.
_database_file = Path(tempfile.gettempdir()) / f"wellwise-tests-{uuid.uuid4().hex}.sqlite3"
os.environ["DATABASE_URL"] = f"sqlite:///{_database_file.as_posix()}"
os.environ["APP_SECRET"] = "test-only-secret-not-for-deployment"

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import ValidationError
from backend.app import main
from backend.app.database import SessionLocal
from backend.app.models import UserAccount
from backend.app.schemas import AuthRequest, StartAssessment, AnswerRequest, ExplanationRequest, ParseRequest, ChatRequest

def make_account(email):
    db = SessionLocal()
    try:
        result = main.register(AuthRequest(email=email, password="long-test-password-123"), db)
        assert result["email"] == email
        with pytest.raises(HTTPException) as duplicate:
            main.register(AuthRequest(email=email, password="long-test-password-123"), db)
        assert duplicate.value.status_code == 409
        login = main.login(AuthRequest(email=email.upper(), password="long-test-password-123"), db)
        assert login["email"] == email
        return db.query(UserAccount).filter_by(email=email).one()
    finally: db.close()

def test_anonymous_guest_sessions_isolate_history_without_login():
    first = main.anonymous_session()
    second = main.anonymous_session()
    guest_one = main.current_user(HTTPAuthorizationCredentials(scheme="Bearer", credentials=first["token"]))
    guest_two = main.current_user(HTTPAuthorizationCredentials(scheme="Bearer", credentials=second["token"]))
    assert guest_one.id != guest_two.id
    db = SessionLocal()
    try:
        started = main.start_assessment(StartAssessment(age=32, sex="Female", evidence=["E_91"], consent=True), db, guest_one)
        assert len(main.history(db, guest_one)) == 1
        assert main.history(db, guest_two) == []
        with pytest.raises(HTTPException) as anonymous:
            main.history(db, None)
        assert anonymous.value.status_code == 401
        main.clear_history(db, guest_one)
    finally:
        db.close()

def test_symptom_assessment_safety_consent_followup_and_account_isolation():
    account = make_account("one@example.test")
    other = make_account("two@example.test")
    db = SessionLocal()
    try:
        with pytest.raises(HTTPException) as denied:
            main.start_assessment(StartAssessment(age=35, sex="Female", evidence=["E_91"]), db, account)
        assert denied.value.status_code == 400
        with pytest.raises(HTTPException) as empty:
            main.start_assessment(StartAssessment(age=35, sex="Female", evidence=[], consent=True), db, account)
        assert empty.value.status_code == 400
        with pytest.raises(HTTPException) as invalid:
            main.start_assessment(StartAssessment(age=35, sex="Female", evidence=["E_invalid"], consent=True), db, account)
        assert invalid.value.status_code == 400
        urgent = main.start_assessment(StartAssessment(age=35, sex="Female", evidence=["E_91"], free_text="I have severe chest pain", consent=True), db, account)
        assert urgent["emergency_warning"] and urgent["assessment"] is None
        started = main.start_assessment(StartAssessment(age=35, sex="Female", evidence=["E_91", "E_91"], consent=True), db, account)
        payload = started["assessment"]
        assert payload["evidence"] == ["E_91"]
        assert payload["result"]["results"] and "not a calibrated probability" in payload["result"]["score_label"].lower()
        assert "unknown, not absent" in payload["result"]["explanation"]
        assert "not calibrated for india" in payload["result"]["explanation"].lower()
        assert started["next_question"]
        with pytest.raises(HTTPException) as mismatch:
            main.answer(payload["id"], AnswerRequest(code="E_91", response="no"), db, account)
        assert mismatch.value.status_code == 400
        with pytest.raises(HTTPException) as sharing:
            main.explain_assessment(payload["id"], ExplanationRequest(allow_gemini=False), db, account)
        assert sharing.value.status_code == 400
        question = started["next_question"]
        assert question
        followup = main.answer(payload["id"], AnswerRequest(code=question["code"], response="no"), db, account)
        assert followup["assessment"]["answers"][question["code"]] == "no"
        assert followup["assessment"]["result"]["results"] != payload["result"]["results"]
        with pytest.raises(HTTPException) as repeated:
            main.answer(payload["id"], AnswerRequest(code=question["code"], response="no"), db, account)
        assert repeated.value.status_code == 400
        positive_question = followup["next_question"]
        assert positive_question
        positive_token = positive_question["code"] if positive_question["type"] == "B" else f"{positive_question['code']}_@_{positive_question['options'][0]['value']}"
        positive = main.answer(payload["id"], AnswerRequest(code=positive_question["code"], evidence=positive_token, response="yes"), db, account)
        assert positive_question["code"] in {token.split("_@_")[0] for token in positive["assessment"]["evidence"]}
        with pytest.raises(HTTPException) as hidden:
            main.answer(payload["id"], AnswerRequest(code=question["code"], response="no"), db, other)
        assert hidden.value.status_code == 404
        assert len(main.history(db, account)) == 1
        assert main.history(db, other) == []
        assert main.clear_history(db, account)["deleted"] == 1
        assert main.history(db, account) == []
    finally: db.close()

def test_unsure_answer_stays_unknown_and_advances_followup():
    account = make_account("unsure@example.test")
    db = SessionLocal()
    try:
        started = main.start_assessment(StartAssessment(age=42, sex="Male", evidence=["E_201"], consent=True), db, account)
        question = started["next_question"]
        assert question
        updated = main.answer(started["assessment"]["id"], AnswerRequest(code=question["code"], response="unsure"), db, account)
        assert updated["assessment"]["answers"][question["code"]] == "unsure"
        assert updated["assessment"]["evidence"] == started["assessment"]["evidence"]
        assert updated["assessment"]["result"]["results"] == started["assessment"]["result"]["results"]
        assert updated["next_question"] is None or updated["next_question"]["code"] != question["code"]
    finally: db.close()

def test_followups_stop_at_the_configured_limit_and_emergency_recheck():
    account = make_account("followup-limit@example.test")
    db = SessionLocal()
    try:
        started = main.start_assessment(
            StartAssessment(age=42, sex="Male", evidence=["E_201"], consent=True),
            db,
            account,
        )
        assessment_id = started["assessment"]["id"]
        for _ in range(main.MAX_FOLLOWUPS):
            record = db.get(main.AssessmentRecord, assessment_id)
            answers = json.loads(record.answers_json)
            evidence = json.loads(record.evidence_json)
            answered = list(answers)
            question = main.prepare_question(
                main.next_question(record.age, record.sex, evidence, answered, answers)
            )
            if question is None:
                break
            if question["type"] == "B":
                updated = main.answer(
                    assessment_id,
                    AnswerRequest(code=question["code"], response="no"),
                    db,
                    account,
                )
            else:
                continue
        record = db.get(main.AssessmentRecord, assessment_id)
        assert len(json.loads(record.answers_json)) <= main.MAX_FOLLOWUPS
        assert main.next_question(
            record.age,
            record.sex,
            json.loads(record.evidence_json),
            list(json.loads(record.answers_json)),
            json.loads(record.answers_json),
        ) is None
        assert main.emergency_warning("severe chest pain")
    finally:
        db.close()

def test_natural_language_mapping_emergency_and_reference_fallback(monkeypatch):
    assert main.health()["symptom_model_ready"]
    assert len(main.symptoms()) == 223
    assert main.about()["disclaimer"]
    assert main.condition_label("IVRS ou virémie") == "URTI"
    parsed = main.parse_symptoms(ParseRequest(text="I have a fever and cough"))
    assert set(parsed["evidence"]) == {"E_91", "E_201"}
    assert main.parse_symptoms(ParseRequest(text="I feel nauseous"))["evidence"] == ["E_148"]
    assert "E_91" not in main.parse_symptoms(ParseRequest(text="I do not have a fever"))["evidence"]
    assert "E_91" not in main.parse_symptoms(ParseRequest(text="I have a fever, but I do not have a fever"))["evidence"]
    assert main.parse_symptoms(ParseRequest(text="I have a fever and a fever"))["evidence"].count("E_91") == 1
    assert main.parse_symptoms(ParseRequest(text="an unrecognized phrase xyzzy"))["evidence"] == []
    current_migraine = main.parse_symptoms(ParseRequest(text="I have a migraine headache"))
    assert "E_99" not in current_migraine["evidence"]
    assert {item["label"] for item in current_migraine["unmodelled_mentions"]} == {"Migraine mentioned", "Headache"}
    abdomen = main.parse_symptoms(ParseRequest(text="I have stomach pain"))
    assert {item["label"] for item in abdomen["unmodelled_mentions"]} == {"Stomach or abdominal pain"}
    cold = main.parse_symptoms(ParseRequest(text="I caught a common cold"))
    assert {item["label"] for item in cold["unmodelled_mentions"]} == {"Common cold mentioned"}
    negated = main.parse_symptoms(ParseRequest(text="I do not have a headache or stomach pain"))
    assert negated["unmodelled_mentions"] == []
    assert "E_99" in main.parse_symptoms(ParseRequest(text="My family has a history of migraine"))["evidence"]
    assert main.parse_symptoms(ParseRequest(text="pressure in my chest"))["emergency_warning"]
    assert main.parse_symptoms(ParseRequest(text="no chest pain"))["emergency_warning"] is None
    assert main.parse_symptoms(ParseRequest(text="weakness on one side of my body"))["emergency_warning"]
    monkeypatch.setattr(main, "gemini_text", lambda prompt: (_ for _ in ()).throw(AssertionError("Gemini called without consent")))
    answer = main.assistant(ChatRequest(message="How do I recognize an emergency?", allow_gemini=False))
    assert answer["sources"] and all(source.get("url") for source in answer["sources"])

def test_typhoid_profile_adds_outside_model_consideration():
    from backend.app.indian_disease import extract_indian_symptoms, predict_indian

    text = "rising high fever—often reaching 103–104°F (39–40°C)—along with weakness, stomach pain, headache, and digestive changes for few days"
    symptoms = extract_indian_symptoms(text)
    prediction = predict_indian(text, limit=10)
    assert {"fever", "high fever", "weakness", "abdominal pain", "headache"} <= set(symptoms)
    assert prediction["ready"] is True
    assert prediction["results"][0]["condition"] == "Typhoid"
    assert len(prediction["results"]) == 10

def test_indian_common_disease_cases_and_discriminating_followups():
    cases = {
        "URTI": (["E_91", "E_201", "E_181", "E_97", "E_175"], "URTI"),
        "Influenza": (["E_91", "E_94", "E_201", "E_144", "E_175", "E_181", "E_97"], "Influenza"),
        "Pneumonia": (["E_91", "E_201", "E_77", "E_66", "E_220", "E_45"], "Pneumonia"),
        "Bronchitis": (["E_201", "E_77", "E_214", "E_181", "E_97", "E_66"], "Bronchitis"),
        "Tuberculosis": (["E_201", "E_45", "E_162", "E_91", "E_175", "E_66"], "Tuberculosis"),
        "Viral pharyngitis": (["E_97", "E_91", "E_201", "E_181"], "Viral pharyngitis"),
    }
    for name, (evidence, expected) in cases.items():
        result = main.assess(35, "Female", evidence)
        names = [item["condition"] for item in result["model_candidates"]]
        assert expected in names, f"{name} missing from DDXPlus top ten: {names}"
        question = main.next_question(35, "Female", evidence)
        if name == "Pneumonia":
            assert question is None
        else:
            assert question is not None

    dengue = main.parse_symptoms(ParseRequest(text="I have dengue with sudden fever, rash, headache, muscle pain and nausea"))
    malaria = main.parse_symptoms(ParseRequest(text="I may have malaria with fever, chills and headache"))
    typhoid = main.parse_symptoms(ParseRequest(text="I may have typhoid with prolonged fever, abdominal pain and diarrhoea"))
    labels = {item["label"] for item in dengue["unmodelled_mentions"]}
    assert "Dengue mentioned" in labels
    labels = {item["label"] for item in malaria["unmodelled_mentions"]}
    assert "Malaria mentioned" in labels
    labels = {item["label"] for item in typhoid["unmodelled_mentions"]}
    assert "Typhoid mentioned" in labels

def test_free_text_headache_and_body_aches_use_indian_symptoms():
    from backend.app.indian_disease import extract_indian_symptoms
    from backend.app.clinical import parse_free_text

    assert "headache" in extract_indian_symptoms("I have a headache and body aches")
    assert "muscle pain" in extract_indian_symptoms("I have a headache and body aches")
    assert parse_free_text("I have body aches") == ["E_144"]
    assert parse_free_text("I do not have body aches") == []

def test_headache_only_creates_transparent_coverage_result_without_fake_ml_ranking():
    account = make_account("headache-only@example.test")
    db = SessionLocal()
    try:
        started = main.start_assessment(
            StartAssessment(age=29, sex="Female", free_text="I have a headache", consent=True), db, account
        )
        result = started["assessment"]["result"]
        assert result["coverage_only"] is True
        assert result["results"] == []
        assert result["reported_symptoms"] == ["headache"]
        assert started["next_question"] is None
        assert any("headache" in item["matched_symptoms"] for item in result["knowledge_matches"])
    finally:
        db.close()

def test_health_information_is_not_sent_to_gemini_without_consent(monkeypatch):
    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("Gemini was called without explicit consent")

    monkeypatch.setattr(main, "extract_symptoms", unexpected_call)
    parsed = main.parse_symptoms(ParseRequest(text="I have a fever", allow_gemini=False))
    assert parsed["source"] == "local phrase matching"
    assert "E_91" in parsed["evidence"]

    monkeypatch.setattr(main, "gemini_text", unexpected_call)
    main.assistant(ChatRequest(message="What is a fever?", allow_gemini=False))
    with pytest.raises(HTTPException) as denied:
        main.explain_assessment("missing-assessment", ExplanationRequest(allow_gemini=False), None, None)
    assert denied.value.status_code == 400

def test_assistant_uses_gemini_for_open_questions_without_retrieved_references(monkeypatch):
    captured = {}
    monkeypatch.setattr(main.retriever, "search", lambda *_args, **_kwargs: [])
    def fake_gemini(prompt):
        captured["prompt"] = prompt
        return "Hello! What would you like help with?"
    monkeypatch.setattr(main, "gemini_text", fake_gemini)
    response = main.assistant(ChatRequest(message="hello", allow_gemini=True))
    assert response["answer"].startswith("Hello!")
    assert response["provider"] == "Google Gemini"
    assert response["sources"] == []
    assert "do not refuse" in captured["prompt"].lower()

def test_assistant_no_consent_fallback_explains_how_to_enable_gemini(monkeypatch):
    monkeypatch.setattr(main.retriever, "search", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(main, "gemini_text", lambda *_args, **_kwargs: pytest.fail("Gemini called without consent"))
    response = main.assistant(ChatRequest(message="hello", allow_gemini=False))
    assert "Enable Google Gemini" in response["answer"]
    assert response["provider"] == "Built-in references"

def test_assessment_start_reuses_reviewed_findings_without_second_gemini_call(monkeypatch):
    account = make_account("single-gemini-call@example.test")
    db = SessionLocal()
    try:
        def unexpected_second_extraction(*_args, **_kwargs):
            raise AssertionError("Assessment start should reuse the user's reviewed evidence")

        monkeypatch.setattr(main, "extract_symptoms", unexpected_second_extraction)
        started = main.start_assessment(
            StartAssessment(age=35, sex="Female", evidence=["E_91"], free_text="I have a fever", consent=True, allow_gemini=True),
            db,
            account,
        )
        assert started["assessment"]["evidence"] == ["E_91"]
    finally:
        db.close()

def test_rag_semantic_or_keyword_retrieval_returns_trusted_source(monkeypatch):
    assert main.retriever.kind in {"Sentence Transformers + ChromaDB", "keyword"}
    semantic_or_active = main.retriever.search("What are signs and symptoms of sepsis?")
    assert any(row["title"] == "WHO: Sepsis fact sheet" for row in semantic_or_active)
    assert all(row["url"].startswith("https://") for row in semantic_or_active)

    # Exercise the documented fallback even when this environment has Chroma available.
    monkeypatch.setattr(main.retriever, "collection", None)
    fallback = main.retriever.search("sepsis symptoms")
    assert any(row["title"] == "WHO: Sepsis fact sheet" for row in fallback)
    india_context = main.retriever.search("India prolonged fever typhoid chills", k=8)
    assert any("Typhoid" in row["title"] or "Enteric Fever" in row["title"] for row in india_context)

def test_request_schemas_reject_invalid_and_missing_values():
    with pytest.raises(ValidationError):
        StartAssessment(age=121, sex="Female", evidence=["E_91"], consent=True)
    with pytest.raises(ValidationError):
        StartAssessment(age=35, sex="unknown", evidence=["E_91"], consent=True)
    with pytest.raises(ValidationError):
        ParseRequest(text="")
    with pytest.raises(ValidationError):
        AnswerRequest(code="E_91", response="maybe")
    with pytest.raises(ValidationError):
        AuthRequest(email="person@example.test", password="short")

def test_gemini_client_uses_bounded_retries_and_closes_without_leaking_prompt(monkeypatch):
    from google import genai
    from backend.app import llm

    captured = {}

    class FakeModels:
        def generate_content(self, **_kwargs):
            raise TimeoutError("provider timed out")

    class FakeClient:
        def __init__(self, *, api_key, http_options):
            captured["api_key"] = api_key
            captured["http_options"] = http_options
            self.models = FakeModels()

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(llm, "GEMINI_API_KEY", "test-key-must-not-be-logged")
    monkeypatch.setattr(genai, "Client", FakeClient)
    assert llm.gemini_text("private symptom text") is None
    options = captured["http_options"]
    assert options.timeout == llm.GEMINI_TIMEOUT_MS
    assert options.retry_options.attempts == 2
    assert captured["closed"] is True

def test_gemini_review_combines_candidates_and_does_not_change_ml_result(monkeypatch):
    account = make_account("gemini-review@example.test")
    db = SessionLocal()
    try:
        started = main.start_assessment(StartAssessment(age=35, sex="Female", evidence=["E_91"], consent=True), db, account)
        assessment = started["assessment"]
        original_results = assessment["result"]["results"]
        saved_record = db.get(main.AssessmentRecord, assessment["id"])
        saved_record.answers_json = json.dumps({"E_93": "no"})
        db.commit()
        received = {}

        def review(results, labels, model_conditions, sources, free_text, region):
            received["results"] = results
            received["labels"] = labels
            received["model_conditions"] = model_conditions
            received["sources"] = sources
            received["free_text"] = free_text
            received["region"] = region
            return {"summary": "Combined educational review.", "final_shortlist": [], "outside_model_considerations": []}

        monkeypatch.setattr(main.retriever, "search", lambda *_args, **_kwargs: [main.SOURCES[5]])
        monkeypatch.setattr(main, "explain_rankings", review)
        response = main.explain_assessment(assessment["id"], ExplanationRequest(allow_gemini=True, free_text="fever for one day"), db, account)
        assert response["explanation"] == ""
        assert "does not diagnose" in response["disclaimer"].lower()
        assert "probabilities" in response["disclaimer"].lower()
        assert received["results"][:5] == original_results
        assert len(received["results"]) == 10
        assert received["labels"]
        assert any("Follow-up:" in label and "No" in label for label in received["labels"])
        assert received["region"] == "India"
        assert received["free_text"] == "fever for one day"
        saved = db.get(main.AssessmentRecord, assessment["id"])
        assert json.loads(saved.result_json) == assessment["result"]

        monkeypatch.setattr(main, "explain_rankings", lambda *_args, **_kwargs: None)
        with pytest.raises(HTTPException) as unavailable:
            main.explain_assessment(assessment["id"], ExplanationRequest(allow_gemini=True), db, account)
        assert unavailable.value.status_code == 503
        assert "unchanged" in unavailable.value.detail.lower()
    finally:
        db.close()

def test_gemini_review_prompt_forbids_diagnosis_and_probability_claims(monkeypatch):
    from backend.app import llm

    captured = {}
    response = {
        "summary": "The model is not calibrated for India.",
        "outside_model_considerations": [{
            "condition": "typhoid fever",
            "symptom_overlap": "The reference describes prolonged fever.",
            "user_evidence_quote": "prolonged fever",
            "source_quote": "prolonged fever",
            "source_title": "WHO: Typhoid fact sheet",
        }],
    }

    def fake_gemini(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["json_response"] = kwargs.get("json_response")
        return json.dumps(response)

    monkeypatch.setattr(llm, "gemini_text", fake_gemini)
    india_source = next(source for source in main.SOURCES if source["title"] == "WHO: Typhoid fact sheet")
    review = llm.explain_rankings([{"condition": "Example", "score": 1.2}], ["fever"], ["Example"], [india_source], "prolonged fever", "India")
    prompt = captured["prompt"].lower()
    assert "cannot diagnose, confirm, validate" in prompt
    assert "assign percentages" in prompt
    assert "reported findings" in prompt
    assert "not calibrated for india" in prompt
    assert "not diagnoses or predictions" in prompt
    assert captured["json_response"] is True
    assert review["outside_model_considerations"][0]["source_url"] == india_source["url"]

    blocked = dict(response)
    blocked["outside_model_considerations"] = [{
        "condition": "Example",
        "symptom_overlap": "This is in the ML label list.",
        "source_title": "WHO: Typhoid fact sheet",
    }]
    monkeypatch.setattr(llm, "gemini_text", lambda *_args, **_kwargs: json.dumps(blocked))
    review_with_duplicate = llm.explain_rankings([], [], ["Example"], [india_source])
    assert review_with_duplicate["outside_model_considerations"] == []

def test_indian_model_uses_every_reported_symptom_it_has_a_column_for():
    """Names such as 'fever' and 'shortness of breath' used to be silently dropped."""
    from backend.app.indian_disease import predict_indian

    for text in ("I have fever", "I have shortness of breath", "I feel weakness"):
        result = predict_indian(text, limit=5)
        assert result["ready"] and result["results"], text
        assert result["not_modelled_symptoms"] == [], text
    # Scores are real model probabilities, so the top 10 cannot add to more than 100.
    result = predict_indian("fever chills shivering sweating headache vomiting", limit=10)
    assert sum(item["score"] for item in result["results"]) <= 100.5
    assert result["results"][0]["condition"] == "Malaria"

def test_indian_model_reports_symptoms_it_cannot_use_and_does_not_guess():
    from backend.app.indian_disease import predict_indian

    result = predict_indian("sore throat", limit=5)
    assert result["results"] == []
    assert result["not_modelled_symptoms"] == ["sore throat"]
    mixed = predict_indian("sore throat and fever with chills", limit=5)
    assert mixed["results"] and mixed["not_modelled_symptoms"] == ["sore throat"]

def test_indian_model_evaluation_report_is_not_inflated():
    import json
    from backend.app.config import INDIAN_MODEL_PATH

    report = json.loads(INDIAN_MODEL_PATH.with_name("indian_evaluation.json").read_text(encoding="utf-8"))
    assert report["unique_symptom_patterns"] < report["raw_rows"] / 10
    assert "split by symptom pattern" in report["evaluation"]
    assert report["unseen_pattern_accuracy"]["any_2_symptoms"]["top_1"] < 0.9


def test_combine_rankings_merges_same_disease_and_marks_agreement():
    from backend.app.combine import combine_rankings

    ddx = [{"condition": "URTI", "score": 40.0}, {"condition": "Pneumonia", "score": 30.0},
           {"condition": "Influenza", "score": 20.0}]
    indian = [{"condition": "Common Cold", "score": 50.0}, {"condition": "Dengue", "score": 30.0},
              {"condition": "Pneumonia", "score": 10.0}]
    rows = combine_rankings(ddx, indian, weight_indian=0.5, limit=10)
    names = [row["condition"] for row in rows]
    assert names.count("Common Cold") == 1 and "URTI" not in names  # same disease, one row
    cold = rows[0]
    assert cold["condition"] == "Common Cold" and cold["also_called"] == "URTI"
    assert cold["score"] == 45.0 and cold["models_agree"] and cold["sources"] == ["DDXPlus", "Indian model"]
    pneumonia = next(row for row in rows if row["condition"] == "Pneumonia")
    assert pneumonia["score"] == 20.0 and pneumonia["models_agree"]
    dengue = next(row for row in rows if row["condition"] == "Dengue")
    assert dengue["sources"] == ["Indian model"] and dengue["score"] == 15.0 and not dengue["models_agree"]
    assert [row["score"] for row in rows] == sorted((row["score"] for row in rows), reverse=True)

def test_combine_rankings_uses_full_weight_when_only_one_model_has_input():
    from backend.app.combine import combine_rankings

    only_indian = combine_rankings([], [{"condition": "Malaria", "score": 80.0}])
    assert only_indian[0]["score"] == 80.0 and only_indian[0]["sources"] == ["Indian model"]
    only_ddx = combine_rankings([{"condition": "Influenza", "score": 60.0}], [])
    assert only_ddx[0]["score"] == 60.0 and only_ddx[0]["sources"] == ["DDXPlus"]
    assert combine_rankings([], []) == []

def test_assess_returns_a_combined_ranking_with_both_models():
    from backend.app.main import assess

    result = assess(35, "Female", ["E_91", "E_94", "E_144", "E_175"],
                    reported_text="fever chills shivering sweating headache vomiting")
    combined = result["combined"]
    assert 0 < len(combined) <= 5 and combined[0]["condition"] == "Malaria"
    assert combined[0]["sources"] == ["Indian model"] or combined[0]["models_agree"]
    assert result["combined_note"]
    # Symptoms DDXPlus cannot code still give a combined list from the Indian model alone.
    coverage_only = assess(35, "Female", [], reported_text="fever chills shivering sweating headache vomiting")
    assert coverage_only["combined"] and coverage_only["combined"][0]["sources"] == ["Indian model"]
