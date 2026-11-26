import json
import logging
import re

from .config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TIMEOUT_MS

logger = logging.getLogger(__name__)

def gemini_text(prompt: str, *, json_response: bool = False) -> str | None:
    if not GEMINI_API_KEY:
        return None
    client = None
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(
                timeout=GEMINI_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(
                    attempts=2,
                    initial_delay=1,
                    max_delay=2,
                ),
            ),
        )
        request_options = {}
        if json_response:
            request_options["config"] = types.GenerateContentConfig(response_mime_type="application/json")
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, **request_options)
        return response.text.strip() if response.text else None
    except Exception as exc:
        # Avoid logging prompts, personal health text, API keys, or provider payloads.
        logger.warning("Gemini request failed (%s)", type(exc).__name__)
        return None
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                logger.debug("Gemini client cleanup failed")

def extract_symptoms(text: str, valid_codes: list[str]) -> list[str] | None:
    prompt = ("Extract only symptoms or relevant history explicitly stated in this user text. "
              "Map them to the provided evidence codes. Do not infer absent symptoms, diagnose, or add advice. "
              "Return only a JSON array of codes from the allowed list.\nAllowed codes: " + json.dumps(valid_codes) +
              "\nUser text: " + text)
    raw = gemini_text(prompt)
    if not raw:
        return None
    try:
        start, end = raw.find("["), raw.rfind("]")
        values = json.loads(raw[start:end + 1])
        return list(dict.fromkeys(v for v in values if v in valid_codes))
    except Exception:
        return None

def explain_rankings(
    results: list[dict],
    evidence_labels: list[str],
    model_conditions: list[str],
    sources: list[dict],
    free_text: str = "",
    region: str = "India",
    knowledge_matches: list[dict] | None = None,
) -> dict | None:
    candidates = {}
    normalize = lambda value: re.sub(r"[^a-z0-9]+", "", str(value).casefold())
    for item in (results or [])[:10]:
        name = item.get("condition", "")
        if name:
            candidates[normalize(name)] = {"condition": name, "origin": "DDXPlus model",
                                           "score": item.get("score")}
    for item in (knowledge_matches or [])[:10]:
        name = item.get("condition", "")
        key = normalize(name)
        if not name or not key: continue
        if key in candidates:
            candidates[key]["origin"] = "DDXPlus model and symptom CSVs"
        else:
            origin = ("Outside model · trusted reference"
                      if item.get("label") == "Reference-guided outside-model consideration"
                      else "symptom CSV overlap")
            candidates[key] = {"condition": name, "origin": origin,
                               "matched_terms": item.get("matched_symptoms", [])}
    candidate_pool = list(candidates.values())
    source_context = [{"title": source["title"], "summary": source["summary"]} for source in sources]
    prompt = (
        "Return one JSON object with keys summary and final_shortlist. summary must be a cautious "
        "educational comparison of the reported findings and supplied ML ranking, at most 90 words. Say that this "
        "review cannot diagnose, confirm, validate, or independently predict a condition. Select at most five names from the candidate pool "
        "for final_shortlist, in relevance order, using both reported findings and follow-up answers. Consider candidates from both "
        "the DDXPlus model and symptom CSVs together. You may also include a condition outside the pool ONLY if it is "
        "explicitly named in a supplied trusted reference summary and the user's reported findings support overlap. "
        "The final_shortlist must contain no more than five total entries. This is an AI-organized review list, not a diagnosis, "
        "validated prediction, probability, or replacement for the original ML ranking; these are not diagnoses or predictions. If evidence is insufficient return []. State that "
        "DDXPlus is synthetic, has 49 conditions, uses U.S. Census-derived population priors, and is not calibrated "
        "for " + region + " or its local disease prevalence. ML scores are relative scores, not probabilities.\n"
        "Each entry has condition, reason, origin. For a candidate-pool item, origin must match its supplied origin. "
        "For an outside-pool item use origin 'Outside model · trusted reference' and also provide symptom_overlap, "
        "user_evidence_quote, source_quote, source_title. Quotes must be short exact substrings of the user's text/findings "
        "and supplied reference summary respectively. Outside-pool items must not be in the complete ML condition list. "
        "Do not assign percentages, probabilities, confidence, likelihood, severity, or treatment. Never say Gemini "
        "confirms or changes the ML ranking.\n"
        "Complete ML condition list: " + json.dumps(model_conditions) +
        "\nCurrent DDXPlus model candidates (first five shown to the user; up to ten available for review): " + json.dumps(results) +
        "\nCombined candidate pool (use exact names only): " + json.dumps(candidate_pool) +
        "\nStructured reported findings: " + json.dumps(evidence_labels) +
        "\nAdditional user wording (may include duration/details): " + json.dumps(free_text) +
        "\nTrusted reference summaries (use exact source_title values): " + json.dumps(source_context) +
        '\nRequired JSON format: {"summary": "...", "final_shortlist": [{"condition": "exact candidate or sourced outside condition", "reason": "...", "origin": "candidate origin", "symptom_overlap": "...", "user_evidence_quote": "...", "source_quote": "...", "source_title": "..."}]}'
    )
    raw = gemini_text(prompt, json_response=True)
    if not raw:
        return None
    try:
        payload = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    except (ValueError, TypeError):
        return None
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if not isinstance(summary, str) or not summary.strip():
        return None

    known_conditions = {normalize(condition) for condition in model_conditions}
    sources_by_title = {source["title"]: source for source in sources}
    reported_text = normalize(" ".join(evidence_labels) + " " + free_text)
    final_shortlist = []
    seen = set()
    raw_shortlist = payload.get("final_shortlist")
    # Accept the prior response shape for existing API stubs/integrations.
    if not isinstance(raw_shortlist, list):
        raw_shortlist = [*(payload.get("reviewed_shortlist", []) or []),
                         *(payload.get("outside_model_considerations", []) or [])]
    for item in raw_shortlist:
        if not isinstance(item, dict):
            continue
        name = item.get("condition")
        if not isinstance(name, str):
            continue
        name = name.strip()[:100]
        key = normalize(name)
        reason = item.get("reason") or item.get("symptom_overlap")
        if not key or key in seen or not isinstance(reason, str) or not normalize(reason):
            continue
        candidate = candidates.get(key)
        if candidate:
            final_shortlist.append({"condition": candidate["condition"], "origin": candidate["origin"],
                                    "reason": reason.strip()[:240], "score": candidate.get("score")})
        else:
            source = sources_by_title.get(item.get("source_title"))
            user_quote, source_quote = item.get("user_evidence_quote"), item.get("source_quote")
            overlap = item.get("symptom_overlap") or reason
            if (not source or not isinstance(user_quote, str) or not isinstance(source_quote, str)
                    or not isinstance(overlap, str) or key in known_conditions):
                continue
            source_text = normalize(source["title"] + " " + source["summary"])
            if (key not in source_text or normalize(source_quote) not in source_text
                    or normalize(user_quote) not in reported_text):
                continue
            final_shortlist.append({"condition": name, "origin": "Outside model · trusted reference",
                                    "reason": reason.strip()[:240], "symptom_overlap": overlap.strip()[:320],
                                    "source_title": source["title"], "source_url": source["url"],
                                    "publisher": source["publisher"]})
        seen.add(key)
        if len(final_shortlist) == 5:
            break
    reviewed = [item for item in final_shortlist if item["origin"] != "Outside model · trusted reference"]
    extra = [item for item in final_shortlist if item["origin"] == "Outside model · trusted reference"]
    return {"summary": summary.strip()[:700], "final_shortlist": final_shortlist,
            "reviewed_shortlist": reviewed, "outside_model_considerations": extra}

