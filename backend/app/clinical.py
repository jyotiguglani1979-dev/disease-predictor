import json, re
from pathlib import Path
from .config import DB_DIR

EVIDENCES = json.loads((DB_DIR / "release_evidences.json").read_text(encoding="utf-8"))
CONDITIONS_RAW = json.loads((DB_DIR / "release_conditions.json").read_text(encoding="utf-8"))
EVIDENCE_BY_CODE = {v["name"]: v for v in EVIDENCES.values()}
CONDITIONS = {v["condition_name"]: v for v in CONDITIONS_RAW.values()}
CONDITIONS_BY_ALIAS = {}
for item in CONDITIONS_RAW.values():
    for label in (item.get("condition_name"), item.get("cond-name-eng"), item.get("cond-name-fr")):
        if label: CONDITIONS_BY_ALIAS[label.casefold()] = item
LOCAL_SYMPTOM_ALIASES = {
    "fever": ("E_91",), "high fever": ("E_91",), "chills": ("E_94",), "shivers": ("E_94",),
    "cough": ("E_201",), "coughing": ("E_201",), "nausea": ("E_148",), "nauseous": ("E_148",),
    "vomiting": ("E_148",), "vomiting repeatedly": ("E_211",), "repeated vomiting": ("E_211",),
    "diarrhea": ("E_51",), "diarrhoea": ("E_51",),
    "body ache": ("E_144",), "body aches": ("E_144",), "muscle ache": ("E_144",),
    "muscle aches": ("E_144",), "muscle pain": ("E_144",), "myalgia": ("E_144",), "myalgias": ("E_144",),
    "rash": ("E_129",), "skin rash": ("E_129",), "skin lesions": ("E_129",),
    "joint swelling": ("E_151",), "swollen joints": ("E_151",),
    "sore throat": ("E_97",), "runny nose": ("E_181",), "stuffy nose": ("E_181",), "nasal congestion": ("E_181",),
    "wheezing": ("E_214",), "wheeze": ("E_214",), "weight loss": ("E_162",), "unintentional weight loss": ("E_162",),
    "dark urine": ("E_188",), "pale stools": ("E_188",), "fatigue": ("E_175",), "tiredness": ("E_175",),
    "abdominal pain": ("E_53",), "stomach pain": ("E_53",), "joint pain": ("E_53",), "pain behind the eyes": ("E_53",),
}

def evidence_info(code: str) -> dict | None:
    info = EVIDENCE_BY_CODE.get(code)
    if not info:
        return None
    meanings = info.get("value_meaning", {})
    options = []
    for value in info.get("possible-values", []):
        meaning = meanings.get(str(value), {}) if isinstance(meanings, dict) else {}
        label = meaning.get("en", str(value)) if isinstance(meaning, dict) else str(value)
        options.append({"value": value, "label": label})
    return {"code": code, "question": info.get("question_en", code), "type": info.get("data_type", "B"), "is_antecedent": info.get("is_antecedent", False), "options": options}

def parse_free_text(text: str) -> list[str]:
    """Conservative local symptom matching; avoid assigning evidence from generic words."""
    folded = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    folded = " " + " ".join(folded.split()) + " "
    def affirmed(phrase: str) -> bool:
        phrase = " ".join(re.sub(r"[^a-z0-9 ]", " ", phrase.lower()).split())
        pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
        states = []
        for match in re.finditer(pattern, folded):
            context = folded[max(0, match.start() - 45):match.start()]
            context = re.split(r"\b(?:but|however|except)\b", context)[-1]
            negated = re.search(r"(?:\bno\b|\bnot\b|\bnever\b|\bwithout\b|don t|doesn t|didn t|do not)(?:\s+\w+){0,4}\s*$", context)
            states.append(not bool(negated))
        # Conflicting mentions are left unparsed so the user must resolve them explicitly.
        return bool(states) and all(states)
    matches = []
    for phrase, codes in LOCAL_SYMPTOM_ALIASES.items():
        if affirmed(phrase):
            matches.extend(code for code in codes if code in EVIDENCE_BY_CODE and EVIDENCE_BY_CODE[code].get("data_type") == "B")
    # Preserve support for a user quoting an exact DDXPlus question, without
    # extracting arbitrary single words (e.g. "pain", "body", or "severe").
    for code, info in EVIDENCE_BY_CODE.items():
        question = info.get("question_en", "").strip().rstrip("?.").lower()
        if question and affirmed(question):
            matches.append(code)
    # Headache and migraine are history-only DDXPlus items; current mentions
    # must not be mistaken for family or prior history.
    if re.search(r"\b(family|relative|mother|father|sister|brother|parent|sibling)\b", folded) and re.search(r"\bcluster headaches?\b", folded) and affirmed("cluster headache"):
        matches.append("E_25")
    history_context = re.search(r"\b(history|previously|previous|ever|before|past|family|relative|mother|father|sister|brother|parent|sibling)\b", folded)
    if history_context and re.search(r"\bmigraines?\b", folded) and affirmed("migraine"):
        matches.append("E_99")
    return list(dict.fromkeys(matches))

def emergency_warning(text: str) -> str | None:
    phrases = [
        r"(severe )?chest (pain|pressure)", r"(?:pain|pressure)\s+(?:in|on|of)\s+(?:my\s+|the\s+)?chest", r"chest pain.{0,40}(sweat|faint|breathless|shortness of breath)",
        r"(cannot|can't|unable to) breathe", r"(severe )?difficulty breathing", r"shortness of breath.{0,30}(severe|sudden|worsening)",
        r"(sudden.{0,30})?(weakness|numbness).{0,30}(one side|one side of (my|the) body|face|arm)",
        r"(face is )?drooping", r"slurred speech", r"sudden confusion", r"unconscious", r"not responding", r"unresponsive",
        r"choking", r"blue lips", r"anaphylaxis", r"severe allergic reaction.{0,40}breath", r"uncontrolled bleeding", r"severe bleeding", r"vomiting blood",
        r"sudden severe headache", r"suicidal.{0,20}(plan|intent|kill)", r"want to (die|kill myself)", r"overdose",
    ]
    folded = " " + " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()) + " "
    def affirmed(match) -> bool:
        context = folded[max(0, match.start() - 45):match.start()]
        context = re.split(r"\b(?:but|however|except)\b", context)[-1]
        return not re.search(r"(?:\bno\b|\bnot\b|\bnever\b|\bwithout\b|don t|doesn t|didn t|do not)(?:\s+\w+){0,4}\s*$", context)
    if any(any(affirmed(m) for m in re.finditer(p, folded)) for p in phrases):
        return "Your message mentions symptoms that can sometimes need urgent care. If you have severe or worsening chest pain, difficulty breathing, stroke signs, fainting, or rapid deterioration, call your local emergency number now. Do not wait for an app assessment."
    return None

def condition_label(name: str) -> str:
    item = CONDITIONS_BY_ALIAS.get(name.casefold(), {})
    return item.get("cond-name-eng") or name
