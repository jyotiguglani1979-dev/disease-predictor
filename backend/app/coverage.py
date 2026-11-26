"""Local recognition for common concerns outside DDXPlus evidence coverage.

These mentions are displayed as user-reported context only. They are never
converted to model evidence or used to rank conditions.
"""
import re

_CATEGORIES = [
    ("Dengue mentioned", "Condition mentioned", r"\b(dengue|breakbone fever)\b",
     "https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue"),
    ("Malaria mentioned", "Condition mentioned", r"\b(malaria|plasmodium)\b",
     "https://www.who.int/news-room/fact-sheets/detail/malaria"),
    ("Typhoid mentioned", "Condition mentioned", r"\b(typhoid|enteric fever)\b",
     "https://www.who.int/news-room/fact-sheets/detail/typhoid"),
    ("Common cold mentioned", "Condition mentioned", r"\b(common cold|cold symptoms|(?:have|got|caught)\s+(?:a\s+)?cold|suffering from\s+(?:a\s+)?cold)\b",
     "https://www.cdc.gov/common-cold/about/index.html"),
    ("Migraine mentioned", "Condition mentioned", r"\bmigraines?\b",
     "https://medlineplus.gov/migraine.html"),
    ("Headache", "Symptom", r"\b(headaches?|head pain|pain in my head|my head hurts)\b",
     "https://medlineplus.gov/migraine.html"),
    ("Stomach or abdominal pain", "Symptom", r"\b(stomach\s*aches?|stomach pain|stomach pains|belly\s*aches?|belly pain|tummy\s*aches?|tummy pain|abdominal pain|stomach cramps|belly cramps|abdominal cramps)\b",
     "https://medlineplus.gov/ency/article/003120.htm"),
]
_NEGATION = re.compile(r"(?:\bno\b|\bnot\b|\bnever\b|\bwithout\b|don t|doesn t|didn t|do not)(?:\s+\w+){0,4}\s*$")


def detect_unmodelled_mentions(text: str) -> list[dict[str, str]]:
    """Return explicit, affirmative user mentions outside DDXPlus coverage."""
    folded = " " + " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()) + " "
    found = []
    for label, kind, pattern, source_url in _CATEGORIES:
        states = []
        for match in re.finditer(pattern, folded):
            context = folded[max(0, match.start() - 50):match.start()]
            context = re.split(r"\b(?:but|however|except)\b", context)[-1]
            states.append(not bool(_NEGATION.search(context)))
        # Contradictory mentions are not silently resolved by the parser.
        if states and all(states):
            found.append({"label": label, "kind": kind, "source_url": source_url})
    return found
