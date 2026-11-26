"""Compare DDXPlus-only, Indian-only and combined rankings on the 10 cases in check_india_cases.

Run from the project root:  python -m backend.check_combined
Saves backend/artifacts/combined_case_check.json
"""
import json
import warnings
from pathlib import Path

from backend.app.combine import DEFAULT_INDIAN_WEIGHT, combine_rankings
from backend.app.indian_disease import predict_indian
from backend.app.main import assess
from backend.check_india_cases import CASES, TEXTS

# Names that count as a correct hit for each case (either model's wording).
ACCEPT = {
    "Dengue": {"dengue"}, "Malaria": {"malaria"}, "Typhoid": {"typhoid"},
    "URTI": {"urti", "common cold"}, "Pneumonia": {"pneumonia"},
    "Tuberculosis": {"tuberculosis"}, "Influenza": {"influenza"},
    "Bronchitis": {"bronchitis"}, "Viral pharyngitis": {"viral pharyngitis"},
    "Gastroenteritis": {"gastroenteritis"},
}
WEIGHTS = [0.0, 0.3, DEFAULT_INDIAN_WEIGHT, 0.7, 1.0]  # 0.0 = DDXPlus only, 1.0 = Indian only


def rank_of(rows, truth):
    for position, row in enumerate(rows, 1):
        names = {str(row["condition"]).strip().casefold()}
        if row.get("also_called"):
            names.add(str(row["also_called"]).strip().casefold())
        if names & truth:
            return position
    return None


def main():
    warnings.filterwarnings("ignore")
    cases = []
    for disease, evidence in CASES.items():
        text = TEXTS[disease]
        ddx = assess(35, "Female", evidence, reported_text=text)["model_candidates"]
        indian = predict_indian(text, limit=10)["results"]
        ranks = {str(w): rank_of(combine_rankings(ddx, indian, weight_indian=w, limit=10), ACCEPT[disease])
                 for w in WEIGHTS}
        cases.append({"case": disease, "rank_of_right_disease_by_indian_weight": ranks})
    summary = {}
    for w in WEIGHTS:
        ranks = [c["rank_of_right_disease_by_indian_weight"][str(w)] for c in cases]
        summary[str(w)] = {f"top_{k}": sum(1 for r in ranks if r and r <= k) for k in (1, 3, 5)}
    report = {
        "cases": len(cases),
        "default_indian_weight": DEFAULT_INDIAN_WEIGHT,
        "note": "weight 0.0 = DDXPlus only, 1.0 = Indian model only. Ten hand-made cases: a sanity check, not proof.",
        "correct_out_of_cases": summary,
        "cases_detail": cases,
    }
    path = Path("backend/artifacts/combined_case_check.json")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"correct_out_of_cases": summary}, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
