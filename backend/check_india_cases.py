"""Check representative Indian disease presentations against the current project."""
import json
from pathlib import Path

from backend.app.coverage import detect_unmodelled_mentions
from backend.app.main import assess, next_question


CASES = {
    "Dengue": ["E_91", "E_94", "E_144", "E_129", "E_148", "E_51"],
    "Malaria": ["E_91", "E_94", "E_144", "E_175"],
    "Typhoid": ["E_91", "E_94", "E_144", "E_53", "E_51", "E_148", "E_175"],
    "URTI": ["E_91", "E_201", "E_181", "E_97", "E_175"],
    "Pneumonia": ["E_91", "E_201", "E_77", "E_66", "E_220", "E_45"],
    "Tuberculosis": ["E_201", "E_45", "E_162", "E_91", "E_175", "E_66"],
    "Influenza": ["E_91", "E_94", "E_201", "E_144", "E_175", "E_181", "E_97"],
    "Bronchitis": ["E_201", "E_77", "E_214", "E_181", "E_97", "E_66"],
    "Viral pharyngitis": ["E_97", "E_91", "E_201", "E_181"],
    "Gastroenteritis": ["E_51", "E_148", "E_211", "E_53", "E_91"],
}

OUTSIDE_MODEL = {"Dengue", "Malaria", "Typhoid", "Gastroenteritis"}


def main():
    report = []
    for disease, evidence in CASES.items():
        result = assess(35, "Female", evidence)
        top_ten = [item["condition"] for item in result["model_candidates"]]
        question = next_question(35, "Female", evidence)
        report.append({
            "disease": disease,
            "status": "Outside DDXPlus label set" if disease in OUTSIDE_MODEL else "Supported by DDXPlus",
            "disease_in_top_ten": disease in top_ten,
            "top_ten": top_ten,
            "follow_up": question["question"] if question else None,
        })

    report_path = Path("backend/artifacts/india_case_check.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
