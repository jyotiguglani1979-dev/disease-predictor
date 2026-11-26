"""Merge the DDXPlus ranking and the Indian-model ranking into one list.

Both models return a probability-like score (0-100) over their OWN disease list.
DDXPlus knows 49 diseases, the Indian model knows 30, and only a few are the same.
We combine them as a weighted mix:

    combined score = (1 - w) * DDXPlus score + w * Indian score

- A disease that only one model knows keeps that model's weighted score.
- A disease both models know gets both parts, so agreement raises it.
- If only one model has usable symptoms, that model gets the full weight.

The scores are a convenient mix, NOT a calibrated probability.
"""

DEFAULT_INDIAN_WEIGHT = 0.5

# Indian-model name (lower case) -> DDXPlus name, for diseases that are the same
# thing under two names. Names that already match (GERD, Pneumonia, Tuberculosis)
# are merged automatically, so they are not listed here.
SAME_DISEASE = {
    "common cold": "URTI",
    "bronchial asthma": "Bronchospasm / acute asthma exacerbation",
    "heart attack": "Possible NSTEMI / STEMI",
}


def _key(name: str) -> str:
    return str(name).strip().casefold()


def combine_rankings(ddx_results, indian_results, weight_indian=DEFAULT_INDIAN_WEIGHT, limit=10):
    """Return one ranked list: [{condition, score, sources, ddxplus_score, indian_score, ...}]."""
    ddx_results = list(ddx_results or [])
    indian_results = list(indian_results or [])
    if not ddx_results and not indian_results:
        return []
    if ddx_results and indian_results:
        w_indian = min(max(float(weight_indian), 0.0), 1.0)
    else:
        w_indian = 1.0 if indian_results else 0.0
    w_ddx = 1.0 - w_indian

    merged = {}
    for item in ddx_results:
        name = str(item["condition"]).strip()
        entry = merged.setdefault(_key(name), {"condition": name, "also_called": None,
                                               "ddxplus_score": None, "indian_score": None})
        entry["ddxplus_score"] = float(item["score"])
    for item in indian_results:
        name = str(item["condition"]).strip()
        ddx_name = SAME_DISEASE.get(_key(name))
        key = _key(ddx_name) if ddx_name else _key(name)
        entry = merged.setdefault(key, {"condition": name, "also_called": None,
                                        "ddxplus_score": None, "indian_score": None})
        entry["indian_score"] = float(item["score"])
        if entry["ddxplus_score"] is not None and _key(entry["condition"]) != _key(name):
            # Same disease, two names: show the everyday (Indian-model) name first.
            entry["also_called"] = entry["condition"]
            entry["condition"] = name

    combined = []
    for entry in merged.values():
        d, i = entry["ddxplus_score"], entry["indian_score"]
        score = w_ddx * (d or 0.0) + w_indian * (i or 0.0)
        sources = [label for label, value in (("DDXPlus", d), ("Indian model", i)) if value is not None]
        combined.append({
            "condition": entry["condition"],
            "also_called": entry["also_called"],
            "score": round(score, 1),
            "sources": sources,
            "models_agree": len(sources) == 2,
            "ddxplus_score": None if d is None else round(d, 1),
            "indian_score": None if i is None else round(i, 1),
        })
    combined.sort(key=lambda row: row["score"], reverse=True)
    return combined[:limit]
