<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# Wellwise health assessment

A local, research-oriented application for symptom-based assessment. Model rankings are educational, not diagnoses. Emergency phrase matches show an urgent-care warning and skip scoring.

## Included

- React + Vite responsive interface: free-text symptom intake, optional consented Gemini extraction, adaptive follow-up flow, ranked DDXPlus assessment, separate CSV symptom-overlap matches, consented Gemini review, open-ended assistant, saved history, and medical disclaimer. Users describe symptoms in their own words and do not select from a symptom list. The frontend uses signed, anonymous guest sessions to keep history isolated without a login screen; the backend retains its optional legacy account endpoints.
- FastAPI backend with DDXPlus mappings, local phrase matching, free-text symptom matching against the supplied CSV vocabulary (including headache and muscle/body aches), explicit outside-model mention recognition for common cold, headache/migraine, and stomach/abdominal pain, optional Gemini extraction and general-purpose assistant, safety checks, and Sentence Transformers + ChromaDB retrieval when available (keyword fallback otherwise).
- Scikit-learn benchmark comparing a Decision Tree, Random Forest, and Linear SVM across four regularization settings. Features are AGE, SEX, and EVIDENCES; PATHOLOGY is the target. `DIFFERENTIAL_DIAGNOSIS` and `INITIAL_EVIDENCE` are excluded from features.
- A transparent symptom overlap lookup using `Disease and symptoms dataset.csv` and `symtoms_df.csv`. Free-text terms such as headache can appear in this separate lookup even when DDXPlus has no supported input for them. In that case the app shows no DDXPlus ranking rather than inventing one. The supplied `improved_disease_dataset.csv` remains in `db/` but is withheld from matching because its symptom patterns look noisy and require source/quality verification. CSV overlaps are not trained or evaluated predictions and are displayed separately from DDXPlus scores.
- SQLite is the local default; PostgreSQL is supported with `DATABASE_URL`.

## Dataset inspection

### Synthea synthetic benchmark

The `db/synthea/` directory contains 1,000 fictional patients generated locally with
MITRE Synthea using deterministic seed `20261121`. It can be used for application,
data-pipeline, and software benchmarks. It is not real patient data, is not calibrated
for India, and must not be used to claim real-world diagnostic accuracy.

Run the educational condition benchmark with:

```powershell
python -m backend.train_synthea
```

The result is written to `db/synthea/synthea_benchmark.json` and reports top-1 accuracy
and top-5 recall for the synthetic population. Do not compare those values directly
with a real clinical deployment target.

The preserved `db/` directory contains:

- `release_train_patients` (1,025,601 rows; CSV)
- `release_validate_patients` (132,448 rows; CSV)
- `release_test_patients 1` (134,529 rows; CSV; trailing ` 1` is part of the supplied filename)
- `release_evidences.json` (223 evidence definitions)
- `release_conditions.json` (49 condition mappings)
- `Disease and symptoms dataset.csv` and `symtoms_df.csv` (loaded lazily on first assessment; used only for the separately labeled symptom overlap lookup)
- `improved_disease_dataset.csv` (retained but not used until its row-level symptom labels and source can be verified)

Patient columns are `AGE`, `DIFFERENTIAL_DIAGNOSIS`, `SEX`, `PATHOLOGY`, `EVIDENCES`, and `INITIAL_EVIDENCE`. Categorical/multi-choice evidence is represented as `E_code_@_value`; binary evidence uses its code. Original data files are not modified.

## Run locally

Requirements: Python 3.11+, Node 20+, npm. PostgreSQL is optional for a local research run.

1. Create a virtual environment and install backend dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r backend\requirements.txt
   Copy-Item backend\.env.example backend\.env
   ```

2. Set a strong `APP_SECRET` and optionally configure `DATABASE_URL` and `GEMINI_API_KEY` in the process environment before starting the API.

3. Train and compare Decision Tree, Random Forest, and four Linear SVM regularization settings on deterministic reservoir samples from the official splits:

   ```powershell
   python -m backend.train_ddxplus
   ```

   Defaults: 50,000 train, 12,000 validation, and 16,000 held-out test rows. The script scans each file and preserves official split boundaries. Adjust with `--train-sample`, `--valid-sample`, and `--test-sample` if you have more time and memory. The selected model is based on validation macro-F1; held-out metrics and confusion matrices are saved in `backend/artifacts/ddxplus_evaluation.json`. Artifact: `backend/artifacts/ddxplus_model.joblib`. This sample benchmark is not full-population or clinical validation. Scores are explicitly not clinically calibrated probabilities. DDXPlus is a synthetic 49-condition set; its generated population uses U.S. Census-derived priors, so this model is not calibrated for India and cannot rank conditions outside its label set.

4. Start FastAPI:

   ```powershell
   uvicorn backend.app.main:app --reload --port 8000
   ```

   Run the backend workflow/safety checks from the project root with `python -m pytest backend/tests/test_api.py -q`.

5. In another terminal, start the UI:

   ```powershell
   cd frontend
   npm install
   npm run dev
   ```

   Open `http://localhost:5173`.

6. Optional Gemini: set `GEMINI_API_KEY`, `GEMINI_MODEL` (default `gemini-3.6-flash`), and optionally `GEMINI_TIMEOUT_MS` (default 35000), and `HEALTH_REGION` (default `India`) before starting the API. Calls use a bounded timeout and at most two attempts; if Gemini is unavailable, local extraction and the model/CSV results remain available. Each assistant user must opt in before their question is sent to Google. With consent enabled, the general-purpose assistant can answer even when retrieval finds no supplied reference; when disabled, it uses matching built-in references only. The assistant cannot diagnose, prescribe, or rule out emergencies. The symptom parser separately recognizes common cold, headache/migraine, and stomach/abdominal pain; these are not converted into DDXPlus evidence or scores. After follow-ups, Gemini may organize up to five candidates from the DDXPlus ranking and CSV overlap list; it cannot add names outside those lists. This is not a validated prediction; the original DDXPlus scores remain unchanged. Gemini may also show up to two unranked outside-model considerations when supported by retrieved trusted references.

   To enable semantic retrieval, install the Sentence Transformers model once while online (the app itself does not download models on startup):

   ```powershell
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
   ```

   On the next API startup, the model builds a local ChromaDB index. Until then the API uses its built-in keyword retriever.

7. Optional PostgreSQL: set `DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME`. Use a private database and TLS for remote connections. SQLite is created in `backend/artifacts/` by default.

## Safety and privacy

- The app does not prescribe medications or dosages.
- Emergency phrase checks advise contacting local emergency services and skip prediction.
- Account support is for local research use, not a substitute for security review or clinical deployment controls. Before any shared deployment, use HTTPS, a strong `APP_SECRET`, protected database backups, and a security review. Do not enter identifiable health data into a public/shared instance.
- Protect credentials, licensed data, database backups, and generated model artifacts. Never commit API keys or private health records.
- DDXPlus is synthetic; its own documentation warns against deployment without rigorous evaluation.

## Source references

Built-in references include U.S. National Library of Medicine MedlinePlus emergency/chest-pain guidance, WHO sepsis information, and NHLBI information about Framingham and BioLINCC. The compact reference set is not a comprehensive medical knowledge base.
=======
# disease-predictor
it predict disease
>>>>>>> af5fd92d473e49560fef9880d14bedc792e3a451
=======
# disease-predictor
it predict disease
>>>>>>> af5fd92d473e49560fef9880d14bedc792e3a451
=======
# disease-predictor
it predict disease
>>>>>>> af5fd92d473e49560fef9880d14bedc792e3a451
