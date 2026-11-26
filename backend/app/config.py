import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "db"
ARTIFACTS = BACKEND_DIR / "artifacts"

def configured_path(name: str, default: Path, relative_to: Path) -> Path:
    value = os.getenv(name)
    path = Path(value) if value else default
    return path if path.is_absolute() else relative_to / path

MODEL_PATH = configured_path("MODEL_PATH", ARTIFACTS / "ddxplus_model.joblib", BACKEND_DIR)
HEART_MODEL_PATH = configured_path("HEART_MODEL_PATH", ARTIFACTS / "heart_model.joblib", BACKEND_DIR)
HEART_DATASET_PATH = configured_path("HEART_DATASET_PATH", DB_DIR / "framingham.csv", ROOT)
RAG_INDEX_PATH = configured_path("RAG_INDEX_PATH", ARTIFACTS / "rag", BACKEND_DIR)
DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///{(ARTIFACTS / 'health_assessments.sqlite3').as_posix()}"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "35000"))
HEALTH_REGION = os.getenv("HEALTH_REGION", "India")
CONSENT_REQUIRED = os.getenv("CONSENT_REQUIRED", "true").lower() == "true"
APP_SECRET = os.getenv("APP_SECRET") or secrets.token_urlsafe(48)
ARTIFACTS.mkdir(parents=True, exist_ok=True)
