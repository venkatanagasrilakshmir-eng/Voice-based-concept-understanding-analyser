"""
config.py
Central configuration for the VBCUA application.
Loads settings from environment variables (.env file) with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    # --- Directories ---
    BASE_DIR: Path = BASE_DIR
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "uploads"))
    REPORT_DIR: Path = Path(os.getenv("REPORT_DIR", BASE_DIR / "reports"))
    DATA_DIR: Path = BASE_DIR / "data"

    # --- Database ---
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/vbcua.db")

    # --- Models ---
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")
    SENTENCE_BERT_MODEL: str = os.getenv("SENTENCE_BERT_MODEL", "all-MiniLM-L6-v2")

    # --- Gemini (optional LLM insights) ---
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # --- API server ---
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # --- Scoring weights (must sum to 1.0) ---
    WEIGHT_UNDERSTANDING: float = 0.5
    WEIGHT_FLUENCY: float = 0.3
    WEIGHT_CLARITY: float = 0.2

    def ensure_dirs(self):
        for d in (self.UPLOAD_DIR, self.REPORT_DIR, self.DATA_DIR):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
