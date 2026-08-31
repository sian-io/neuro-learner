import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-3.7-flash")
    llm_base_url: str | None = os.getenv("LLM_BASE_URL", None)
    persistence_db_path: str = os.getenv("PERSISTENCE_DB_PATH", "./data/neuro_learner.db")
    target_retention: float = 0.90
    decay_factor: float = 19.0 / 81.0

    def ensure_db_dir(self) -> None:
        db_path = Path(self.persistence_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

settings = Settings()
