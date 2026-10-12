from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from mlt.config import MODELS_DIR


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLT_")

    knn_model_path: Path = MODELS_DIR / "best_knn_model.joblib"

    decision_threshold: float = 0.5


settings = Settings()
