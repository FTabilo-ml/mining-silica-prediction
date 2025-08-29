# src/settings.py
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    APP_NAME: str = "Silica Predictor"
    APP_ENV: str = "prod"  # dev, staging, prod
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Tipo de modelo por defecto: "xgb", "lstm" o "transformer"
    MODEL_KIND: str = Field(default="xgb")

    # Rutas a los modelos (ajustadas a los nombres reales en /models)
    MODEL_XGB_PATH: str = Field(default="models/model_xgb.pkl")
    MODEL_LSTM_PATH: str = Field(default="models/model_lstm.pt")
    MODEL_TRANSF_PATH: str = Field(default="models/model_transformer.pt")

    # (Opcional) Conexión a base de datos
    DB_CONN: str | None = None

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
