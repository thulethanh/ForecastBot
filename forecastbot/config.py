import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_int_env(name: str, default: int = 0) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except Exception:
        return default


def _get_float_env(name: str, default: float = 0.0) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except Exception:
        return default


@dataclass
class Settings:
    app_title: str = os.getenv("APP_TITLE", "ForecastBot Demo")

    # backward compatibility
    postgres_uri: str = os.getenv("POSTGRES_URI", "")

    # multi-database
    forecastbot_db_uri: str = os.getenv("FORECASTBOT_DB_URI", "")
    m5_db_uri: str = os.getenv("M5_DB_URI", "")
    hcmc_bus_db_uri: str = os.getenv("HCMC_BUS_DB_URI", "")

    # cloud LLM
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

    # local forecast via Ollama
    enable_local_forecast: bool = _get_bool_env("ENABLE_LOCAL_FORECAST", True)
    local_forecast_model: str = os.getenv("LOCAL_FORECAST_MODEL", "gemma4:e4b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    local_forecast_temperature: float = _get_float_env(
        "LOCAL_FORECAST_TEMPERATURE", 0.0
    )
    local_forecast_timeout: int = _get_int_env(
        "LOCAL_FORECAST_TIMEOUT", 900
    )

    # forecast experiment settings
    daily_llm_runs: int = _get_int_env("DAILY_LLM_RUNS", 1)

    # weighted history ensemble
    history_short_weight: float = _get_float_env(
        "HISTORY_SHORT_WEIGHT", 0.6
    )
    history_long_weight: float = _get_float_env(
        "HISTORY_LONG_WEIGHT", 0.4
    )


def get_settings() -> Settings:
    return Settings()