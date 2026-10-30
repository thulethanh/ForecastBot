import os
from functools import lru_cache

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is not configured.")
    return value


@lru_cache(maxsize=1)
def get_forecastbot_engine():
    return create_engine(_require_env("FORECASTBOT_DB_URI"))


@lru_cache(maxsize=1)
def get_m5_engine():
    return create_engine(_require_env("M5_DB_URI"))


@lru_cache(maxsize=1)
def get_hcmc_bus_engine():
    return create_engine(_require_env("HCMC_BUS_DB_URI"))


def get_engine():
    postgres_uri = os.getenv("POSTGRES_URI")
    if postgres_uri:
        return create_engine(postgres_uri)
    return get_forecastbot_engine()


def get_dataset_engine(dataset_code: str):
    if dataset_code == "M5":
        return get_m5_engine()
    if dataset_code == "HCMC_BUS":
        return get_hcmc_bus_engine()
    raise ValueError(f"Unsupported dataset_code: {dataset_code}")


def fetch_dataframe(sql: str, dataset_code: str | None = None) -> pd.DataFrame:
    if dataset_code:
        engine = get_dataset_engine(dataset_code)
    else:
        engine = get_forecastbot_engine()

    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)