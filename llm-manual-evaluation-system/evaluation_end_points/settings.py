from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    openrouter_api_key: str = ""


@lru_cache
def get_settings():
    return Settings()
