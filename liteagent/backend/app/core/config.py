from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "LiteAgent"
    debug: bool = True
    database_url: str = "sqlite+aiosqlite:///./liteagent.db"

    # Default provider API keys (users can add their own)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
