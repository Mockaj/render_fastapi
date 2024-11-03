import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
load_dotenv()
class Settings(BaseSettings):
    # Langtail settings
    LANGTAIL_API_KEY: str = Field(..., env='LANGTAIL_API_KEY')
    LANGTAIL_WORKSPACE: str = Field(..., env='LANGTAIL_WORKSPACE')
    LANGTAIL_PROJECT: str = Field(..., env='LANGTAIL_PROJECT')
    LANGTAIL_PROMPT: str = Field(..., env='LANGTAIL_PROMPT')
    LANGTAIL_ENVIRONMENT: str = Field(..., env='LANGTAIL_ENVIRONMENT')

    # Qdrant settings
    QDRANT_HOST: str = Field(default='localhost', env='QDRANT_HOST')
    QDRANT_PORT: int = Field(default=6333, env='QDRANT_PORT')
    QDRANT_API_KEY: str = Field(default='', env='QDRANT_API_KEY')
    QDRANT_COLLECTION_NAME: str = Field(..., env='QDRANT_COLLECTION_NAME')

    # VoyageAI settings
    VOYAGE_API_KEY: str = Field(..., env='VOYAGE_API_KEY')
    VOYAGE_MODEL: str = Field(..., env='VOYAGE_MODEL')
    # OpenData settings
    OPEN_DATA_API_KEY: str = Field(..., env='OPEN_DATA_API_KEY')

    # General settings
    DEFAULT_N: int = Field(default=3)

    # Auth
    AUTH_USERNAME1: str = Field(..., env='AUTH_USERNAME1')
    AUTH_PASSWORD1: str = Field(..., env='AUTH_PASSWORD1')

    model_config = SettingsConfigDict(env_file=os.getenv('ENV_FILE_RAG', None))

settings = Settings()