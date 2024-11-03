import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    # Extra settings for .env.stt file
    OPENAI_API_KEY: str = Field(..., env='OPENAI_API_KEY')
    LANGCHAIN_TRACING_V2: bool = Field(..., env='LANGCHAIN_TRACING_V2')
    LANGCHAIN_ENDPOINT: str = Field(..., env='LANGCHAIN_ENDPOINT')
    LANGCHAIN_PROJECT: str = Field(..., env='LANGCHAIN_PROJECT')
    LANGCHAIN_API_KEY: str = Field(..., env='LANGCHAIN_API_KEY')

    model_config = SettingsConfigDict(env_file=os.getenv('ENV_FILE_STT', None))


settings = Settings()