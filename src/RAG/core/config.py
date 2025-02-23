"""配置文件"""
import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Chroma 配置
    PERSIST_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../chroma_db")
    COLLECTION_NAME: str = "quickstart"

    # Ollama 配置
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBED_MODEL_NAME: str = "nomic-embed-text"
    LLM_MODEL: str = "deepscaler"

    # 其他配置
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 2
    REQUEST_TIMEOUT: int = 30


settings = Settings()
