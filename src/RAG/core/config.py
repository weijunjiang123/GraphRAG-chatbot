"""Enhanced configuration settings with support for multiple LLM providers"""
import os
from typing import List, Dict, Any, Optional

from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    """Configuration for LLM models"""
    
    # Default provider (ollama, openai, volces, custom)
    PROVIDER: str = "ollama"
    
    # Model names
    LLM_MODEL: str = "deepscaler"
    EMBED_MODEL_NAME: str = "nomic-embed-text"
    
    # Ollama specific settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # OpenAI specific settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_LLM_MODEL: str = "gpt-3.5-turbo"
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"
    
    # Volces specific settings
    ARK_API_KEY: Optional[str] = "c4c0e0e9-0049-4c35-8c41-a126ffcfa9d0"
    VOLCES_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/v3"
    VOLCES_LLM_MODEL: str = "deepseek-r1-250120"
    VOLCES_EMBED_MODEL: str = "doubao-embedding-text-240715"
    
    # Custom API settings
    CUSTOM_API_KEY: Optional[str] = None
    CUSTOM_BASE_URL: Optional[str] = None
    CUSTOM_LLM_MODEL: str = "llm-model"
    CUSTOM_EMBED_MODEL: str = "embed-model"
    
    # Generic LLM settings
    TEMPERATURE: float = 0.7
    MAX_TOKENS: Optional[int] = None
    TOP_P: float = 1.0
    TIMEOUT: int = 120  # seconds


class ChromaSettings(BaseSettings):
    """Configuration for Chroma vector database"""
    
    PERSIST_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../chroma_db")
    COLLECTION_NAME: str = "quickstart"
    CREATE_COLLECTION_IF_NOT_EXISTS: bool = True


class DocumentProcessingSettings(BaseSettings):
    """Configuration for document processing"""
    
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MAX_DOCUMENT_SIZE_MB: int = 20
    SUPPORTED_EXTENSIONS: List[str] = [".txt", ".pdf", ".docx", ".md", ".html", ".csv"]


class APISettings(BaseSettings):
    """Configuration for the API server"""
    
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # CORS configuration
    ALLOW_ORIGINS: List[str] = ["*"]
    ALLOW_CREDENTIALS: bool = True
    ALLOW_METHODS: List[str] = ["*"]
    ALLOW_HEADERS: List[str] = ["*"]
    
    # Request handling
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 2
    REQUEST_TIMEOUT: int = 60


class Settings(BaseSettings):
    """Main application settings"""
    
    # Individual setting categories
    llm: LLMSettings = LLMSettings()
    chroma: ChromaSettings = ChromaSettings()
    doc_processing: DocumentProcessingSettings = DocumentProcessingSettings()
    api: APISettings = APISettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    def get_llm_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for a specific LLM provider
        
        Args:
            provider: Provider name (default: use PROVIDER from settings)
            
        Returns:
            Dictionary with provider-specific configuration
        """
        provider = provider or self.llm.PROVIDER
        provider = provider.lower()
        
        if provider == "ollama":
            return {
                "model_name": self.llm.LLM_MODEL,
                "base_url": self.llm.OLLAMA_BASE_URL,
                "request_timeout": self.llm.TIMEOUT
            }
        elif provider == "openai":
            return {
                "model_name": self.llm.OPENAI_LLM_MODEL,
                "api_key": self.llm.OPENAI_API_KEY,
                "temperature": self.llm.TEMPERATURE,
                "max_tokens": self.llm.MAX_TOKENS,
                "top_p": self.llm.TOP_P
            }
        elif provider == "volces":
            return {
                "model_name": self.llm.VOLCES_LLM_MODEL,
                "api_key": self.llm.ARK_API_KEY,
                "base_url": self.llm.VOLCES_BASE_URL,
                "temperature": self.llm.TEMPERATURE,
                "max_tokens": self.llm.MAX_TOKENS,
                "top_p": self.llm.TOP_P
            }
        elif provider == "custom":
            return {
                "model_name": self.llm.CUSTOM_LLM_MODEL,
                "api_key": self.llm.CUSTOM_API_KEY,
                "base_url": self.llm.CUSTOM_BASE_URL,
                "temperature": self.llm.TEMPERATURE,
                "max_tokens": self.llm.MAX_TOKENS,
                "top_p": self.llm.TOP_P
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def get_embedding_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for a specific embedding provider
        
        Args:
            provider: Provider name (default: use PROVIDER from settings)
            
        Returns:
            Dictionary with provider-specific configuration
        """
        provider = provider or self.llm.PROVIDER
        provider = provider.lower()
        
        if provider == "ollama":
            return {
                "model_name": self.llm.EMBED_MODEL_NAME,
                "base_url": self.llm.OLLAMA_BASE_URL
            }
        elif provider == "openai":
            return {
                "model_name": self.llm.OPENAI_EMBED_MODEL,
                "api_key": self.llm.OPENAI_API_KEY
            }
        elif provider == "volces":
            return {
                "model_name": self.llm.VOLCES_EMBED_MODEL,
                "api_key": self.llm.ARK_API_KEY,
                "base_url": self.llm.VOLCES_BASE_URL
            }
        elif provider == "custom":
            return {
                "model_name": self.llm.CUSTOM_EMBED_MODEL,
                "api_key": self.llm.CUSTOM_API_KEY,
                "base_url": self.llm.CUSTOM_BASE_URL
            }
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")


# Create settings instance
settings = Settings()

# Create persist directory if it doesn't exist
os.makedirs(settings.chroma.PERSIST_PATH, exist_ok=True)