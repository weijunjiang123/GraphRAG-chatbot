"""LLM adapter module providing compatibility with various LLM providers"""
import logging
import os
from enum import Enum
from typing import Optional, Dict, List, Union, Any, Callable

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    VOLCES = "volces"  # Volces API (OpenAI compatible)
    CUSTOM = "custom"  # For any OpenAI compatible API


class LLMFactory:
    """Factory for creating LLM and embedding model instances based on provider"""
    
    @staticmethod
    def create_llm(
        provider: str,
        model_name: str,
        **kwargs
    ) -> Any:
        """Create an LLM instance based on the specified provider
        
        Args:
            provider: The LLM provider (ollama, openai, volces, custom)
            model_name: Name of the model to use
            **kwargs: Additional provider-specific parameters
            
        Returns:
            An LLM instance compatible with llama_index
        """
        provider = provider.lower()
        
        if provider == LLMProvider.OLLAMA:
            # Configure Ollama
            base_url = kwargs.get("base_url", "http://localhost:11434")
            request_timeout = kwargs.get("request_timeout", 120)
            
            return Ollama(
                model=model_name,
                base_url=base_url,
                request_timeout=request_timeout
            )
            
        elif provider == LLMProvider.OPENAI:
            # Configure OpenAI
            api_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key must be provided either in kwargs or as OPENAI_API_KEY environment variable")
            
            temperature = kwargs.get("temperature", 0.7)
            
            return LlamaIndexOpenAI(
                model=model_name,
                api_key=api_key,
                temperature=temperature
            )
            
        elif provider == LLMProvider.VOLCES:
            # Configure Volces (OpenAI compatible)
            api_key = kwargs.get("api_key") or os.environ.get("ARK_API_KEY")
            if not api_key:
                raise ValueError("Volces API key must be provided either in kwargs or as ARK_API_KEY environment variable")
            
            base_url = kwargs.get("base_url", "https://ark.cn-beijing.volces.com/api/v3")
            temperature = kwargs.get("temperature", 0.7)
            
            return LlamaIndexOpenAI(
                model=model_name,
                api_key=api_key,
                api_base=base_url,
                temperature=temperature
            )
            
        elif provider == LLMProvider.CUSTOM:
            # Configure custom OpenAI-compatible API
            api_key = kwargs.get("api_key") or os.environ.get("CUSTOM_API_KEY")
            if not api_key:
                raise ValueError("API key must be provided either in kwargs or as CUSTOM_API_KEY environment variable")
            
            base_url = kwargs.get("base_url")
            if not base_url:
                raise ValueError("base_url must be provided for custom provider")
                
            temperature = kwargs.get("temperature", 0.7)
            
            return LlamaIndexOpenAI(
                model=model_name,
                api_key=api_key,
                api_base=base_url,
                temperature=temperature
            )
            
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def create_embedding_model(
        provider: str,
        model_name: str,
        **kwargs
    ) -> Any:
        """Create an embedding model instance based on the specified provider
        
        Args:
            provider: The embedding model provider (ollama, openai, volces, custom)
            model_name: Name of the model to use
            **kwargs: Additional provider-specific parameters
            
        Returns:
            An embedding model instance compatible with llama_index
        """
        provider = provider.lower()
        
        if provider == LLMProvider.OLLAMA:
            # Configure Ollama embedding
            base_url = kwargs.get("base_url", "http://localhost:11434")
            
            return OllamaEmbedding(
                model_name=model_name,
                base_url=base_url
            )
            
        elif provider == LLMProvider.OPENAI:
            # Configure OpenAI embedding
            api_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key must be provided either in kwargs or as OPENAI_API_KEY environment variable")
            
            return OpenAIEmbedding(
                model=model_name,
                api_key=api_key
            )
            
        elif provider == LLMProvider.VOLCES:
            # Configure Volces (OpenAI compatible) embedding
            api_key = kwargs.get("api_key") or os.environ.get("ARK_API_KEY")
            if not api_key:
                raise ValueError("Volces API key must be provided either in kwargs or as ARK_API_KEY environment variable")
            
            base_url = kwargs.get("base_url", "https://ark.cn-beijing.volces.com/api/v3")
            
            return OpenAIEmbedding(
                model=model_name,
                api_key=api_key,
                api_base=base_url
            )
            
        elif provider == LLMProvider.CUSTOM:
            # Configure custom OpenAI-compatible API embedding
            api_key = kwargs.get("api_key") or os.environ.get("CUSTOM_API_KEY")
            if not api_key:
                raise ValueError("API key must be provided either in kwargs or as CUSTOM_API_KEY environment variable")
            
            base_url = kwargs.get("base_url")
            if not base_url:
                raise ValueError("base_url must be provided for custom provider")
                
            return OpenAIEmbedding(
                model=model_name,
                api_key=api_key,
                api_base=base_url
            )
            
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")


# Direct client for OpenAI API compatibility layer
class OpenAIClient:
    """Direct client for OpenAI-compatible APIs"""
    
    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """Initialize OpenAI compatible client
        
        Args:
            provider: Provider name (openai, volces, custom)
            api_key: API key for the provider
            base_url: Base URL for the API
            **kwargs: Additional parameters for the client
        """
        self.provider = provider.lower()
        
        # Import here to make it optional
        try:
            from openai import OpenAI
            self._client_cls = OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package is required for OpenAIClient. "
                "Please install it with 'pip install openai>=1.0'"
            )
        
        # Configure API key based on provider
        if self.provider == LLMProvider.OPENAI:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            self.base_url = base_url
        elif self.provider == LLMProvider.VOLCES:
            self.api_key = api_key or os.environ.get("ARK_API_KEY")
            self.base_url = base_url or "https://ark.cn-beijing.volces.com/api/v3"
        elif self.provider == LLMProvider.CUSTOM:
            self.api_key = api_key or os.environ.get("CUSTOM_API_KEY")
            self.base_url = base_url
        else:
            raise ValueError(f"Unsupported provider for OpenAI client: {provider}")
            
        if not self.api_key:
            raise ValueError(f"API key must be provided for {provider}")
            
        # Initialize client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            
        # Add any additional kwargs
        client_kwargs.update(kwargs)
        
        self._client = self._client_cls(**client_kwargs)
        
    @property
    def client(self):
        """Get the underlying OpenAI client"""
        return self._client
        
    @property
    def chat(self):
        """Access the chat completions API"""
        return self._client.chat
        
    @property
    def embeddings(self):
        """Access the embeddings API"""
        return self._client.embeddings
        
    def get_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        """Get embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            model: Embedding model to use
            
        Returns:
            List of embedding vectors
        """
        response = self.embeddings.create(
            model=model,
            input=texts
        )
        
        # Extract the embeddings from the response
        embeddings = [item.embedding for item in response.data]
        return embeddings