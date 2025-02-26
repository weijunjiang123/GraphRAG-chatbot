"""Enhanced Pydantic models with provider support"""
from typing import Dict, Any, List, Optional, Union

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str


class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    question: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The answer to your question is...",
                "question": "What is RAG?"
            }
        }


class IndexResponse(BaseModel):
    """Index building response model"""
    message: str
    filename: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Document uploaded successfully! Added 42 chunks to the index.",
                "filename": "example.pdf"
            }
        }


class ProviderResponse(BaseModel):
    """Provider change response model"""
    message: str
    llm_provider: str
    embed_provider: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Provider changed successfully to openai",
                "llm_provider": "openai",
                "embed_provider": "openai"
            }
        }


class ProviderInfo(BaseModel):
    """Provider information model"""
    available: bool
    current: bool
    models: Dict[str, str]


class ProviderListResponse(BaseModel):
    """Provider list response model"""
    providers: Dict[str, ProviderInfo]
    
    class Config:
        json_schema_extra = {
            "example": {
                "providers": {
                    "ollama": {
                        "available": True,
                        "current": True,
                        "models": {
                            "llm": "deepscaler",
                            "embedding": "nomic-embed-text"
                        }
                    },
                    "openai": {
                        "available": False,
                        "current": False,
                        "models": {
                            "llm": "gpt-3.5-turbo",
                            "embedding": "text-embedding-3-small"
                        }
                    }
                }
            }
        }


class CurrentProviderStatus(BaseModel):
    """Current provider status model"""
    name: str
    status: bool


class StatusResponse(BaseModel):
    """Status response model"""
    status: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": {
                    "ollama": True,
                    "current_provider": {
                        "name": "ollama",
                        "status": True
                    },
                    "chroma": True,
                    "indexed_documents": {
                        "document_count": 42,
                        "collection_name": "quickstart"
                    }
                }
            }
        }


class DocumentInfo(BaseModel):
    """Information about an indexed document"""
    filename: str
    chunks: int
    timestamp: str


class IndexInfoResponse(BaseModel):
    """Response with information about the index"""
    document_count: int
    documents: List[DocumentInfo]
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_count": 2,
                "documents": [
                    {
                        "filename": "example1.pdf",
                        "chunks": 25,
                        "timestamp": "2025-02-26T10:15:30Z"
                    },
                    {
                        "filename": "example2.txt",
                        "chunks": 17,
                        "timestamp": "2025-02-25T14:22:12Z"
                    }
                ]
            }
        }


class ProviderConfigRequest(BaseModel):
    """Request model for provider configuration"""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    llm_model: Optional[str] = None
    embed_model: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None


class ProviderConfigResponse(BaseModel):
    """Response model for provider configuration"""
    message: str
    provider: str
    config: Dict[str, Any]