"""Enhanced API routes with provider configuration"""
from enum import Enum
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, status, Query, Body, Depends
from fastapi.responses import JSONResponse

from src.RAG.core.config import settings
from src.RAG.api.models import ErrorResponse, IndexResponse, QueryResponse, StatusResponse, ProviderResponse
from src.RAG.services.rag_service import RAGService

router = APIRouter()

# Create a global RAG service instance
rag_service = RAGService()


class Provider(str, Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    VOLCES = "volces"
    CUSTOM = "custom"


@router.get("/status", response_model=StatusResponse)
def check_status():
    """Check the status of all services"""
    try:
        status_result = rag_service.get_services_status()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": status_result}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during status check: {str(e)}"
        )


@router.get("/query", response_model=QueryResponse)
def query_index(
    question: str,
    top_k: int = Query(3, description="Number of top results to consider", ge=1, le=10),
    provider: Optional[Provider] = Query(None, description="LLM provider to use")
):
    """Query the RAG system with a question
    
    Args:
        question: The question to answer
        top_k: Number of top results to consider
        provider: Optional LLM provider to use
        
    Returns:
        Answer to the question
    """
    try:
        answer = rag_service.query(
            question=question, 
            similarity_top_k=top_k,
            llm_provider=provider
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"answer": answer, "question": question}
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing error: {str(e)}"
        )


@router.post(
    "/build_index",
    response_model=IndexResponse,
    responses={
        200: {"model": IndexResponse, "description": "Index built successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"}
    }
)
async def build_index(file: UploadFile = File(...)):
    """Build or update the index with the uploaded file"""
    try:
        result = await rag_service.build_index(file)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result
        )
    except ValueError as ve:
        if "provider is not available" in str(ve):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(ve)
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error building index: {str(e)}"
        )


@router.delete(
    "/clear_index",
    responses={
        200: {"description": "Index cleared successfully"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
def clear_index():
    """Clear the entire index"""
    try:
        result = rag_service.clear_index()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing index: {str(e)}"
        )


@router.post(
    "/provider",
    response_model=ProviderResponse,
    responses={
        200: {"model": ProviderResponse, "description": "Provider changed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
def change_provider(
    llm_provider: Provider = Body(..., embed=True),
    embed_provider: Optional[Provider] = Body(None, embed=True)
):
    """Change the LLM and embedding providers
    
    Args:
        llm_provider: New LLM provider
        embed_provider: Optional new embedding provider (default: same as llm_provider)
        
    Returns:
        Status message
    """
    try:
        result = rag_service.change_provider(
            llm_provider=llm_provider,
            embed_provider=embed_provider
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error changing provider: {str(e)}"
        )


@router.get(
    "/providers",
    responses={
        200: {"description": "List of available providers"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
def list_providers():
    """List all available providers and their status"""
    try:
        providers = ["ollama", "openai", "volces", "custom"]
        provider_status = {}
        
        for provider in providers:
            provider_status[provider] = {
                "available": rag_service.check_provider_available(provider),
                "current": provider == rag_service.llm_provider,
                "models": _get_provider_models(provider)
            }
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"providers": provider_status}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing providers: {str(e)}"
        )


def _get_provider_models(provider: str):
    """Get available models for a provider from settings
    
    Args:
        provider: Provider name
        
    Returns:
        Dictionary with LLM and embedding model names
    """
    if provider == "ollama":
        return {
            "llm": settings.llm.LLM_MODEL,
            "embedding": settings.llm.EMBED_MODEL_NAME
        }
    elif provider == "openai":
        return {
            "llm": settings.llm.OPENAI_LLM_MODEL,
            "embedding": settings.llm.OPENAI_EMBED_MODEL
        }
    elif provider == "volces":
        return {
            "llm": settings.llm.VOLCES_LLM_MODEL,
            "embedding": settings.llm.VOLCES_EMBED_MODEL
        }
    elif provider == "custom":
        return {
            "llm": settings.llm.CUSTOM_LLM_MODEL,
            "embedding": settings.llm.CUSTOM_EMBED_MODEL
        }
    return {}


@router.get(
    "/health",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy"}
    }
)
def health_check():
    """Simple health check endpoint"""
    services = rag_service.get_services_status()
    
    # Check if the current provider is available
    current_provider = services.get("current_provider", {})
    is_healthy = current_provider.get("status", False) and services.get("chroma", False)
    
    if is_healthy:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "healthy", "services": services}
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "services": services}
        )