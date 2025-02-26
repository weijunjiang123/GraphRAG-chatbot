"""Enhanced main application entry point with middleware and error handling"""
import logging
import os
from contextlib import asynccontextmanager
import sys
from pathlib import Path

# 获取当前脚本的绝对路径（main.py）
current_file = Path(__file__).resolve()

# 计算项目根目录路径（假设项目根目录是 GraphRAG-chatbot）
ROOT_DIR = current_file.parent.parent.parent  # 根据实际层级调整

# 将项目根目录添加到系统路径
sys.path.append(str(ROOT_DIR))
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.RAG.api.routes import router as rag_router
from src.RAG.core.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.api.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "rag_service.log"))
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: log application start
    logger.info("Starting RAG Service")
    yield
    # Shutdown: log application stop
    logger.info("Shutting down RAG Service")


app = FastAPI(
    title="GraphRAG API",
    description="RAG service for document question answering",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.ALLOW_ORIGINS,
    allow_credentials=settings.api.ALLOW_CREDENTIALS,
    allow_methods=settings.api.ALLOW_METHODS,
    allow_headers=settings.api.ALLOW_HEADERS,
)


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.debug(f"Response status: {response.status_code}")
    return response


# Error handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail)}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    errors = [{"loc": err["loc"], "msg": err["msg"]} for err in exc.errors()]
    logger.error(f"Validation error: {errors}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "errors": errors}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"}
    )


# Include routers
app.include_router(rag_router, prefix="/api")


# Root endpoint
@app.get("/")
async def root():
    return {
        "service": "GraphRAG",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    # Start Uvicorn server
    uvicorn.run(
        "src.RAG.main:app",
        host=settings.api.HOST,
        port=settings.api.PORT,
        reload=settings.api.DEBUG,
        log_level=settings.api.LOG_LEVEL.lower()
    )