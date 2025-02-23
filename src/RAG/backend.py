"""FastAPI 后端服务"""
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .services.rag_service import RAGService
from .core.config import settings

# 初始化服务
rag_service = RAGService()

app = FastAPI(title="RAG Agent 后端服务")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ErrorResponse(BaseModel):
    detail: str

@app.post("/build_index", 
    responses={
        200: {"model": dict, "description": "索引构建成功"},
        400: {"model": ErrorResponse, "description": "无效的请求"},
        500: {"model": ErrorResponse, "description": "服务器内部错误"},
        503: {"model": ErrorResponse, "description": "服务暂时不可用"}
    }
)
async def build_index(file: UploadFile = File(...)):
    """构建索引接口"""
    if not file.filename.endswith('.txt'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="只支持 .txt 文件"
        )
    
    try:
        content = await file.read()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="文件编码必须是 UTF-8"
            )
            
        message = rag_service.build_index(text)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": message, "filename": file.filename}
        )
    except ValueError as ve:
        if "Ollama 服务" in str(ve):
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
            detail="索引构建过程中发生错误"
        )

@app.get("/query",
    responses={
        200: {"model": dict, "description": "查询成功"},
        400: {"model": ErrorResponse, "description": "无效的查询"},
        500: {"model": ErrorResponse, "description": "服务器内部错误"},
        503: {"model": ErrorResponse, "description": "服务暂时不可用"}
    }
)
def query_index(question: str):
    """查询接口"""
    try:
        answer = rag_service.query(question)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"answer": answer, "question": question}
        )
    except ValueError as ve:
        if "Ollama 服务" in str(ve):
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
            detail="查询过程中发生错误"
        )

@app.get("/status")
def check_status():
    """状态检查接口"""
    try:
        status_result = rag_service.get_services_status()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": status_result}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="状态检查过程中发生错误"
        )

@app.get("/")
def read_root():
    return {"message": "RAG Agent 后端服务已启动"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.HOST, 
        port=settings.PORT
    )