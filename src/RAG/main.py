import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.RAG.api.routes import router

app = FastAPI(title="RAG Agent 后端服务")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router)


@app.get("/")
def read_root():
    return {"message": "RAG Agent 后端服务已启动"}


if __name__ == "__main__":
    uvicorn.run(
        "src.RAG.main:app",  # 使用模块路径:app变量名的格式
        host="0.0.0.0",
        port=8000,
        reload=True  # 开发模式下启用热重载
    )

