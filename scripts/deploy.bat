@echo off
echo 开始部署 RAG Agent 服务...

REM 检查 Docker 是否安装
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo 错误: 未找到 Docker，请先安装 Docker
    exit /b 1
)

REM 构建并启动服务
docker-compose up --build -d

echo RAG Agent 服务已成功部署！ 