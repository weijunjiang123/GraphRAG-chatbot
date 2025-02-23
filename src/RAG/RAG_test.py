# services.py
import os
import chromadb
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import logging
import sys
from typing import Optional
import requests
from functools import wraps
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 配置参数
PERSIST_PATH = "../chroma_db"  # Chroma 持久化存储路径
COLLECTION_NAME = "quickstart"  # Chroma collection 名称
EMBED_MODEL_NAME = "nomic-embed-text"  # embedding 模型名称
LLM_MODEL = "llama3.2"  # 查询时使用的 LLM 模型

# 定义 embedding 模型实例
embed_model = OllamaEmbedding(model_name=EMBED_MODEL_NAME)

def check_ollama_service(max_retries=3, retry_delay=1):
    """
    检查 Ollama 服务是否可用的装饰器
    
    Args:
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    # 检查 Ollama 服务是否在运行
                    response = requests.get("http://localhost:11434/api/tags")
                    if response.status_code == 200:
                        return func(*args, **kwargs)
                    else:
                        logger.error(f"Ollama 服务响应异常: {response.status_code}")
                except requests.exceptions.ConnectionError:
                    if attempt < max_retries - 1:
                        logger.warning(f"Ollama 服务未启动，{retry_delay}秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    logger.error("无法连接到 Ollama 服务，请确保服务已启动")
                    raise ValueError("Ollama 服务未启动，请先启动 Ollama 服务")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@check_ollama_service()
def build_index_from_file_service(file_content: str) -> str:
    """
    接收上传的文档内容，构建或更新索引。
    如果索引已存在，则将文档插入到现有索引中；否则，新建索引。
    
    Args:
        file_content: 文档内容字符串
    Returns:
        str: 处理结果消息
    Raises:
        Exception: 索引构建过程中的错误
    """
    if not file_content.strip():
        raise ValueError("文档内容不能为空")
        
    logger.info("开始处理文档并构建索引...")
    
    # 将文件内容构造为 Document 对象
    
    doc = Document(text=file_content)

    # 初始化持久化的 Chroma 客户端和 collection
    db = chromadb.PersistentClient(path=PERSIST_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    try:
        # 尝试加载已有索引
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
        # 插入新文档（假设支持增量更新）
        index.insert(doc)
        
        logger.info("索引构建/更新成功")
        return "文档上传成功，索引已更新！"
    except Exception as e:
        logger.error(f"索引构建失败: {str(e)}")
        raise

@check_ollama_service()
def query_index_service(question: str) -> Optional[str]:
    """
    加载持久化索引，并使用 RAG 功能回答用户问题
    
    Args:
        question: 用户提问
    Returns:
        str: 回答内容
    Raises:
        ValueError: 问题为空时抛出
        Exception: 查询过程中的其他错误
    """
    if not question.strip():
        raise ValueError("问题不能为空")
        
    logger.info(f"开始处理问题: {question}")
    
    try:
        db = chromadb.PersistentClient(path=PERSIST_PATH)
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # 从持久化存储加载索引
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

        try:
            # 创建查询引擎，使用 Ollama 作为 LLM
            query_engine = index.as_query_engine(
                llm=Ollama(model=LLM_MODEL, request_timeout=360.0)
            )
            response = query_engine.query(question)
            
            logger.info("查询完成")
            return str(response)
        except Exception as e:
            if "status code: 502" in str(e):
                raise ValueError("Ollama 服务暂时不可用，请稍后重试")
            raise
    except Exception as e:
        logger.error(f"查询失败: {str(e)}")
        raise

# 添加一个新的函数用于检查服务状态
def check_services_status() -> dict:
    """
    检查所有必要服务的状态
    
    Returns:
        dict: 包含各服务状态的字典
    """
    status = {
        "ollama": False,
        "chroma": False
    }
    
    # 检查 Ollama 服务
    try:
        response = requests.get("http://localhost:11434/api/tags")
        status["ollama"] = response.status_code == 200
    except:
        pass
    
    # 检查 Chroma 数据库
    try:
        db = chromadb.PersistentClient(path=PERSIST_PATH)
        db.get_or_create_collection(COLLECTION_NAME)
        status["chroma"] = True
    except:
        pass
    
    return status