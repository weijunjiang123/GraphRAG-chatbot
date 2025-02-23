"""RAG 服务实现"""
import logging
from typing import Optional

import chromadb
import requests
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.chat_engine.types import ChatMode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.RAG.core.config import settings
from src.RAG.services.base import BaseService, retry_on_failure, logger

logger = logging.getLogger(__name__)


class RAGService(BaseService):
    def __init__(self):
        super().__init__()
        logger.info("初始化 RAG 服务...")
        self.embed_model = OllamaEmbedding(model_name=settings.EMBED_MODEL_NAME)

    def check_ollama_available(self) -> bool:
        """检查 Ollama 服务是否可用"""
        try:
            response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            return response.status_code == 200
        except:
            return False

    def check_chroma_available(self) -> bool:
        """检查 Chroma 是否可用"""
        try:
            db = chromadb.PersistentClient(path=settings.PERSIST_PATH)
            db.get_or_create_collection(settings.COLLECTION_NAME)
            return True
        except:
            return False

    def get_services_status(self) -> dict:
        """获取所有服务状态"""
        return {
            "ollama": self.check_ollama_available(),
            "chroma": self.check_chroma_available()
        }

    @retry_on_failure(max_retries=3, delay=1)
    def build_index(self, content: str) -> str:
        """构建索引"""
        if not content.strip():
            raise ValueError("文档内容不能为空")

        if not self.check_ollama_available():
            raise ValueError("Ollama 服务未启动，请先启动服务")

        logger.info("开始处理文档并构建索引...")

        doc = Document(text=content)
        db = chromadb.PersistentClient(path=settings.PERSIST_PATH)
        chroma_collection = db.get_or_create_collection(settings.COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        try:
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=self.embed_model
            )
            index.insert(doc)
            logger.info("索引构建/更新成功")
            return "文档上传成功，索引已更新！"
        except Exception as e:
            logger.error(f"索引构建失败: {str(e)}")
            raise

    @retry_on_failure(max_retries=3, delay=1)
    def query(self, question: str) -> Optional[str]:
        """查询"""
        if not question.strip():
            raise ValueError("问题不能为空")

        if not self.check_ollama_available():
            raise ValueError("Ollama 服务未启动，请先启动服务")

        logger.info(f"开始处理问题: {question}")

        try:
            db = chromadb.PersistentClient(path=settings.PERSIST_PATH)
            chroma_collection = db.get_or_create_collection(settings.COLLECTION_NAME)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            # data = SimpleDirectoryReader(input_dir="../../../data/paul_graham").load_data()
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=self.embed_model
            )
            chat_engine = index.as_query_engine(
                chat_mode=ChatMode.CONTEXT,
                llm=Ollama(base_url=settings.OLLAMA_BASE_URL, model=settings.LLM_MODEL),
            )
            resp = chat_engine.chat(question)
            print(resp)
            logger.info("查询完成")
            return str(resp)
        except Exception as e:
            logger.error(f"查询失败: {str(e)}")
            raise

# rag_service = RAGService()
# response = rag_service.query("who is paul graham?")
# print(response)
