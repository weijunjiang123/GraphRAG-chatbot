import os

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# 持久化存储目录和索引文件名称
PERSIST_DIR = "src/RAG/chroma_db"
COLLECTION_NAME = "quickstart"
# 配置参数
PERSIST_PATH = "../chroma_db"  # Chroma 持久化存储路径
EMBED_MODEL_NAME = "nomic-embed-text"  # embedding 模型名称
LLM_MODEL = "deepseek-r1"  # 查询时使用的 LLM 模型

# 确保存储目录存在
if not os.path.exists(PERSIST_DIR):
    os.makedirs(PERSIST_DIR)

# 定义 embedding 模型实例
embed_model = OllamaEmbedding(model_name=EMBED_MODEL_NAME)
llm = Ollama(model=LLM_MODEL, request_timeout=120.0)


def query_index_service(question: str) -> str:
    """
    加载持久化索引，并使用 RAG 功能回答用户问题
    """
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 从持久化存储加载索引
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    # 创建查询引擎，使用 Ollama 作为 LLM
    query_engine = index.as_query_engine(
        llm=Ollama(model=LLM_MODEL, request_timeout=360.0)
    )
    response = query_engine.query(question)
    return str(response)


def agent_RAG_query(query: str):
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # 从持久化存储加载索引
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    # 创建查询引擎，使用 Ollama 作为 LLM
    query_engine = index.as_query_engine(
        llm=Ollama(model=LLM_MODEL, request_timeout=360.0)
    )
    # 定义 RAG 工具，将 rag_query 函数包装为 FunctionTool
    rag_tool = QueryEngineTool.from_defaults(
        query_engine,
        name="RAGQueryTool",
        description="用于基于上传文档回答问题的工具。"
    )

    # 创建一个包含 RAG 工具的 ReActAgent
    agent = ReActAgent.from_tools(
        [rag_tool],
        llm=llm,
        verbose=True
    )
    response = agent.chat(query)
    return response


print(agent_RAG_query("作者在文中讲了什么内容"))
