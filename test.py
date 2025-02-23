from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.ollama import Ollama
from ollama._types import ResponseError

# you can also set DEEPSEEK_API_KEY in your environment variables
# llm = DeepSeek(model="deepseek-chat", api_key="sk-4a451c597e0c44a3bf1708f3bfd0c933", stream=True)
llm = Ollama(base_url="http://localhost:11434",model="llama3.2", request_timeout=120.0)
# You might also want to set deepseek as your default llm
# from llama_index.core import Settings
# Settings.llm = llm

try:
    response = llm.complete("GraphRAG对比传统的RAG有什么优势")
    print(response)
except ResponseError as e:
    print(f"ResponseError: {str(e)} (status code: {e.status_code})")
except Exception as e:
    print(f"An error occurred: {str(e)}")