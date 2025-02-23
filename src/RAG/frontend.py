# frontend.py
import streamlit as st
import requests
from typing import Optional
import time

# 后端 API 地址，根据实际情况调整
BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG 文档问答系统", layout="wide")
st.title("RAG 文档问答系统")
st.markdown("一个基于 FastAPI 后端和 llama_index 构建的文档问答系统。上传文档后即可基于上传内容进行问答。")

# 侧边栏操作选择
option = st.sidebar.radio("选择操作", ["上传文档构建索引", "查询问答"])

def make_request(url: str, method: str = "GET", **kwargs) -> Optional[dict]:
    """统一的请求处理函数"""
    try:
        if method == "GET":
            response = requests.get(url, **kwargs)
        elif method == "POST":
            response = requests.post(url, **kwargs)
        else:
            raise ValueError(f"不支持的请求方法: {method}")
            
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("无法连接到后端服务，请确保后端服务已启动")
        return None
    except requests.exceptions.HTTPError as e:
        error_msg = "未知错误"
        try:
            error_msg = e.response.json().get("detail", str(e))
        except:
            pass
        st.error(f"请求失败: {error_msg}")
        return None
    except Exception as e:
        st.error(f"发生错误: {str(e)}")
        return None

# 在侧边栏添加状态检查
with st.sidebar:
    st.markdown("---")
    st.markdown("### 系统状态")
    if st.button("检查服务状态", key="check_status"):
        result = make_request(f"{BACKEND_URL}/status")
        if result and "status" in result:
            status = result["status"]
            if status["ollama"]:
                st.success("✅ Ollama 服务正常")
            else:
                st.error("❌ Ollama 服务未启动")
            if status["chroma"]:
                st.success("✅ Chroma 数据库正常")
            else:
                st.error("❌ Chroma 数据库异常")

def handle_query(question: str):
    """处理查询请求"""
    # 先检查服务状态
    status_result = make_request(f"{BACKEND_URL}/status")
    if status_result and "status" in status_result:
        if not status_result["status"]["ollama"]:
            st.error("❌ Ollama 服务未启动，请先启动服务")
            st.info("提示：请在终端运行 'ollama serve' 启动服务")
            return

    with st.spinner("正在思考，请稍候..."):
        result = make_request(
            f"{BACKEND_URL}/query",
            params={"question": question},
            timeout=360  # 6分钟超时
        )
        if result:
            st.markdown("### 回答")
            st.write(result.get("answer", "未能获取到回答"))

            # 添加反馈按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 回答有帮助"):
                    st.success("感谢您的反馈！")
            with col2:
                if st.button("👎 回答需要改进"):
                    st.info("感谢您的反馈，我们会继续改进！")

if option == "上传文档构建索引":
    st.header("上传文档构建索引")
    st.markdown("""
    ### 使用说明
    1. 请上传 UTF-8 编码的文本文件
    2. 文件大小不要超过 10MB
    3. 支持增量更新索引
    """)
    
    uploaded_file = st.file_uploader("选择一个文本文件 (txt)", type=["txt"])
    
    if uploaded_file is not None:
        file_details = {
            "文件名": uploaded_file.name,
            "文件大小": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.write(file_details)
        
        if st.button("提交上传", key="upload_button"):
            with st.spinner("正在构建索引，请稍候..."):
                files = {"file": uploaded_file}
                result = make_request(
                    f"{BACKEND_URL}/build_index",
                    method="POST",
                    files=files,
                    timeout=300  # 5分钟超时
                )
                if result:
                    st.success(result.get("message", "索引构建成功！"))

elif option == "查询问答":
    st.header("查询问答")
    st.markdown("""
    ### 使用说明
    1. 输入您的问题
    2. 系统将基于已构建的索引进行回答
    3. 如果没有相关内容，系统会提示您重新提问
    """)
    
    question = st.text_input("请输入你的问题：")
    
    if st.button("查询", key="query_button"):
        if not question.strip():
            st.warning("请输入有效的问题")
        else:
            handle_query(question)
