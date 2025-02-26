"""Enhanced Streamlit frontend with provider selection"""
import time
from typing import Optional, List, Dict, Any

import requests
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

# Backend API address
BACKEND_URL = "http://localhost:8000/api"

# Page configuration
st.set_page_config(
    page_title="GraphRAG - Document Q&A System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d1e7dd;
        color: #0f5132;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #f8d7da;
        color: #842029;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #cff4fc;
        color: #055160;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #664d03;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">GraphRAG - Document Q&A System</div>', unsafe_allow_html=True)
st.markdown("""
A powerful Retrieval-Augmented Generation system for document question answering.
Upload documents and ask questions to get insights from your data.
""")

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "current_provider" not in st.session_state:
    st.session_state.current_provider = None
if "available_providers" not in st.session_state:
    st.session_state.available_providers = {}
if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""
if "volces_key" not in st.session_state:
    st.session_state.volces_key = ""
if "custom_key" not in st.session_state:
    st.session_state.custom_key = ""
if "custom_url" not in st.session_state:
    st.session_state.custom_url = ""


def make_request(url: str, method: str = "GET", timeout: int = 30, **kwargs) -> Optional[Dict]:
    """Unified request handler with error handling"""
    spinner_text = kwargs.pop("spinner_text", "Processing request...")
    show_spinner = kwargs.pop("show_spinner", True)
    
    spinner_context = st.spinner(spinner_text) if show_spinner else DeltaGenerator()
    
    with spinner_context:
        try:
            if method == "GET":
                response = requests.get(url, timeout=timeout, **kwargs)
            elif method == "POST":
                response = requests.post(url, timeout=timeout, **kwargs)
            elif method == "DELETE":
                response = requests.delete(url, timeout=timeout, **kwargs)
            else:
                st.error(f"Unsupported request method: {method}")
                return None

            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the backend service. Please ensure the backend service is running.")
            return None
        except requests.exceptions.Timeout:
            st.error(f"Request timed out after {timeout} seconds. The operation may still be processing in the background.")
            return None
        except requests.exceptions.HTTPError as e:
            error_msg = "Unknown error"
            try:
                error_data = e.response.json()
                error_msg = error_data.get("detail", str(e))
            except:
                error_msg = f"HTTP Error: {str(e)}"
            
            st.error(f"Request failed: {error_msg}")
            return None
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None


def load_providers():
    """Load available providers from the backend"""
    result = make_request(
        f"{BACKEND_URL}/providers",
        spinner_text="Loading providers...",
        show_spinner=False
    )
    
    if result and "providers" in result:
        st.session_state.available_providers = result["providers"]
        
        # Set current provider
        for provider, info in result["providers"].items():
            if info.get("current", False):
                st.session_state.current_provider = provider
                break


def handle_query(question: str, provider: Optional[str] = None):
    """Process a query with optional provider override"""
    # Check if we should use a provider override
    params = {"question": question}
    if provider:
        params["provider"] = provider
    
    with st.spinner("Thinking..."):
        result = make_request(
            f"{BACKEND_URL}/query",
            params=params,
            timeout=360  # 6 minute timeout
        )
        
        if result:
            st.markdown("### Answer")
            st.write(result.get("answer", "No answer found"))
            
            # Add to chat history
            st.session_state.chat_history.append({
                "question": question,
                "answer": result.get("answer", "No answer found"),
                "provider": provider or st.session_state.current_provider
            })
            
            # Show feedback buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Helpful"):
                    st.success("Thank you for your feedback!")
            with col2:
                if st.button("👎 Not helpful"):
                    st.info("Thank you for your feedback. We'll work to improve our responses.")


# Provider configuration form
def provider_config_form(provider: str):
    """Display and handle provider configuration form"""
    st.markdown(f"### Configure {provider.upper()} Provider")
    
    if provider == "openai":
        api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_key,
            type="password",
            key="openai_key_input"
        )
        model = st.selectbox(
            "LLM Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            key="openai_model_select"
        )
        embed_model = st.selectbox(
            "Embedding Model",
            ["text-embedding-3-small", "text-embedding-3-large"],
            key="openai_embed_model_select"
        )
        
        if st.button("Save OpenAI Configuration"):
            st.session_state.openai_key = api_key
            config_data = {
                "api_key": api_key,
                "llm_model": model,
                "embed_model": embed_model
            }
            
            if api_key:
                change_provider("openai", config_data)
            else:
                st.error("API key is required")
                
    elif provider == "volces":
        api_key = st.text_input(
            "Volces API Key",
            value=st.session_state.volces_key,
            type="password",
            key="volces_key_input"
        )
        base_url = st.text_input(
            "Volces Base URL",
            value="https://ark.cn-beijing.volces.com/api/v3",
            key="volces_url_input"
        )
        model = st.selectbox(
            "LLM Model",
            ["deepseek-r1-250120", "llama3-70b", "qwen2-72b"],
            key="volces_model_select"
        )
        
        if st.button("Save Volces Configuration"):
            st.session_state.volces_key = api_key
            config_data = {
                "api_key": api_key,
                "base_url": base_url,
                "llm_model": model
            }
            
            if api_key:
                change_provider("volces", config_data)
            else:
                st.error("API key is required")
                
    elif provider == "custom":
        api_key = st.text_input(
            "API Key",
            value=st.session_state.custom_key,
            type="password",
            key="custom_key_input"
        )
        base_url = st.text_input(
            "Base URL",
            value=st.session_state.custom_url,
            key="custom_url_input"
        )
        model = st.text_input(
            "LLM Model Name",
            value="model",
            key="custom_model_input"
        )
        embed_model = st.text_input(
            "Embedding Model Name",
            value="embedding-model",
            key="custom_embed_model_input"
        )
        
        if st.button("Save Custom Configuration"):
            st.session_state.custom_key = api_key
            st.session_state.custom_url = base_url
            config_data = {
                "api_key": api_key,
                "base_url": base_url,
                "llm_model": model,
                "embed_model": embed_model
            }
            
            if api_key and base_url:
                change_provider("custom", config_data)
            else:
                st.error("API key and Base URL are required")


def change_provider(provider: str, config_data: Optional[Dict] = None):
    """Change the current provider
    
    Args:
        provider: Provider name
        config_data: Optional configuration data
    """
    if config_data:
        # First update provider configuration
        result = make_request(
            f"{BACKEND_URL}/provider/config",
            method="POST",
            json={"provider": provider, "config": config_data},
            spinner_text=f"Configuring {provider}..."
        )
        
        if not result:
            return
    
    # Then change the provider
    result = make_request(
        f"{BACKEND_URL}/provider",
        method="POST",
        json={"llm_provider": provider},
        spinner_text=f"Changing provider to {provider}..."
    )
    
    if result:
        st.session_state.current_provider = provider
        st.success(f"Provider changed to {provider}")
        
        # Refresh providers
        load_providers()


# Sidebar navigation and provider selection
with st.sidebar:
    st.markdown("### Navigation")
    option = st.radio(
        "Select Operation",
        ["Chat with Documents", "Upload Documents", "Manage Providers", "System Status"]
    )
    
    st.markdown("---")
    st.markdown("### Provider")
    
    # Load providers if not loaded
    if not st.session_state.available_providers:
        load_providers()
    
    # Display current provider
    current = st.session_state.current_provider or "Loading..."
    st.markdown(f"**Current provider**: {current.upper()}")
    
    # Provider quick change
    providers = list(st.session_state.available_providers.keys())
    if providers:
        selected_provider = st.selectbox(
            "Change provider",
            providers,
            index=providers.index(current) if current in providers else 0
        )
        
        if st.button("Switch Provider") and selected_provider != current:
            change_provider(selected_provider)


# Main content area based on selected option
if option == "Chat with Documents":
    st.markdown("## Chat with Documents")
    
    # Check document count and show warning if no documents
    status_result = make_request(
        f"{BACKEND_URL}/status",
        show_spinner=False
    )
    
    if status_result and "status" in status_result:
        indexed_docs = status_result["status"].get("indexed_documents", {})
        doc_count = indexed_docs.get("document_count", 0)
        
        if doc_count == 0:
            st.warning("No documents have been indexed yet. Please upload documents first.")
    
    # Query input
    question = st.text_input("Ask a question about your documents:", key="question_input")
    
    # Provider override option
    use_override = st.checkbox("Use different provider for this query", key="override_checkbox")
    
    if use_override and st.session_state.available_providers:
        override_provider = st.selectbox(
            "Select provider for this query:",
            list(st.session_state.available_providers.keys()),
            key="override_provider"
        )
    else:
        override_provider = None
    
    # Submit button
    if st.button("Ask Question", key="ask_button"):
        if not question.strip():
            st.error("Please enter a question")
        else:
            provider = override_provider if use_override else None
            handle_query(question, provider)
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Chat History")
        
        for i, exchange in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {exchange['question']}", expanded=i == 0):
                st.markdown(f"**Provider**: {exchange['provider'].upper()}")
                st.markdown(f"**Answer**: {exchange['answer']}")
                
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.experimental_rerun()

elif option == "Upload Documents":
    st.markdown("## Upload Documents")
    st.markdown("""
    ### Instructions
    1. Select a document file to upload
    2. Supported formats: TXT, PDF, DOCX, MD, HTML, CSV
    3. Maximum file size: 20MB
    4. Documents will be processed and added to the knowledge base
    """)
    
    uploaded_file = st.file_uploader(
        "Upload a document:",
        type=["txt", "pdf", "docx", "md", "html", "csv"],
        key="doc_uploader"
    )
    
    if uploaded_file:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB",
            "File type": uploaded_file.type
        }
        
        st.json(file_details)
        
        if st.button("Process Document", key="process_doc_button"):
            with st.spinner("Processing document..."):
                files = {"file": uploaded_file}
                result = make_request(
                    f"{BACKEND_URL}/build_index",
                    method="POST",
                    files=files,
                    timeout=300  # 5 minute timeout
                )
                
                if result:
                    st.success(result.get("message", "Document processed successfully!"))
                    
                    # Add to documents list
                    st.session_state.documents.append({
                        "filename": result.get("filename", uploaded_file.name),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
    
    # Display indexed documents
    status_result = make_request(
        f"{BACKEND_URL}/status",
        show_spinner=False
    )
    
    if status_result and "status" in status_result:
        indexed_docs = status_result["status"].get("indexed_documents", {})
        doc_count = indexed_docs.get("document_count", 0)
        
        if doc_count > 0:
            st.markdown(f"### Indexed Documents ({doc_count} documents)")
            
            if st.session_state.documents:
                for doc in st.session_state.documents:
                    st.markdown(f"- **{doc['filename']}** (added: {doc['timestamp']})")
            else:
                st.info("Documents have been indexed but details are not available.")
            
            if st.button("Clear All Documents", key="clear_docs_button"):
                result = make_request(
                    f"{BACKEND_URL}/clear_index",
                    method="DELETE",
                    spinner_text="Clearing index..."
                )
                
                if result:
                    st.success("All documents have been removed from the index")
                    st.session_state.documents = []
                    st.experimental_rerun()

elif option == "Manage Providers":
    st.markdown("## Manage LLM Providers")
    
    # Load providers if not loaded
    if not st.session_state.available_providers:
        load_providers()
    
    # Display current provider status
    st.markdown("### Current Configuration")
    current = st.session_state.current_provider or "None"
    st.markdown(f"**Active provider**: {current.upper()}")
    
    # Provider tabs
    providers = ["ollama", "openai", "volces", "custom"]
    tabs = st.tabs([p.upper() for p in providers])
    
    for i, provider in enumerate(providers):
        with tabs[i]:
            # Get provider status
            provider_info = st.session_state.available_providers.get(provider, {})
            is_available = provider_info.get("available", False)
            is_current = provider_info.get("current", False)
            
            # Show status
            if is_current:
                st.markdown('<div class="success-box">✅ This provider is currently active</div>', unsafe_allow_html=True)
            
            if is_available and not is_current:
                st.markdown('<div class="info-box">✅ This provider is available but not active</div>', unsafe_allow_html=True)
                if st.button(f"Activate {provider.upper()} Provider", key=f"activate_{provider}"):
                    change_provider(provider)
            elif not is_available:
                st.markdown('<div class="warning-box">⚠️ This provider is not properly configured or not available</div>', unsafe_allow_html=True)
            
            # Configuration form
            provider_config_form(provider)
    
    # Refresh button
    if st.button("Refresh Provider Status"):
        load_providers()
        st.experimental_rerun()

elif option == "System Status":
    st.markdown("## System Status")
    
    if st.button("Check System Status", key="check_system"):
        result = make_request(
            f"{BACKEND_URL}/status",
            spinner_text="Checking system status..."
        )
        
        if result and "status" in result:
            status = result["status"]
            
            # Create status cards
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Services")
                
                # Current provider status
                current_provider = status.get("current_provider", {})
                provider_name = current_provider.get("name", "unknown")
                provider_status = current_provider.get("status", False)
                
                if provider_status:
                    st.markdown(f'<div class="success-box">✅ {provider_name.upper()} Provider: Running</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="error-box">❌ {provider_name.upper()} Provider: Not available</div>', unsafe_allow_html=True)
                
                # Legacy Ollama status for compatibility
                if "ollama" in status:
                    ollama_status = status.get("ollama", False)
                    if ollama_status:
                        st.markdown('<div class="success-box">✅ Ollama Service: Running</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">⚠️ Ollama Service: Not running</div>', unsafe_allow_html=True)
                
                # ChromaDB status
                chroma_status = status.get("chroma", False)
                if chroma_status:
                    st.markdown('<div class="success-box">✅ ChromaDB: Connected</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="error-box">❌ ChromaDB: Not connected</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Documents")
                
                # Index information
                indexed_docs = status.get("indexed_documents", {})
                doc_count = indexed_docs.get("document_count", 0)
                collection = indexed_docs.get("collection_name", "unknown")
                
                st.markdown(f"**Collection**: {collection}")
                st.markdown(f"**Document count**: {doc_count}")
                
                if doc_count == 0:
                    st.markdown('<div class="warning-box">⚠️ No documents indexed</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-box">✅ Documents available for querying</div>', unsafe_allow_html=True)
            
            # System information
            st.markdown("### System Information")
            
            # Create a placeholder for system metrics
            system_metrics = {
                "Backend API": f"{BACKEND_URL}",
                "Python Version": "3.10",
                "Environment": "Development"
            }
            
            st.json(system_metrics)
            
            # Action buttons
            if st.button("Restart Services"):
                st.warning("This functionality is not yet implemented")
        
        else:
            st.error("Failed to retrieve system status")

# Add footer
st.markdown("---")
st.markdown("GraphRAG - Powered by llama_index and FastAPI")