"""Enhanced RAG service with multi-provider LLM and embedding support"""
import logging
import os
import tempfile
from typing import Optional, List, Dict, Any, Tuple

import chromadb
import requests
from fastapi import UploadFile
from llama_index.core import VectorStoreIndex, Document, Settings as LlamaIndexSettings
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import  DocxReader, UnstructuredReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.readers.docling import DoclingReader
from src.RAG.core.config import settings
from src.RAG.services.base import BaseService, retry_on_failure
from src.RAG.services.llm_adapter import LLMFactory

logger = logging.getLogger(__name__)

# Supported file extensions and their corresponding readers
FILE_READERS = {
    '.txt': None,  # Plain text doesn't need a special reader
    '.pdf': DoclingReader(),
    '.docx': DocxReader(),
    # Generic reader for other formats
    '.md': UnstructuredReader(),
    '.html': UnstructuredReader(),
    '.csv': UnstructuredReader(),
}


class RAGService(BaseService):
    def __init__(self, llm_provider: Optional[str] = None, embed_provider: Optional[str] = None):
        """Initialize the RAG service
        
        Args:
            llm_provider: Provider for the LLM (default: from settings)
            embed_provider: Provider for the embedding model (default: same as llm_provider)
        """
        super().__init__()
        logger.info("Initializing RAG service...")
        
        # Set providers
        self.llm_provider = llm_provider or settings.llm.PROVIDER
        self.embed_provider = embed_provider or self.llm_provider
        
        # Initialize embedding model
        self._init_embedding_model()
        
        # Configure Llama Index settings
        LlamaIndexSettings.embed_model = self.embed_model
        LlamaIndexSettings.node_parser = SentenceSplitter(
            chunk_size=settings.doc_processing.CHUNK_SIZE,
            chunk_overlap=settings.doc_processing.CHUNK_OVERLAP
        )
        
        # Initialize vector store
        self._init_vector_store()
        
        # Create temp directory for file processing
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {self.temp_dir}")
        
        logger.info(f"RAG service initialized with LLM provider: {self.llm_provider}, "
                   f"Embedding provider: {self.embed_provider}")

    def _init_embedding_model(self) -> None:
        """Initialize the embedding model based on provider configuration"""
        try:
            embed_config = settings.get_embedding_config(self.embed_provider)
            self.embed_model = LLMFactory.create_embedding_model(
                provider=self.embed_provider,
                **embed_config
            )
            logger.info(f"Embedding model initialized: {embed_config.get('model_name')}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise RuntimeError(f"Embedding model initialization failed: {str(e)}")

    def _init_vector_store(self) -> None:
        """Initialize the vector store connection"""
        try:
            # Create directories if they don't exist
            os.makedirs(settings.chroma.PERSIST_PATH, exist_ok=True)
            
            # Connect to ChromaDB
            self.db = chromadb.PersistentClient(path=settings.chroma.PERSIST_PATH)
            self.chroma_collection = self.db.get_or_create_collection(settings.chroma.COLLECTION_NAME)
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            
            # Initialize vector store index
            self.index = self._get_or_create_index()
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise RuntimeError(f"Vector store initialization failed: {str(e)}")

    def _get_or_create_index(self) -> VectorStoreIndex:
        """Get existing index or create a new one"""
        try:
            return VectorStoreIndex.from_vector_store(
                self.vector_store,
                embed_model=self.embed_model
            )
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise

    def _init_llm(self) -> Any:
        """Initialize the LLM based on provider configuration"""
        try:
            llm_config = settings.get_llm_config(self.llm_provider)
            return LLMFactory.create_llm(
                provider=self.llm_provider,
                **llm_config
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise RuntimeError(f"LLM initialization failed: {str(e)}")

    def check_provider_available(self, provider: str) -> bool:
        """Check if a specific provider is available
        
        Args:
            provider: Provider name to check
            
        Returns:
            True if the provider is available, False otherwise
        """
        try:
            if provider.lower() == "ollama":
                return self.check_ollama_available()
            elif provider.lower() in ["openai", "volces", "custom"]:
                # For API-based providers, try to create a simple embedding
                embed_config = settings.get_embedding_config(provider)
                model = LLMFactory.create_embedding_model(
                    provider=provider,
                    **embed_config
                )
                # Test with a simple embedding
                _ = model.get_text_embedding("test")
                return True
            else:
                logger.warning(f"Unknown provider: {provider}")
                return False
        except Exception as e:
            logger.warning(f"Provider {provider} check failed: {str(e)}")
            return False

    def check_ollama_available(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = requests.get(
                f"{settings.llm.OLLAMA_BASE_URL}/api/tags",
                timeout=settings.api.REQUEST_TIMEOUT
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama service check failed: {str(e)}")
            return False

    def check_chroma_available(self) -> bool:
        """Check if Chroma is available"""
        try:
            # Try to get collection info
            self.chroma_collection.count()
            return True
        except Exception as e:
            logger.warning(f"ChromaDB check failed: {str(e)}")
            return False

    def get_services_status(self) -> Dict[str, Any]:
        """Get status of all required services"""
        # Check the current provider
        provider_status = self.check_provider_available(self.llm_provider)
        
        # For compatibility, also check Ollama if it's not the current provider
        ollama_status = provider_status if self.llm_provider.lower() == "ollama" else self.check_ollama_available()
        
        return {
            "ollama": ollama_status,
            "current_provider": {
                "name": self.llm_provider,
                "status": provider_status
            },
            "chroma": self.check_chroma_available(),
            "indexed_documents": self.get_index_stats()
        }
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        try:
            doc_count = self.chroma_collection.count()
            return {
                "document_count": doc_count,
                "collection_name": settings.chroma.COLLECTION_NAME
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {str(e)}")
            return {"error": str(e)}

    async def process_file(self, file: UploadFile) -> Tuple[str, List[Document]]:
        """Process an uploaded file and extract documents based on file type"""
        filename = file.filename
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext not in FILE_READERS:
            raise ValueError(f"Unsupported file format: {file_ext}. "
                             f"Supported formats: {', '.join(FILE_READERS.keys())}")
        
        # Save file to temp location
        file_path = os.path.join(self.temp_dir, filename)
        content = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
            
        logger.info(f"Processing file: {filename} ({file_ext})")
        
        try:
            # Process based on file type
            if file_ext == '.txt':
                try:
                    text = content.decode("utf-8")
                    documents = [Document(text=text, metadata={"filename": filename})]
                except UnicodeDecodeError:
                    raise ValueError("File encoding must be UTF-8")
            else:
                reader = FILE_READERS[file_ext]
                documents = reader.load_data(file_path)
                
                # Add metadata to documents
                for doc in documents:
                    if not hasattr(doc, 'metadata') or doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata.update({"filename": filename})
            
            # Clean up temp file
            os.remove(file_path)
            
            return filename, documents
        except Exception as e:
            # Clean up on error
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.error(f"Error processing file {filename}: {str(e)}")
            raise ValueError(f"Error processing file: {str(e)}")

    @retry_on_failure(max_retries=settings.api.MAX_RETRIES, delay=settings.api.RETRY_DELAY)
    async def build_index(self, file: UploadFile) -> Dict[str, str]:
        """Build or update index with documents from uploaded file"""
        if not self.check_provider_available(self.embed_provider):
            raise ValueError(f"{self.embed_provider} provider is not available, please check your configuration")

        try:
            filename, documents = await self.process_file(file)
            
            if not documents:
                raise ValueError("No content extracted from document")
                
            # Insert documents into index
            self.index.insert_nodes(documents)
            
            logger.info(f"Index built/updated successfully with {len(documents)} documents from {filename}")
            return {
                "message": f"Document uploaded successfully! Added {len(documents)} chunks to the index.",
                "filename": filename
            }
        except Exception as e:
            logger.error(f"Index building failed: {str(e)}")
            raise

    @retry_on_failure(max_retries=settings.api.MAX_RETRIES, delay=settings.api.RETRY_DELAY)
    def query(
        self, 
        question: str, 
        similarity_top_k: int = 3,
        llm_provider: Optional[str] = None
    ) -> Optional[str]:
        """Query the RAG system with a question
        
        Args:
            question: The question to answer
            similarity_top_k: Number of similar documents to retrieve
            llm_provider: Override the default LLM provider
            
        Returns:
            Answer to the question
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")
            
        # Use specified provider or default
        provider = llm_provider or self.llm_provider

        if not self.check_provider_available(provider):
            raise ValueError(f"{provider} provider is not available, please check your configuration")

        logger.info(f"Processing question with {provider} provider: {question}")

        try:
            # Initialize LLM with the selected provider
            llm = self._init_llm()
            
            # Create chat engine with context mode
            chat_engine = self.index.as_chat_engine(
                chat_mode=ChatMode.CONTEXT,
                llm=llm,
                similarity_top_k=similarity_top_k
            )
            
            # Get response
            response = chat_engine.chat(question)
            
            # Extract sources for citation
            source_nodes = getattr(response, 'source_nodes', [])
            sources = []
            
            if source_nodes:
                for node in source_nodes:
                    if hasattr(node, 'metadata') and node.metadata:
                        if 'filename' in node.metadata and node.metadata['filename'] not in sources:
                            sources.append(node.metadata['filename'])
            
            # Add sources to response if available
            answer = str(response)
            if sources:
                answer += f"\n\nSources: {', '.join(sources)}"
                
            logger.info("Query completed successfully")
            return answer
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise

    def clear_index(self) -> Dict[str, str]:
        """Clear the entire index"""
        try:
            # Delete collection and recreate it
            self.db.delete_collection(settings.chroma.COLLECTION_NAME)
            self.chroma_collection = self.db.get_or_create_collection(settings.chroma.COLLECTION_NAME)
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            self.index = self._get_or_create_index()
            
            logger.info("Index cleared successfully")
            return {"message": "Index cleared successfully"}
        except Exception as e:
            logger.error(f"Failed to clear index: {str(e)}")
            raise ValueError(f"Failed to clear index: {str(e)}")
    
    def change_provider(self, llm_provider: str, embed_provider: Optional[str] = None) -> Dict[str, str]:
        """Change the LLM and embedding providers
        
        Args:
            llm_provider: New LLM provider name
            embed_provider: New embedding provider name (default: same as llm_provider)
            
        Returns:
            Status message
        """
        try:
            # Check if the new provider is available
            if not self.check_provider_available(llm_provider):
                raise ValueError(f"{llm_provider} provider is not available")
                
            # Use the same provider for embedding if not specified
            embed_provider = embed_provider or llm_provider
            
            # Check if the embedding provider is available
            if not self.check_provider_available(embed_provider):
                raise ValueError(f"{embed_provider} embedding provider is not available")
                
            # Update providers
            self.llm_provider = llm_provider
            self.embed_provider = embed_provider
            
            # Re-initialize embedding model
            self._init_embedding_model()
            
            # Update Llama Index settings
            LlamaIndexSettings.embed_model = self.embed_model
            
            # Re-initialize index with new embedding model
            self.index = self._get_or_create_index()
            
            logger.info(f"Provider changed to: LLM={llm_provider}, Embeddings={embed_provider}")
            return {
                "message": f"Provider changed successfully to {llm_provider}",
                "llm_provider": llm_provider,
                "embed_provider": embed_provider
            }
        except Exception as e:
            logger.error(f"Failed to change provider: {str(e)}")
            raise ValueError(f"Failed to change provider: {str(e)}")
            
    def __del__(self):
        """Clean up resources when the service is destroyed"""
        try:
            # Remove temp directory
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")