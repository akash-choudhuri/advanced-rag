"""
Streamlit web application for RAG system.
Provides a user-friendly interface for document upload, processing, and querying.
"""

import streamlit as st
import os
import sys
from pathlib import Path
import logging
from typing import List, Dict, Any
import tempfile
import shutil

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_interface import LLMInterface, SimpleRAGChain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system components."""
    try:
        # Initialize components
        doc_processor = DocumentProcessor(
            chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200))
        )
        
        vector_store = VectorStore(
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            persist_directory=os.getenv("VECTOR_DB_PATH", "./data/chroma_db")
        )
        
        llm_interface = LLMInterface(
            model_name=os.getenv("LLM_MODEL", "distilgpt2"),
            max_tokens=int(os.getenv("MAX_TOKENS", 512)),
            temperature=float(os.getenv("TEMPERATURE", 0.7))
        )
        
        rag_chain = SimpleRAGChain(vector_store, llm_interface)
        
        return doc_processor, vector_store, llm_interface, rag_chain
    
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None, None, None, None


def save_uploaded_files(uploaded_files) -> str:
    """Save uploaded files to temporary directory."""
    temp_dir = tempfile.mkdtemp()
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    return temp_dir


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 RAG System</h1>', unsafe_allow_html=True)
    st.markdown("**Retrieval-Augmented Generation with Open-Source Models**")
    
    # Initialize RAG system
    with st.spinner("Initializing RAG system..."):
        doc_processor, vector_store, llm_interface, rag_chain = initialize_rag_system()
    
    if not all([doc_processor, vector_store, llm_interface, rag_chain]):
        st.error("Failed to initialize RAG system. Please check your configuration.")
        return
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Model information
    if llm_interface:
        model_info = llm_interface.get_model_info()
        st.sidebar.subheader("Model Information")
        st.sidebar.write(f"**Model:** {model_info.get('model_name', 'Unknown')}")
        st.sidebar.write(f"**Device:** {model_info.get('device', 'Unknown')}")
        st.sidebar.write(f"**Max Tokens:** {model_info.get('max_tokens', 'Unknown')}")
    
    # Vector store statistics
    if vector_store:
        stats = vector_store.get_collection_stats()
        st.sidebar.subheader("Vector Store Stats")
        st.sidebar.write(f"**Documents:** {stats.get('total_documents', 0)}")
        st.sidebar.write(f"**Embedding Model:** {stats.get('embedding_model', 'Unknown')}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📚 Document Management", "💬 Chat Interface", "📊 System Status"])
    
    # Document Management Tab
    with tab1:
        st.header("Document Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload Documents")
            uploaded_files = st.file_uploader(
                "Choose files",
                type=['pdf', 'txt'],
                accept_multiple_files=True,
                help="Upload PDF or text files to add to the knowledge base"
            )
            
            if uploaded_files:
                st.write(f"Selected {len(uploaded_files)} file(s):")
                for file in uploaded_files:
                    st.write(f"- {file.name} ({file.size} bytes)")
                
                if st.button("Process Documents", type="primary"):
                    with st.spinner("Processing documents..."):
                        try:
                            # Save uploaded files
                            temp_dir = save_uploaded_files(uploaded_files)
                            
                            # Process documents
                            chunks = doc_processor.process_documents(temp_dir)
                            
                            if chunks:
                                # Add to vector store
                                success = vector_store.add_documents(chunks)
                                
                                if success:
                                    st.success(f"Successfully processed {len(chunks)} document chunks!")
                                    
                                    # Show statistics
                                    stats = doc_processor.get_document_stats(chunks)
                                    st.write("**Processing Statistics:**")
                                    col_a, col_b, col_c = st.columns(3)
                                    with col_a:
                                        st.metric("Total Chunks", stats.get('total_chunks', 0))
                                    with col_b:
                                        st.metric("Avg Chunk Size", f"{stats.get('avg_chunk_size', 0):.0f}")
                                    with col_c:
                                        st.metric("Sources", stats.get('unique_sources', 0))
                                else:
                                    st.error("Failed to add documents to vector store.")
                            else:
                                st.error("No document chunks were created.")
                            
                            # Cleanup
                            shutil.rmtree(temp_dir)
                            
                        except Exception as e:
                            st.error(f"Error processing documents: {str(e)}")
        
        with col2:
            st.subheader("Actions")
            
            if st.button("Clear Vector Store", type="secondary"):
                if st.confirm("Are you sure you want to clear all documents?"):
                    with st.spinner("Clearing vector store..."):
                        success = vector_store.clear_collection()
                        if success:
                            st.success("Vector store cleared successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to clear vector store.")
            
            st.subheader("Sample Documents")
            st.write("You can upload sample documents to test the system:")
            st.write("- PDF files")
            st.write("- Text files")
            st.write("- Multiple files at once")
    
    # Chat Interface Tab
    with tab2:
        st.header("Chat with Your Documents")
        
        # Query parameters
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("Parameters")
            top_k = st.slider("Documents to retrieve", 1, 10, 3)
            score_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.0, 0.1)
        
        with col1:
            # Chat interface
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and "sources" in message:
                        with st.expander("View Sources"):
                            for i, source in enumerate(message["sources"], 1):
                                st.write(f"{i}. {source}")
            
            # Chat input
            if prompt := st.chat_input("Ask a question about your documents"):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            response = rag_chain.query(
                                prompt,
                                top_k=top_k,
                                score_threshold=score_threshold
                            )
                            
                            answer = response["answer"]
                            sources = response["sources"]
                            num_retrieved = response["num_retrieved"]
                            
                            st.markdown(answer)
                            
                            if sources:
                                with st.expander(f"View {num_retrieved} Sources"):
                                    for i, source in enumerate(sources, 1):
                                        st.write(f"{i}. {source}")
                            
                            # Add assistant message
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": sources
                            })
                            
                        except Exception as e:
                            error_msg = f"Error generating response: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": error_msg
                            })
            
            # Clear chat button
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
    
    # System Status Tab
    with tab3:
        st.header("System Status")
        
        # System information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Vector Store Information")
            if vector_store:
                stats = vector_store.get_collection_stats()
                st.json(stats)
        
        with col2:
            st.subheader("Model Information")
            if llm_interface:
                model_info = llm_interface.get_model_info()
                st.json(model_info)
        
        # Health checks
        st.subheader("Health Checks")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Vector store health
            try:
                stats = vector_store.get_collection_stats()
                st.success("✅ Vector Store: Healthy")
            except Exception as e:
                st.error(f"❌ Vector Store: {str(e)}")
        
        with col2:
            # LLM health
            try:
                model_info = llm_interface.get_model_info()
                st.success("✅ LLM Interface: Healthy")
            except Exception as e:
                st.error(f"❌ LLM Interface: {str(e)}")
        
        with col3:
            # Document processor health
            try:
                # Test with empty directory
                test_chunks = doc_processor.process_documents("./data/documents")
                st.success("✅ Document Processor: Healthy")
            except Exception as e:
                st.error(f"❌ Document Processor: {str(e)}")


if __name__ == "__main__":
    main()