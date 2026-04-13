"""
Configuration module for RAG application.
Loads environment variables and provides default settings.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG application."""
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    DOCUMENTS_DIR = DATA_DIR / "documents"
    VECTOR_DB_PATH = DATA_DIR / "chroma_db"
    
    # Hugging Face Configuration
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
    
    # Model Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    LLM_MODEL = os.getenv("LLM_MODEL", "distilgpt2")
    
    # Document Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    
    # Application Configuration
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 512))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))
    
    # Vector Database Configuration
    COLLECTION_NAME = "rag_documents"
    SIMILARITY_THRESHOLD = 0.0
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.DOCUMENTS_DIR.mkdir(exist_ok=True)
        cls.VECTOR_DB_PATH.mkdir(exist_ok=True)
    
    @classmethod
    def get_model_config(cls):
        """Get model configuration as dictionary."""
        return {
            "embedding_model": cls.EMBEDDING_MODEL,
            "llm_model": cls.LLM_MODEL,
            "max_tokens": cls.MAX_TOKENS,
            "temperature": cls.TEMPERATURE,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP
        }
    
    @classmethod
    def validate_config(cls):
        """Validate configuration and return any issues."""
        issues = []
        
        # Check if required directories exist (create if not)
        try:
            cls.create_directories()
        except Exception as e:
            issues.append(f"Cannot create directories: {e}")
        
        # Validate numeric values
        if cls.CHUNK_SIZE <= 0:
            issues.append("CHUNK_SIZE must be positive")
        
        if cls.CHUNK_OVERLAP < 0:
            issues.append("CHUNK_OVERLAP must be non-negative")
        
        if cls.MAX_TOKENS <= 0:
            issues.append("MAX_TOKENS must be positive")
        
        if not (0.0 <= cls.TEMPERATURE <= 2.0):
            issues.append("TEMPERATURE must be between 0.0 and 2.0")
        
        return issues


# Initialize configuration
Config.create_directories()