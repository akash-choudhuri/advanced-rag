"""
Document processing module for RAG application.
Handles document loading, chunking, and preprocessing.
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path

from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handle document loading and processing for RAG pipeline."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def load_documents(self, directory_path: str) -> List[Document]:
        """
        Load all documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of loaded documents
        """
        documents = []
        
        try:
            # Load PDF files
            pdf_loader = DirectoryLoader(
                directory_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            
            # Load text files
            txt_loader = DirectoryLoader(
                directory_path,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)
            
            logger.info(f"Loaded {len(documents)} documents from {directory_path}")
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of document chunks
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking documents: {str(e)}")
            return []
    
    def process_documents(self, directory_path: str) -> List[Document]:
        """
        Complete document processing pipeline.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of processed document chunks
        """
        # Load documents
        documents = self.load_documents(directory_path)
        
        if not documents:
            logger.warning("No documents found to process")
            return []
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Add metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunks)
            })
        
        return chunks
    
    def get_document_stats(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about processed documents.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Dictionary with document statistics
        """
        if not chunks:
            return {}
            
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        sources = set(chunk.metadata.get("source", "unknown") for chunk in chunks)
        
        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "unique_sources": len(sources),
            "avg_chunk_size": total_chars / len(chunks) if chunks else 0,
            "sources": list(sources)
        }