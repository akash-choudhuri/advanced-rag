"""
LLM interface module for RAG application.
Handles LLM integration using Hugging Face transformers for open-source models.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMInterface:
    """Handle LLM operations using Hugging Face transformers."""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 max_tokens: int = 512,
                 temperature: float = 0.7):
        """
        Initialize LLM interface.
        
        Args:
            model_name: Name of the Hugging Face model
            max_tokens: Maximum tokens for generation
            temperature: Sampling temperature
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left"
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_full_text=False
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fallback to a simpler model
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback model if the primary model fails."""
        try:
            logger.info("Loading fallback model: distilgpt2")
            
            self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # CPU only for fallback
                return_full_text=False
            )
            
            logger.info("Fallback model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading fallback model: {str(e)}")
            raise
    
    def generate_response(self, 
                         prompt: str,
                         context: List[str] = None,
                         max_length: int = None) -> str:
        """
        Generate response using the LLM.
        
        Args:
            prompt: User query/prompt
            context: List of relevant context documents
            max_length: Maximum length of generated response
            
        Returns:
            Generated response string
        """
        try:
            # Use provided max_length or default
            max_length = max_length or self.max_tokens
            
            # Construct the full prompt with context
            full_prompt = self._construct_rag_prompt(prompt, context)
            
            # Generate response
            outputs = self.generator(
                full_prompt,
                max_new_tokens=max_length,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
            # Extract the generated text
            if outputs and len(outputs) > 0:
                response = outputs[0]['generated_text'].strip()
                # Clean up the response
                response = self._clean_response(response)
                return response
            else:
                return "I apologize, but I couldn't generate a proper response."
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request."
    
    def _construct_rag_prompt(self, query: str, context: List[str] = None) -> str:
        """
        Construct a RAG prompt with context and query.
        
        Args:
            query: User query
            context: List of context documents
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add system instruction
        prompt_parts.append("You are a helpful AI assistant. Use the provided context to answer the user's question accurately and concisely.")
        
        # Add context if available
        if context and len(context) > 0:
            prompt_parts.append("\nContext:")
            for i, ctx in enumerate(context[:3], 1):  # Limit to top 3 contexts
                prompt_parts.append(f"{i}. {ctx}")
        
        # Add query
        prompt_parts.append(f"\nQuestion: {query}")
        prompt_parts.append("\nAnswer:")
        
        return "\n".join(prompt_parts)
    
    def _clean_response(self, response: str) -> str:
        """
        Clean up the generated response.
        
        Args:
            response: Raw generated response
            
        Returns:
            Cleaned response string
        """
        # Remove common artifacts
        response = response.strip()
        
        # Remove repetitive patterns
        lines = response.split('\n')
        cleaned_lines = []
        prev_line = ""
        
        for line in lines:
            line = line.strip()
            if line and line != prev_line:  # Remove empty lines and repetitions
                cleaned_lines.append(line)
                prev_line = line
        
        cleaned_response = '\n'.join(cleaned_lines)
        
        # Truncate at natural stopping points
        for stop_phrase in ['\n\nQuestion:', '\n\nContext:', 'User:', 'Human:']:
            if stop_phrase in cleaned_response:
                cleaned_response = cleaned_response.split(stop_phrase)[0]
        
        return cleaned_response.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "tokenizer_vocab_size": len(self.tokenizer) if self.tokenizer else 0
        }


class SimpleRAGChain:
    """Simple RAG chain combining retrieval and generation."""
    
    def __init__(self, vector_store, llm_interface):
        """
        Initialize RAG chain.
        
        Args:
            vector_store: VectorStore instance
            llm_interface: LLMInterface instance
        """
        self.vector_store = vector_store
        self.llm_interface = llm_interface
    
    def query(self, 
              question: str,
              top_k: int = 3,
              score_threshold: float = 0.0) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Retrieve relevant documents
            retrieved_docs = self.vector_store.similarity_search(
                question, 
                top_k=top_k,
                score_threshold=score_threshold
            )
            
            # Extract context from retrieved documents
            context = [doc['content'] for doc in retrieved_docs]
            
            # Generate answer
            answer = self.llm_interface.generate_response(
                prompt=question,
                context=context
            )
            
            return {
                "question": question,
                "answer": answer,
                "retrieved_documents": retrieved_docs,
                "num_retrieved": len(retrieved_docs),
                "sources": [doc['metadata'].get('source', 'Unknown') for doc in retrieved_docs]
            }
            
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return {
                "question": question,
                "answer": "I apologize, but I encountered an error while processing your question.",
                "retrieved_documents": [],
                "num_retrieved": 0,
                "sources": []
            }