import openai
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, api_key: str = None, use_local: bool = False, use_google: bool = False):
        self.api_key = api_key
        self.use_local = use_local
        self.use_google = use_google
        
        if use_local:
            # Use local sentence transformer model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384
        elif use_google:
            # Use Google AI embeddings
            # Note: Google AI doesn't have direct embedding API, so we use local model
            # But we need to check if API key is valid first
            if api_key and api_key.strip() and api_key != "your_google_gemini_api_key_here":
                try:
                    genai.configure(api_key=api_key)
                    # Google AI doesn't have embedding API, so fallback to local
                    self.use_local = True
                    self.model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.embedding_dim = 384
                    logger.info("Google API key provided but using local embeddings (Google doesn't have embedding API)")
                except Exception as e:
                    logger.warning(f"Failed to configure Google API, using local embeddings: {str(e)}")
                    self.use_local = True
                    self.model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.embedding_dim = 384
            else:
                # No valid API key, use local model
                self.use_local = True
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_dim = 384
        else:
            # Use OpenAI embeddings
            if api_key:
                openai.api_key = api_key
            self.embedding_dim = 1536  # text-embedding-ada-002 dimension
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not texts:
            return []
        
        try:
            if self.use_local:
                return self._generate_local_embeddings(texts)
            elif self.use_google:
                return self._generate_google_embeddings(texts)
            else:
                return self._generate_openai_embeddings(texts)
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Fallback to local model if OpenAI fails
            if not self.use_local:
                logger.info("Falling back to local embeddings")
                self.use_local = True
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_dim = 384
                return self._generate_local_embeddings(texts)
            return []
    
    def _generate_google_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Google AI (actually uses local model)"""
        # Google AI doesn't have a direct embedding API, so we use local model
        # This method is kept for compatibility but always uses local embeddings
        return self._generate_local_embeddings(texts)
    
    def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        try:
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding error: {str(e)}")
            raise e
    
    def _generate_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local sentence transformer"""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Local embedding error: {str(e)}")
            raise e
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.embedding_dim
