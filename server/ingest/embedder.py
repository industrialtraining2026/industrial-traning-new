import openai
from openai import OpenAI
import google.generativeai as genai
import numpy as np
from typing import List, Dict, Any
import logging
import os
import re
import hashlib

logger = logging.getLogger(__name__)

# Try to import sentence-transformers, but make it optional
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    logger.warning("sentence-transformers not available. OpenAI API key is required for embeddings.")

class EmbeddingGenerator:
    def __init__(self, api_key: str = None, use_local: bool = False, use_google: bool = False, use_groq: bool = False):
        self.api_key = api_key
        self.use_local = use_local
        self.use_google = use_google
        self.use_groq = use_groq
        self.model = None
        self.use_hashing = False
        self.openai_client = None  # Explicit OpenAI client
        
        if use_local:
            # Try to use local sentence transformer model
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning("sentence-transformers not available, falling back to OpenAI embeddings")
                self.use_local = False
                if api_key:
                    try:
                        self.openai_client = OpenAI(api_key=api_key)
                    except Exception as e:
                        logger.warning(f"Failed to initialize OpenAI client: {str(e)}")
                        self.openai_client = None
                self.embedding_dim = 1536
            else:
                try:
                    self.model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.embedding_dim = 384
                except Exception as e:
                    logger.error(f"Failed to load sentence-transformers model: {str(e)}, falling back to OpenAI")
                    self.use_local = False
                    if api_key:
                        try:
                            self.openai_client = OpenAI(api_key=api_key)
                        except Exception as e2:
                            logger.warning(f"Failed to initialize OpenAI client: {str(e2)}")
                            self.openai_client = None
                    self.embedding_dim = 1536
        elif use_groq:
            # Groq uses OpenAI-compatible API, so we can use OpenAI client
            # Note: Groq may not have embeddings API, so we try with OpenAI endpoint
            if api_key:
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI client for Groq: {str(e)}")
                    self.openai_client = None
                # Try to use OpenAI-compatible endpoint for Groq if available
                # For now, we'll use OpenAI embeddings API format
                # If Groq doesn't support embeddings, this will fallback to local
                try:
                    # Test if embeddings work with OpenAI client
                    # We'll try to use it during generation
                    self.embedding_dim = 1536  # OpenAI embedding dimension
                    logger.info("Using Groq API key for embeddings (OpenAI-compatible)")
                except Exception as e:
                    logger.warning(f"Groq may not support embeddings, falling back: {str(e)}")
                    # Fallback to local if available
                    if SENTENCE_TRANSFORMERS_AVAILABLE:
                        try:
                            self.use_local = True
                            self.use_groq = False
                            self.model = SentenceTransformer('all-MiniLM-L6-v2')
                            self.embedding_dim = 384
                            logger.info("Falling back to local embeddings (Groq may not support embeddings)")
                        except Exception as e2:
                            raise ValueError(f"Groq doesn't support embeddings and local model failed: {str(e2)}")
                    else:
                        raise ValueError("Groq doesn't support embeddings and sentence-transformers not available")
            else:
                raise ValueError("Groq API key not provided")
        elif use_google:
            # Use Google AI embeddings
            # Note: Google AI doesn't have direct embedding API, so we use local model
            # But we need to check if API key is valid first
            if api_key and api_key.strip() and api_key != "your_google_gemini_api_key_here":
                try:
                    genai.configure(api_key=api_key)
                    # Google AI doesn't have embedding API, so fallback to local or OpenAI
                    if SENTENCE_TRANSFORMERS_AVAILABLE:
                        try:
                            self.use_local = True
                            self.model = SentenceTransformer('all-MiniLM-L6-v2')
                            self.embedding_dim = 384
                            logger.info("Google API key provided but using local embeddings (Google doesn't have embedding API)")
                        except Exception as e:
                            logger.warning(f"Failed to load sentence-transformers, using OpenAI: {str(e)}")
                            self.use_local = False
                            if self.api_key:
                                try:
                                    self.openai_client = OpenAI(api_key=self.api_key)
                                except Exception as e2:
                                    logger.warning(f"Failed to initialize OpenAI client: {str(e2)}")
                                    self.openai_client = None
                            self.embedding_dim = 1536
                    else:
                        logger.warning("sentence-transformers not available, using OpenAI embeddings")
                        self.use_local = False
                        # Try to use OpenAI API key if available
                        openai_key = os.getenv("OPENAI_API_KEY", "")
                        if openai_key:
                            try:
                                self.openai_client = OpenAI(api_key=openai_key)
                            except Exception as e:
                                logger.warning(f"Failed to initialize OpenAI client: {str(e)}")
                                self.openai_client = None
                            self.embedding_dim = 1536
                        else:
                            raise ValueError("No embedding solution available: sentence-transformers not installed and no OpenAI API key")
                except Exception as e:
                    logger.warning(f"Failed to configure Google API: {str(e)}")
                    if SENTENCE_TRANSFORMERS_AVAILABLE:
                        try:
                            self.use_local = True
                            self.model = SentenceTransformer('all-MiniLM-L6-v2')
                            self.embedding_dim = 384
                        except Exception as e2:
                            logger.error(f"Failed to load sentence-transformers: {str(e2)}")
                            raise ValueError("No embedding solution available")
                    else:
                        raise ValueError("No embedding solution available: sentence-transformers not installed")
            else:
                # No valid API key, try local model
                if SENTENCE_TRANSFORMERS_AVAILABLE:
                    try:
                        self.use_local = True
                        self.model = SentenceTransformer('all-MiniLM-L6-v2')
                        self.embedding_dim = 384
                    except Exception as e:
                        logger.error(f"Failed to load sentence-transformers: {str(e)}")
                        raise ValueError("No embedding solution available: sentence-transformers failed and no API key")
                else:
                    raise ValueError("No embedding solution available: sentence-transformers not installed and no API key")
        else:
            # Use OpenAI embeddings
            if api_key:
                # Use explicit OpenAI client instead of module-level API
                # This avoids issues with proxies parameter in some environments
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                    logger.info("OpenAI client initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI client: {str(e)}, will fallback if API calls fail")
                    self.openai_client = None
            self.embedding_dim = 1536  # text-embedding-ada-002 dimension

        # Final fallback: hashing embeddings (pure python, no extra deps)
        # This keeps FAISS rebuild working on Render even without OpenAI + sentence-transformers.
        if (not getattr(self, "embedding_dim", None)) or (
            (not self.use_local) and (not self.use_google) and (not self.use_groq) and (not self.api_key)
        ):
            self._enable_hashing_fallback()
        elif (not SENTENCE_TRANSFORMERS_AVAILABLE) and self.use_local:
            self._enable_hashing_fallback()
        elif (not SENTENCE_TRANSFORMERS_AVAILABLE) and (not self.api_key) and (not self.use_groq) and (not self.use_google):
            self._enable_hashing_fallback()

    def _enable_hashing_fallback(self):
        self.use_hashing = True
        self.use_local = False
        self.use_google = False
        self.use_groq = False
        self.model = None
        self.embedding_dim = 384
        logger.warning("Using hashing embeddings fallback (no OpenAI key / no sentence-transformers). Quality will be lower.")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not texts:
            return []
        
        try:
            if self.use_hashing:
                return self._generate_hashing_embeddings(texts)
            if self.use_local:
                return self._generate_local_embeddings(texts)
            elif self.use_groq:
                # Groq uses OpenAI-compatible API, but may not support embeddings
                # Try OpenAI format first
                try:
                    return self._generate_openai_embeddings(texts)
                except Exception as e:
                    logger.warning(f"Groq embeddings failed: {str(e)}, trying fallback")
                    # Fallback to local if available
                    if SENTENCE_TRANSFORMERS_AVAILABLE and not self.use_local:
                        logger.info("Falling back to local embeddings")
                        self.use_local = True
                        self.use_groq = False
                        try:
                            self.model = SentenceTransformer('all-MiniLM-L6-v2')
                            self.embedding_dim = 384
                            return self._generate_local_embeddings(texts)
                        except Exception as e2:
                            logger.error(f"Local embeddings fallback failed: {str(e2)}")
                    raise e
            elif self.use_google:
                return self._generate_google_embeddings(texts)
            else:
                return self._generate_openai_embeddings(texts)
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Fallback to local model if OpenAI fails
            if not self.use_local and SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    logger.info("Falling back to local embeddings")
                    self.use_local = True
                    self.model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.embedding_dim = 384
                    return self._generate_local_embeddings(texts)
                except Exception as e2:
                    logger.error(f"Failed to fallback to local embeddings: {str(e2)}")
            # Last fallback
            try:
                self._enable_hashing_fallback()
                return self._generate_hashing_embeddings(texts)
            except Exception:
                pass
            return []

    def _generate_hashing_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Simple hashing-trick embeddings.
        - Pure Python + numpy only (works on Render without rust/tokenizers)
        - Not as good as real embeddings, but keeps indexing functional.
        """
        dim = int(self.embedding_dim or 384)
        vectors = np.zeros((len(texts), dim), dtype=np.float32)
        token_re = re.compile(r"[a-zA-Z0-9_]+")
        for i, text in enumerate(texts):
            if not text:
                continue
            tokens = token_re.findall(text.lower())
            if not tokens:
                continue
            for tok in tokens:
                h = hashlib.md5(tok.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "little") % dim
                sign = 1.0 if (h[4] % 2 == 0) else -1.0
                vectors[i, idx] += sign
            # L2 normalize
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors.tolist()
    
    def _generate_google_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Google AI (actually uses local model)"""
        # Google AI doesn't have a direct embedding API, so we use local model
        # This method is kept for compatibility but always uses local embeddings
        return self._generate_local_embeddings(texts)
    
    def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        try:
            # Use explicit client if available, otherwise fallback to module-level API
            if self.openai_client:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=texts
                )
            else:
                # Fallback to module-level API (for backward compatibility)
                # But initialize client if we have API key
                if self.api_key:
                    self.openai_client = OpenAI(api_key=self.api_key)
                    response = self.openai_client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=texts
                    )
                else:
                    raise ValueError("No OpenAI API key available")
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
