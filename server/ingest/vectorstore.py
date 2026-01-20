import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self, dimension: int, index_path: str = None, pdf_type: str = "chatbot"):
        self.dimension = dimension
        self.pdf_type = pdf_type
        # Use relative path from project root if not specified
        if index_path is None:
            from ..config import settings
            # Create separate index files for each PDF type
            base_path = os.path.join(settings.DATA_FOLDER, f"faiss_index_{pdf_type}")
            self.index_path = base_path
        else:
            self.index_path = index_path
        self.metadata_path = f"{self.index_path}_metadata.pkl"
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Initialize or load index
        self.index = self._load_or_create_index()
        self.metadata = self._load_metadata()
    
    def _load_or_create_index(self):
        """Load existing index or create new one"""
        if os.path.exists(f"{self.index_path}.index"):
            try:
                index = faiss.read_index(f"{self.index_path}.index")
                # Check if dimension matches
                if index.d != self.dimension:
                    logger.warning(f"FAISS index dimension mismatch: existing={index.d}, required={self.dimension}. Recreating index.")
                    # Remove old index and metadata
                    try:
                        os.remove(f"{self.index_path}.index")
                        if os.path.exists(self.metadata_path):
                            os.remove(self.metadata_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove old index files: {str(e)}")
                    # Create new index with correct dimension
                    index = faiss.IndexFlatIP(self.dimension)
                    logger.info(f"Created new FAISS index with dimension {self.dimension}")
                    return index
                logger.info(f"Loaded existing FAISS index with {index.ntotal} vectors (dimension: {index.d})")
                return index
            except Exception as e:
                logger.warning(f"Failed to load existing index: {str(e)}")
        
        # Create new index
        index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        logger.info(f"Created new FAISS index with dimension {self.dimension}")
        return index
    
    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load metadata associated with vectors"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                logger.info(f"Loaded metadata for {len(metadata)} vectors")
                return metadata
            except Exception as e:
                logger.warning(f"Failed to load metadata: {str(e)}")
        
        return []
    
    def _save_metadata(self):
        """Save metadata to disk"""
        try:
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
    
    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]):
        """Add vectors and their metadata to the index"""
        if not vectors or not metadata:
            return
        
        # Normalize vectors for cosine similarity
        vectors_array = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors_array)
        
        # Add to index
        self.index.add(vectors_array)
        
        # Add metadata
        self.metadata.extend(metadata)
        
        # Save to disk
        self._save_index()
        self._save_metadata()
        
        logger.info(f"Added {len(vectors)} vectors to index. Total: {self.index.ntotal}")
    
    def _save_index(self):
        """Save index to disk"""
        try:
            faiss.write_index(self.index, f"{self.index_path}.index")
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
    
    def search(self, query_vector: List[float], k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar vectors"""
        if self.index.ntotal == 0:
            return []
        
        # Normalize query vector
        query_array = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        # Search
        scores, indices = self.index.search(query_array, min(k, self.index.ntotal))
        
        # Return results with metadata
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                results.append((self.metadata[idx], float(score)))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': 'FAISS_FlatIP'
        }
    
    def clear(self):
        """Clear all vectors and metadata"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        self._save_index()
        self._save_metadata()
        logger.info("Cleared vector store")

