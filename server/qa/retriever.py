from typing import List, Dict, Any
import logging
from difflib import SequenceMatcher
from ..ingest.indexer import DocumentIndexer

logger = logging.getLogger(__name__)

class DocumentRetriever:
    def __init__(self, indexer: DocumentIndexer):
        self.indexer = indexer
    
    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for a query"""
        try:
            # Get more results to have better selection
            results = self.indexer.search_documents(query, k=k*4)  # Get 4x more results for better coverage
            
            # Log the actual scores for debugging
            if results:
                scores = [result.get('score', 0) for result in results]
                logger.info(f"Retrieval scores: {scores[:5]} (top 5)")
                # Show first result text for debugging
                if results:
                    first_result = results[0]
                    logger.info(f"First result text: {first_result.get('text', '')[:100]}...")
            
            # More lenient filtering - lower threshold and better duplicate handling
            filtered_results = []
            seen_texts = set()
            
            for result in results:
                score = result.get('score', 0)
                text = result.get('text', '').strip()
                
                # Much lower threshold - accept more results
                if score > 0.001 and text:  # Very low threshold
                    # Better duplicate detection - check for substantial similarity
                    is_duplicate = False
                    for seen_text in seen_texts:
                        # Check if texts are too similar (more than 80% overlap)
                        if self._text_similarity(text, seen_text) > 0.8:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        seen_texts.add(text)
                        filtered_results.append(result)
                        
                        # Stop when we have enough unique results
                        if len(filtered_results) >= k:
                            break
            
            # If we still don't have enough results, lower the threshold even more
            if len(filtered_results) < k and results:
                for result in results:
                    if result not in filtered_results:
                        text = result.get('text', '').strip()
                        if text:  # Accept any text content
                            filtered_results.append(result)
                            if len(filtered_results) >= k:
                                break
            
            logger.info(f"Retrieved {len(filtered_results)} relevant chunks for query: {query[:50]}...")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            return []
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context for LLM"""
        if not chunks:
            return ""
        
        context_parts = []
        for chunk in chunks:
            text = chunk.get('text', '').strip()
            
            if text:
                context_parts.append(text)
        
        return "\n\n".join(context_parts)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def get_confidence_score(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieval results"""
        if not chunks:
            return 0.0
        
        # Average score of top results
        scores = [chunk.get('score', 0) for chunk in chunks]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Normalize to 0-1 range (assuming scores are typically 0-1)
        return min(avg_score, 1.0)
