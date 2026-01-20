import os
import glob
from typing import List, Dict, Any
import logging
from .pdf_parser import PDFParser
from .ocr import OCRProcessor
from .chunker import TextChunker
from .embedder import EmbeddingGenerator
from .vectorstore import FAISSVectorStore
from ..config import settings

logger = logging.getLogger(__name__)

class DocumentIndexer:
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.ocr_processor = OCRProcessor()
        self.chunker = TextChunker(chunk_size=500, overlap=50)
        # Embeddings: prefer OpenAI (reliable), otherwise fall back to local model,
        # otherwise fall back to hashing embeddings (implemented inside EmbeddingGenerator).
        if (settings.GOOGLE_API_KEY and 
            settings.GOOGLE_API_KEY.strip() and 
            settings.GOOGLE_API_KEY != "PUT_YOUR_GOOGLE_API_KEY_HERE" and
            settings.GOOGLE_API_KEY != "your_google_gemini_api_key_here"):
            self.embedder = EmbeddingGenerator(
                api_key=settings.GOOGLE_API_KEY,
                use_google=True
            )
        elif settings.OPENAI_API_KEY:
            self.embedder = EmbeddingGenerator(
                api_key=settings.OPENAI_API_KEY,
                use_local=False
            )
        else:
            self.embedder = EmbeddingGenerator(
                use_local=True
            )
        # Store vector stores for different PDF types
        self.vector_stores = {}
        self._dimension = self.embedder.get_embedding_dimension()
    
    def _get_vector_store(self, pdf_type: str = "chatbot") -> FAISSVectorStore:
        """Get or create vector store for a specific PDF type"""
        if pdf_type not in self.vector_stores:
            self.vector_stores[pdf_type] = FAISSVectorStore(
                dimension=self._dimension,
                pdf_type=pdf_type
            )
        return self.vector_stores[pdf_type]
    
    def index_directory(self, directory_path: str, incremental: bool = False, pdf_type: str = "chatbot") -> Dict[str, Any]:
        """
        Index all PDF files in a directory
        
        Args:
            directory_path: Path to directory containing PDFs
            incremental: If True, only index files that haven't been indexed yet
        """
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return {'error': 'Directory not found', 'processed_files': 0}
        
        # Find all PDF files
        pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Get vector store for this PDF type
        vector_store = self._get_vector_store(pdf_type)
        
        # If incremental, check which files are already indexed
        indexed_files = set()
        if incremental:
            # Get list of already indexed files from metadata
            stats = vector_store.get_stats()
            # Note: This is a simple check, in production you'd track indexed files more carefully
            logger.info("Incremental indexing mode: will process all files (full tracking not implemented)")
        
        processed_files = 0
        total_chunks = 0
        errors = []
        
        for pdf_file in pdf_files:
            try:
                # Skip if already indexed in incremental mode
                if incremental and os.path.basename(pdf_file) in indexed_files:
                    logger.info(f"Skipping already indexed: {os.path.basename(pdf_file)}")
                    continue
                
                logger.info(f"Processing: {os.path.basename(pdf_file)}")
                
                # Parse PDF
                pdf_data = self.pdf_parser.extract_text_from_pdf(pdf_file)
                if 'error' in pdf_data:
                    errors.append(f"{pdf_file}: {pdf_data['error']}")
                    continue
                
                # Apply OCR if needed
                if pdf_data.get('needs_ocr', False):
                    logger.info(f"Applying OCR to {os.path.basename(pdf_file)}")
                    pdf_data = self.ocr_processor.extract_text_with_ocr(pdf_data)
                
                # Create chunks
                chunks = self.chunker.process_pdf_pages(pdf_data)
                if not chunks:
                    logger.warning(f"No chunks created for {os.path.basename(pdf_file)}")
                    continue
                
                # Generate embeddings
                texts = [chunk['text'] for chunk in chunks]
                embeddings = self.embedder.generate_embeddings(texts)
                
                if not embeddings:
                    logger.error(f"Failed to generate embeddings for {os.path.basename(pdf_file)}")
                    continue
                
                # Add to vector store
                metadata = []
                for chunk in chunks:
                    chunk_metadata = chunk['metadata'].copy()
                    chunk_metadata['text'] = chunk['text']  # Add text to metadata
                    metadata.append(chunk_metadata)
                vector_store.add_vectors(embeddings, metadata)
                
                processed_files += 1
                total_chunks += len(chunks)
                logger.info(f"Successfully processed {os.path.basename(pdf_file)}: {len(chunks)} chunks")
                
            except Exception as e:
                error_msg = f"{pdf_file}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        return {
            'processed_files': processed_files,
            'total_chunks': total_chunks,
            'errors': errors,
            'vector_store_stats': vector_store.get_stats()
        }
    
    def index_single_file(self, file_path: str, pdf_type: str = "chatbot") -> Dict[str, Any]:
        """
        Index a single PDF file
        
        Args:
            file_path: Path to PDF file
            pdf_type: Type of PDF (chatbot, submission, notification)
            
        Returns:
            Dictionary with indexing result
        """
        if not os.path.exists(file_path):
            return {'error': 'File not found', 'processed': False}
        
        try:
            vector_store = self._get_vector_store(pdf_type)
            logger.info(f"Indexing single file: {os.path.basename(file_path)}")
            
            # Parse PDF
            pdf_data = self.pdf_parser.extract_text_from_pdf(file_path)
            if 'error' in pdf_data:
                return {'error': pdf_data['error'], 'processed': False}
            
            # Apply OCR if needed
            if pdf_data.get('needs_ocr', False):
                logger.info(f"Applying OCR to {os.path.basename(file_path)}")
                pdf_data = self.ocr_processor.extract_text_with_ocr(pdf_data)
            
            # Create chunks
            chunks = self.chunker.process_pdf_pages(pdf_data)
            if not chunks:
                return {'error': 'No chunks created', 'processed': False}
            
            # Generate embeddings
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedder.generate_embeddings(texts)
            
            if not embeddings:
                return {'error': 'Failed to generate embeddings', 'processed': False}
            
            # Add to vector store
            metadata = []
            for chunk in chunks:
                chunk_metadata = chunk['metadata'].copy()
                chunk_metadata['text'] = chunk['text']
                metadata.append(chunk_metadata)
            vector_store.add_vectors(embeddings, metadata)
            
            logger.info(f"Successfully indexed {os.path.basename(file_path)}: {len(chunks)} chunks")
            
            return {
                'processed': True,
                'file_name': os.path.basename(file_path),
                'chunks': len(chunks),
                'vector_store_stats': vector_store.get_stats()
            }
            
        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {str(e)}")
            return {'error': str(e), 'processed': False}
    
    def search_documents(self, query: str, k: int = 5, pdf_type: str = "chatbot") -> List[Dict[str, Any]]:
        """Search for relevant document chunks"""
        if not query.strip():
            return []
        
        try:
            vector_store = self._get_vector_store(pdf_type)
            # Generate query embedding
            query_embeddings = self.embedder.generate_embeddings([query])
            if not query_embeddings:
                return []
            
            # Search vector store
            results = vector_store.search(query_embeddings[0], k=k)
            
            # Format results
            formatted_results = []
            for metadata, score in results:
                formatted_results.append({
                    'text': metadata.get('text', ''),
                    'file_name': metadata.get('file_name', ''),
                    'page_number': metadata.get('page_number', 0),
                    'score': score,
                    'metadata': metadata
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []
    
    def get_stats(self, pdf_type: str = "chatbot") -> Dict[str, Any]:
        """Get indexing statistics for a specific PDF type"""
        vector_store = self._get_vector_store(pdf_type)
        return vector_store.get_stats()
    
    def clear_index(self, pdf_type: str = "chatbot"):
        """Clear all indexed documents for a specific PDF type"""
        vector_store = self._get_vector_store(pdf_type)
        vector_store.clear()
        logger.info(f"Cleared document index for {pdf_type}")
