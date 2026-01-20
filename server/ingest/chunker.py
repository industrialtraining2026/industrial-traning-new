import re
import nltk
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TextChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers (common patterns)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip very short lines that might be page numbers
            if len(line) < 2:  # More lenient
                continue
            # Skip lines that are just numbers (page numbers)
            if line.isdigit() and len(line) < 5:  # Only skip short numbers
                continue
            # Skip common header/footer patterns (but be more lenient)
            if any(pattern in line.lower() for pattern in [
                'page', 'of', 'confidential', 'internal use only'
            ]) and len(line) < 20:  # Only skip if line is short
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def split_into_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        if not text or len(text.strip()) < 10:  # Lower threshold
            return []
        
        cleaned_text = self.clean_text(text)
        if not cleaned_text or len(cleaned_text.strip()) < 10:  # Check cleaned text length
            return []
        
        # Split by sentences first
        sentences = nltk.sent_tokenize(cleaned_text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_length + len(sentence) > self.chunk_size and current_chunk:
                chunk_data = {
                    'text': current_chunk.strip(),
                    'metadata': {
                        **metadata,
                        'chunk_length': len(current_chunk),
                        'chunk_type': 'text'
                    }
                }
                chunks.append(chunk_data)
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_length = len(current_chunk)
        
        # Add the last chunk
        if current_chunk.strip():
            chunk_data = {
                'text': current_chunk.strip(),
                'metadata': {
                    **metadata,
                    'chunk_length': len(current_chunk),
                    'chunk_type': 'text'
                }
            }
            chunks.append(chunk_data)
        
        return chunks
    
    def process_pdf_pages(self, pdf_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process all pages of a PDF into chunks"""
        all_chunks = []
        
        for page_data in pdf_data['pages']:
            if not page_data.get('has_text', False):
                continue
            
            text = page_data['text']
            if not text or len(text.strip()) < 20:
                continue
            
            # Create metadata for this page
            page_metadata = {
                'file_name': pdf_data['file_name'],
                'file_path': pdf_data['file_path'],
                'page_number': page_data['page_number'],
                'char_count': page_data['char_count'],
                'ocr_applied': page_data.get('ocr_applied', False)
            }
            
            # Split page into chunks
            page_chunks = self.split_into_chunks(text, page_metadata)
            all_chunks.extend(page_chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {pdf_data['file_name']}")
        return all_chunks
