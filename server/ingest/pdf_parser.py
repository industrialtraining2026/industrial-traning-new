import PyPDF2
import os
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PDFParser:
    def __init__(self):
        self.supported_extensions = ['.pdf']
    
    def extract_text_from_pdf(self, file_path: str) -> Dict[str, Any]:
        """Extract text, metadata, and structure from PDF"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                pages_data = []
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    
                    # Extract text
                    text = page.extract_text()
                    
                    # Extract metadata
                    page_info = {
                        'page_number': page_num + 1,
                        'text': text,
                        'char_count': len(text),
                        'has_text': len(text.strip()) > 0,
                        'image_count': 0,  # PyPDF2 doesn't easily provide image count
                        'annotation_count': 0  # PyPDF2 doesn't easily provide annotation count
                    }
                    
                    pages_data.append(page_info)
                
                # Document metadata
                metadata = pdf_reader.metadata or {}
                
                return {
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'total_pages': len(pages_data),
                    'pages': pages_data,
                    'metadata': metadata,
                    'needs_ocr': any(not page['has_text'] for page in pages_data)
                }
                
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {str(e)}")
            return {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'error': str(e),
                'pages': []
            }
    
    def is_supported(self, file_path: str) -> bool:
        """Check if file is supported"""
        return any(file_path.lower().endswith(ext) for ext in self.supported_extensions)
