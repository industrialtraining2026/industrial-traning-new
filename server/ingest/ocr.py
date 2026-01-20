import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import logging
from typing import Dict, Any
import os
import tempfile

logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self):
        # Try to find tesseract executable
        self.ocr_available = False
        try:
            pytesseract.get_tesseract_version()
            self.ocr_available = True
            logger.info("Tesseract OCR is available")
        except Exception:
            # Common Windows paths
            possible_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    try:
                        pytesseract.get_tesseract_version()
                        self.ocr_available = True
                        logger.info(f"Tesseract OCR found at: {path}")
                        break
                    except:
                        continue
            
            if not self.ocr_available:
                logger.warning("Tesseract OCR not found. OCR functionality will be disabled.")
    
    def extract_text_with_ocr(self, pdf_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply OCR to pages that need it"""
        if not self.ocr_available:
            logger.info("OCR is not available - returning original PDF data")
            for page_data in pdf_data['pages']:
                page_data['ocr_applied'] = False
            return {
                **pdf_data,
                'pages': pdf_data['pages'],
                'ocr_processed': False,
                'ocr_disabled': True
            }
        
        try:
            enhanced_pages = []
            
            for page_data in pdf_data['pages']:
                page_num = page_data['page_number'] - 1
                
                # If page has little text, try OCR
                if page_data['char_count'] < 100 or not page_data['has_text']:
                    try:
                        logger.info(f"Applying OCR to page {page_num + 1}")
                        
                        # Convert PDF page to image
                        images = convert_from_path(
                            pdf_data['file_path'], 
                            first_page=page_num + 1, 
                            last_page=page_num + 1,
                            dpi=200  # Higher DPI for better OCR
                        )
                        
                        if images:
                            image = images[0]
                            
                            # OCR with multiple languages
                            ocr_text = pytesseract.image_to_string(
                                image, 
                                lang='eng+chi_sim+msa',  # English + Chinese + Malay
                                config='--psm 6'  # Assume single text block
                            )
                            
                            # Use OCR text if it's longer
                            if len(ocr_text.strip()) > len(page_data['text'].strip()):
                                page_data['text'] = ocr_text
                                page_data['ocr_applied'] = True
                                page_data['char_count'] = len(ocr_text)
                                page_data['has_text'] = len(ocr_text.strip()) > 0
                                logger.info(f"OCR improved page {page_num + 1}: {len(ocr_text)} chars")
                            else:
                                page_data['ocr_applied'] = False
                        else:
                            page_data['ocr_applied'] = False
                            
                    except Exception as e:
                        logger.warning(f"OCR failed for page {page_num + 1}: {str(e)}")
                        page_data['ocr_applied'] = False
                else:
                    page_data['ocr_applied'] = False
                
                enhanced_pages.append(page_data)
            
            return {
                **pdf_data,
                'pages': enhanced_pages,
                'ocr_processed': True
            }
            
        except Exception as e:
            logger.error(f"OCR processing failed for {pdf_data['file_path']}: {str(e)}")
            return {
                **pdf_data,
                'ocr_error': str(e),
                'ocr_processed': False
            }
