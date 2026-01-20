"""
Deadline Parser for extracting deadline information from PDF documents.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from dateutil import parser as date_parser

from ..ingest.pdf_parser import PDFParser
from ..ingest.ocr import OCRProcessor
from ..config import settings

logger = logging.getLogger(__name__)


class DeadlineParser:
    """Parse deadline information from PDF documents"""
    
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.ocr_processor = OCRProcessor()
        self.notification_pdf_dir = Path(settings.DATA_FOLDER) / "pdf_notification"
    
    def parse_deadline_pdf(self, pdf_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse deadline information from a PDF file
        
        Args:
            pdf_path: Path to PDF file. If None, uses the latest PDF from notification directory
            
        Returns:
            Dictionary with parsed deadline information
        """
        try:
            # If no path provided, get latest PDF from notification directory
            if pdf_path is None:
                pdf_path = self._get_latest_notification_pdf()
                if not pdf_path:
                    return {
                        "error": "No notification PDF found",
                        "deadline": None
                    }
            
            if not os.path.exists(pdf_path):
                return {
                    "error": "PDF file not found",
                    "deadline": None
                }
            
            # Extract text from PDF
            pdf_data = self.pdf_parser.extract_text_from_pdf(pdf_path)
            
            if 'error' in pdf_data:
                return {
                    "error": pdf_data['error'],
                    "deadline": None
                }
            
            # Apply OCR if needed
            if pdf_data.get('needs_ocr', False):
                logger.info("Applying OCR to notification PDF")
                pdf_data = self.ocr_processor.extract_text_with_ocr(pdf_data)
            
            # Combine all text from pages
            full_text = ""
            for page in pdf_data.get('pages', []):
                full_text += page.get('text', '') + "\n"
            
            # Parse information
            deadline_info = self._extract_deadline_info(full_text)
            
            return {
                "success": True,
                "file_path": pdf_path,
                "file_name": os.path.basename(pdf_path),
                **deadline_info
            }
            
        except Exception as e:
            logger.error(f"Error parsing deadline PDF: {str(e)}")
            return {
                "error": str(e),
                "deadline": None
            }
    
    def _get_latest_notification_pdf(self) -> Optional[str]:
        """Get the latest PDF file from notification directory"""
        if not self.notification_pdf_dir.exists():
            return None
        
        pdf_files = list(self.notification_pdf_dir.glob("*.pdf"))
        if not pdf_files:
            return None
        
        # Get the most recently modified file
        latest_file = max(pdf_files, key=lambda p: p.stat().st_mtime)
        return str(latest_file)
    
    def _extract_deadline_info(self, text: str) -> Dict[str, Any]:
        """Extract deadline information from text using NLP and pattern matching"""
        text_lower = text.lower()
        
        # Extract deadline date
        deadline_date = self._extract_date(text, text_lower)
        
        # Extract deadline time
        deadline_time = self._extract_time(text, text_lower)
        
        # Extract location
        location = self._extract_location(text, text_lower)
        
        # Extract submission items
        submission_items = self._extract_submission_items(text, text_lower)
        
        # Extract submission method
        submission_method = self._extract_submission_method(text, text_lower)
        
        # Extract additional info
        additional_info = self._extract_additional_info(text)
        
        return {
            "deadline": deadline_date,
            "deadline_time": deadline_time,
            "location": location,
            "submission_items": submission_items,
            "submission_method": submission_method,
            "additional_info": additional_info,
            "raw_text": text[:1000]  # First 1000 chars for reference
        }
    
    def _extract_date(self, text: str, text_lower: str) -> Optional[str]:
        """Extract deadline date from text"""
        # Common date patterns
        date_patterns = [
            r'deadline[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'due date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'submit by[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'submission deadline[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4})',
            r'((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4})',
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                try:
                    # Try to parse the date
                    date_str = matches[0]
                    parsed_date = date_parser.parse(date_str, fuzzy=True)
                    return parsed_date.strftime("%Y-%m-%d")
                except:
                    continue
        
        # Try to find any date-like pattern
        date_like_pattern = r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b'
        matches = re.findall(date_like_pattern, text)
        if matches:
            for match in matches:
                try:
                    parsed_date = date_parser.parse(match, fuzzy=True)
                    # Check if date is in the future (reasonable deadline)
                    if parsed_date.year >= datetime.now().year:
                        return parsed_date.strftime("%Y-%m-%d")
                except:
                    continue
        
        return None
    
    def _extract_time(self, text: str, text_lower: str) -> Optional[str]:
        """Extract deadline time from text"""
        time_patterns = [
            r'(\d{1,2}:\d{2}\s*(?:am|pm))',
            r'(\d{1,2}\s*(?:am|pm))',
            r'by\s+(\d{1,2}:\d{2})',
            r'before\s+(\d{1,2}:\d{2})',
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                return matches[0].upper()
        
        return None
    
    def _extract_location(self, text: str, text_lower: str) -> Optional[str]:
        """Extract submission location from text"""
        location_keywords = [
            'location', 'venue', 'address', 'submit to', 'office', 'room',
            'faculty', 'department', 'building'
        ]
        
        # Find sentences containing location keywords
        sentences = re.split(r'[.!?]\s+', text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in location_keywords):
                # Extract the sentence (limit length)
                location = sentence.strip()
                if len(location) > 200:
                    location = location[:200] + "..."
                return location
        
        return None
    
    def _extract_submission_items(self, text: str, text_lower: str) -> list:
        """Extract list of items to submit"""
        items = []
        
        # Look for bullet points or numbered lists
        patterns = [
            r'[-•]\s*([^\n]+)',
            r'\d+[\.\)]\s*([^\n]+)',
            r'submit[:\s]+([^\n]+)',
            r'required[:\s]+([^\n]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                match = match.strip()
                # Filter out very short or very long items
                if 5 < len(match) < 200:
                    # Check if it's a submission-related item
                    if any(keyword in match for keyword in ['form', 'cv', 'document', 'file', 'report', 'letter']):
                        items.append(match)
        
        # Remove duplicates
        return list(set(items))[:10]  # Limit to 10 items
    
    def _extract_submission_method(self, text: str, text_lower: str) -> Optional[str]:
        """Extract submission method from text"""
        method_keywords = ['email', 'submit to', 'send to', 'deliver to', 'online', 'portal']
        
        sentences = re.split(r'[.!?]\s+', text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in method_keywords):
                method = sentence.strip()
                if len(method) > 200:
                    method = method[:200] + "..."
                return method
        
        return None
    
    def _extract_additional_info(self, text: str) -> Optional[str]:
        """Extract any additional relevant information"""
        # Get a summary (first 500 characters)
        if len(text) > 500:
            return text[:500] + "..."
        return text if text else None

