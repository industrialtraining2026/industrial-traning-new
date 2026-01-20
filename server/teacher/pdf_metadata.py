"""
PDF Metadata Manager for tracking PDF file information.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..config import settings

logger = logging.getLogger(__name__)


class PDFMetadataManager:
    """Manage PDF metadata (upload time, uploader, etc.)"""
    
    def __init__(self):
        self.metadata_file = Path(settings.DATA_FOLDER) / "pdf_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from JSON file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {str(e)}")
                return {}
        return {
            "chatbot_pdfs": [],
            "submission_pdfs": [],
            "notification_pdfs": []
        }
    
    def _save_metadata(self):
        """Save metadata to JSON file"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
    
    def add_pdf_metadata(self, filename: str, pdf_type: str, file_size: int, uploaded_by: str) -> bool:
        """
        Add metadata for a PDF file
        
        Args:
            filename: Name of the PDF file
            pdf_type: Type of PDF (chatbot, submission, notification)
            file_size: Size of the file in bytes
            uploaded_by: User ID who uploaded the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            type_key = f"{pdf_type}_pdfs"
            if type_key not in self.metadata:
                self.metadata[type_key] = []
            
            # Check if metadata already exists
            existing = next(
                (item for item in self.metadata[type_key] if item.get("file_name") == filename),
                None
            )
            
            if existing:
                # Update existing metadata
                existing.update({
                    "file_size": file_size,
                    "upload_time": datetime.now().isoformat(),
                    "uploaded_by": uploaded_by,
                    "upload_status": "success",
                    "last_action": "upload",
                    "last_updated": datetime.now().isoformat()
                })
            else:
                # Add new metadata
                self.metadata[type_key].append({
                    "file_name": filename,
                    "file_size": file_size,
                    "upload_time": datetime.now().isoformat(),
                    "uploaded_by": uploaded_by,
                    "pdf_type": pdf_type,
                    "upload_status": "success",
                    "rebuild_status": None,
                    "delete_status": None,
                    "last_action": "upload",
                    "last_updated": datetime.now().isoformat()
                })
            
            self._save_metadata()
            return True
            
        except Exception as e:
            logger.error(f"Error adding PDF metadata: {str(e)}")
            return False
    
    def remove_pdf_metadata(self, filename: str, pdf_type: str) -> bool:
        """
        Remove metadata for a PDF file
        
        Args:
            filename: Name of the PDF file
            pdf_type: Type of PDF (chatbot, submission, notification)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            type_key = f"{pdf_type}_pdfs"
            if type_key not in self.metadata:
                return False
            
            # Remove metadata
            self.metadata[type_key] = [
                item for item in self.metadata[type_key]
                if item.get("file_name") != filename
            ]
            
            self._save_metadata()
            return True
            
        except Exception as e:
            logger.error(f"Error removing PDF metadata: {str(e)}")
            return False
    
    def get_pdf_metadata(self, filename: str, pdf_type: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific PDF file
        
        Args:
            filename: Name of the PDF file
            pdf_type: Type of PDF (chatbot, submission, notification)
            
        Returns:
            Metadata dictionary or None if not found
        """
        try:
            type_key = f"{pdf_type}_pdfs"
            if type_key not in self.metadata:
                return None
            
            return next(
                (item for item in self.metadata[type_key] if item.get("file_name") == filename),
                None
            )
            
        except Exception as e:
            logger.error(f"Error getting PDF metadata: {str(e)}")
            return None
    
    def list_pdf_metadata(self, pdf_type: str) -> List[Dict[str, Any]]:
        """
        List all metadata for a PDF type
        
        Args:
            pdf_type: Type of PDF (chatbot, submission, notification)
            
        Returns:
            List of metadata dictionaries
        """
        try:
            type_key = f"{pdf_type}_pdfs"
            return self.metadata.get(type_key, [])
            
        except Exception as e:
            logger.error(f"Error listing PDF metadata: {str(e)}")
            return []
    
    def update_pdf_status(self, filename: str, pdf_type: str, status_type: str, status: str) -> bool:
        """
        Update status for a PDF file (upload_status, rebuild_status, delete_status)
        
        Args:
            filename: Name of the PDF file
            pdf_type: Type of PDF (chatbot, submission, notification)
            status_type: Type of status to update (upload_status, rebuild_status, delete_status)
            status: Status value (success, failed)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            type_key = f"{pdf_type}_pdfs"
            if type_key not in self.metadata:
                return False
            
            pdf_item = next(
                (item for item in self.metadata[type_key] if item.get("file_name") == filename),
                None
            )
            
            if pdf_item:
                pdf_item[status_type] = status
                pdf_item["last_updated"] = datetime.now().isoformat()
                # Store last action type
                if status_type == "upload_status":
                    pdf_item["last_action"] = "upload"
                elif status_type == "delete_status":
                    pdf_item["last_action"] = "delete"
                elif status_type == "rebuild_status":
                    pdf_item["last_action"] = "rebuild"
                self._save_metadata()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating PDF status: {str(e)}")
            return False

