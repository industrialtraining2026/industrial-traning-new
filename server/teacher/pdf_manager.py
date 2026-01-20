"""
PDF Manager for teacher to upload, delete, and list PDFs.
Supports three types: chatbot, submission, notification
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from ..config import settings

logger = logging.getLogger(__name__)


class PDFManager:
    """Manage PDF files for chatbot, submission, and notification"""
    
    def __init__(self):
        self.pdf_chatbot_dir = Path(settings.DATA_FOLDER) / "pdf_chatbot"
        self.pdf_submission_dir = Path(settings.DATA_FOLDER) / "pdf_submission"
        self.pdf_notification_dir = Path(settings.DATA_FOLDER) / "pdf_notification"
        
        # Create directories if they don't exist
        self.pdf_chatbot_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_submission_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_notification_dir.mkdir(parents=True, exist_ok=True)
    
    def get_directory(self, pdf_type: str) -> Path:
        """Get directory path for PDF type"""
        type_map = {
            "chatbot": self.pdf_chatbot_dir,
            "submission": self.pdf_submission_dir,
            "notification": self.pdf_notification_dir
        }
        
        if pdf_type not in type_map:
            raise ValueError(f"Invalid PDF type: {pdf_type}. Must be one of: chatbot, submission, notification")
        
        return type_map[pdf_type]
    
    def upload_pdf(self, file_content: bytes, filename: str, pdf_type: str, uploaded_by: str = "teacher") -> Dict[str, Any]:
        """
        Upload a PDF file to the specified directory
        
        Args:
            file_content: PDF file content as bytes
            filename: Original filename
            pdf_type: Type of PDF (chatbot, submission, notification)
            uploaded_by: User ID who uploaded the file
            
        Returns:
            Dictionary with file information
        """
        try:
            # Validate file type
            if not filename.lower().endswith('.pdf'):
                raise ValueError("Only PDF files are allowed")
            
            # Validate PDF type
            target_dir = self.get_directory(pdf_type)
            
            # Sanitize filename (prevent path traversal)
            safe_filename = os.path.basename(filename)
            if not safe_filename or safe_filename == '.' or safe_filename == '..':
                raise ValueError("Invalid filename")
            
            # Create full file path
            file_path = target_dir / safe_filename
            
            # Check if file already exists
            if file_path.exists():
                # Add timestamp to filename to avoid overwriting
                name_part = file_path.stem
                ext_part = file_path.suffix
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_filename = f"{name_part}_{timestamp}{ext_part}"
                file_path = target_dir / safe_filename
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Get file info
            file_size = os.path.getsize(file_path)
            upload_time = datetime.now().isoformat()
            
            logger.info(f"Uploaded PDF: {safe_filename} to {pdf_type} directory")
            
            return {
                "success": True,
                "file_name": safe_filename,
                "file_path": str(file_path),
                "file_size": file_size,
                "upload_time": upload_time,
                "pdf_type": pdf_type,
                "uploaded_by": uploaded_by
            }
            
        except Exception as e:
            logger.error(f"Error uploading PDF: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def delete_pdf(self, filename: str, pdf_type: str) -> Dict[str, Any]:
        """
        Delete a PDF file
        
        Args:
            filename: Name of the file to delete
            pdf_type: Type of PDF (chatbot, submission, notification)
            
        Returns:
            Dictionary with deletion result
        """
        try:
            target_dir = self.get_directory(pdf_type)
            
            # Sanitize filename
            safe_filename = os.path.basename(filename)
            file_path = target_dir / safe_filename
            
            if not file_path.exists():
                return {
                    "success": False,
                    "error": "File not found"
                }
            
            # Delete file
            file_path.unlink()
            
            logger.info(f"Deleted PDF: {safe_filename} from {pdf_type} directory")
            
            return {
                "success": True,
                "file_name": safe_filename,
                "message": "File deleted successfully"
            }
            
        except Exception as e:
            logger.error(f"Error deleting PDF: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_pdfs(self, pdf_type: str) -> List[Dict[str, Any]]:
        """
        List all PDFs in the specified directory
        
        Args:
            pdf_type: Type of PDF (chatbot, submission, notification)
            
        Returns:
            List of PDF file information
        """
        try:
            target_dir = self.get_directory(pdf_type)
            
            pdf_files = []
            
            # Get all PDF files in directory
            for file_path in target_dir.glob("*.pdf"):
                try:
                    file_stat = file_path.stat()
                    
                    pdf_files.append({
                        "file_name": file_path.name,
                        "file_size": file_stat.st_size,
                        "upload_time": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                        "modified_time": datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    })
                except Exception as e:
                    logger.warning(f"Error getting info for {file_path.name}: {str(e)}")
                    continue
            
            # Sort by modified time (newest first)
            pdf_files.sort(key=lambda x: x["modified_time"], reverse=True)
            
            return pdf_files
            
        except Exception as e:
            logger.error(f"Error listing PDFs: {str(e)}")
            return []
    
    def get_pdf_info(self, filename: str, pdf_type: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific PDF file
        
        Args:
            filename: Name of the file
            pdf_type: Type of PDF (chatbot, submission, notification)
            
        Returns:
            Dictionary with file information or None if not found
        """
        try:
            target_dir = self.get_directory(pdf_type)
            safe_filename = os.path.basename(filename)
            file_path = target_dir / safe_filename
            
            if not file_path.exists():
                return None
            
            file_stat = file_path.stat()
            
            return {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_size": file_stat.st_size,
                "upload_time": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                "pdf_type": pdf_type
            }
            
        except Exception as e:
            logger.error(f"Error getting PDF info: {str(e)}")
            return None

