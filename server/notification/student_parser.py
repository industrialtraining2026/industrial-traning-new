"""
Student Email Parser for parsing and managing student email lists.
"""

import csv
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import logging

from ..config import settings

logger = logging.getLogger(__name__)


class StudentEmailParser:
    """Parse and manage student email lists"""
    
    def __init__(self):
        self.students_file = Path(settings.DATA_FOLDER) / "notifications" / "student_emails.json"
        self.students_file.parent.mkdir(parents=True, exist_ok=True)
        self.uploaded_files_dir = Path(settings.DATA_FOLDER) / "notifications" / "uploaded_email_files"
        self.uploaded_files_dir.mkdir(parents=True, exist_ok=True)
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """
        Save uploaded email file
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Saved file path
        """
        import os
        
        # Sanitize filename
        safe_filename = os.path.basename(filename)
        if not safe_filename or safe_filename == '.' or safe_filename == '..':
            raise ValueError("Invalid filename")
        
        # Add timestamp if file exists
        file_path = self.uploaded_files_dir / safe_filename
        if file_path.exists():
            name_part = file_path.stem
            ext_part = file_path.suffix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{name_part}_{timestamp}{ext_part}"
            file_path = self.uploaded_files_dir / safe_filename
        
        # Save file
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        logger.info(f"Saved uploaded email file: {safe_filename}")
        return safe_filename
    
    def parse_email_file(self, file_content: bytes, filename: str, save_file: bool = True) -> Dict[str, Any]:
        """
        Parse email list from uploaded file (CSV or TXT)
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            save_file: Whether to save the uploaded file
            
        Returns:
            Dictionary with parsed student emails
        """
        try:
            # Save uploaded file
            saved_filename = None
            if save_file:
                saved_filename = self.save_uploaded_file(file_content, filename)
            
            # Decode file content
            try:
                text_content = file_content.decode('utf-8')
            except:
                text_content = file_content.decode('latin-1')
            
            students = []
            
            # Check file extension
            if filename.lower().endswith('.csv'):
                students = self._parse_csv(text_content)
            elif filename.lower().endswith('.txt'):
                students = self._parse_txt(text_content)
            else:
                return {
                    "success": False,
                    "error": "Unsupported file format. Please upload CSV or TXT file."
                }
            
            # Validate emails
            valid_students = []
            invalid_emails = []
            
            for student in students:
                email = student.get('email', '').strip()
                if email and self._is_valid_email(email):
                    valid_students.append(student)
                elif email:
                    invalid_emails.append(email)
            
            # Save to file
            if valid_students:
                self.save_students(valid_students)
            
            return {
                "success": True,
                "total_parsed": len(students),
                "valid_emails": len(valid_students),
                "invalid_emails": len(invalid_emails),
                "invalid_email_list": invalid_emails[:10],  # First 10 invalid emails
                "students": valid_students,
                "saved_filename": saved_filename
            }
            
        except Exception as e:
            logger.error(f"Error parsing email file: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _parse_csv(self, content: str) -> List[Dict[str, Any]]:
        """Parse CSV file"""
        students = []
        
        # Try to detect delimiter
        sniffer = csv.Sniffer()
        try:
            delimiter = sniffer.sniff(content[:1024]).delimiter
        except:
            delimiter = ','
        
        # Parse CSV
        reader = csv.DictReader(content.splitlines(), delimiter=delimiter)
        
        for row in reader:
            # Try different column names
            email = (
                row.get('email') or 
                row.get('Email') or 
                row.get('EMAIL') or
                row.get('e-mail') or
                row.get('E-mail')
            )
            
            name = (
                row.get('name') or 
                row.get('Name') or 
                row.get('NAME') or
                row.get('student_name') or
                row.get('Student Name')
            )
            
            student_id = (
                row.get('student_id') or 
                row.get('Student ID') or 
                row.get('student_id') or
                row.get('id') or
                row.get('ID')
            )
            
            if email:
                students.append({
                    "email": email.strip(),
                    "name": name.strip() if name else "",
                    "student_id": student_id.strip() if student_id else ""
                })
        
        return students
    
    def _parse_txt(self, content: str) -> List[Dict[str, Any]]:
        """Parse TXT file (one email per line)"""
        students = []
        
        lines = content.splitlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            email = ""
            name = ""
            student_id = ""
            
            # Check if line contains "email:" or "Email:" pattern
            # Format: email: xxx@xxx.com or Email: xxx@xxx.com
            email_match = re.search(r'email\s*:\s*([^\s,]+)', line, re.IGNORECASE)
            if email_match:
                email = email_match.group(1).strip()
            else:
                # Try to extract email, name, and ID from line
                # Format: email, name, student_id (comma separated)
                # Or just email
                parts = [p.strip() for p in line.split(',')]
                email = parts[0] if parts else ""
                name = parts[1] if len(parts) > 1 else ""
                student_id = parts[2] if len(parts) > 2 else ""
            
            # Also check if email pattern exists anywhere in the line
            if not email or not self._is_valid_email(email):
                # Try to find email pattern in the line
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                email_matches = re.findall(email_pattern, line)
                if email_matches:
                    email = email_matches[0]
            
            if email:
                students.append({
                    "email": email,
                    "name": name,
                    "student_id": student_id
                })
        
        return students
    
    def _is_valid_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def save_students(self, students: List[Dict[str, Any]]) -> bool:
        """Save student list to JSON file"""
        try:
            data = {
                "students": students,
                "last_updated": datetime.now().isoformat(),
                "total_students": len(students)
            }
            
            with open(self.students_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(students)} student emails to {self.students_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving students: {str(e)}")
            return False
    
    def load_students(self) -> List[Dict[str, Any]]:
        """Load student list from JSON file"""
        try:
            if not self.students_file.exists():
                return []
            
            with open(self.students_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data.get("students", [])
            
        except Exception as e:
            logger.error(f"Error loading students: {str(e)}")
            return []
    
    def get_student_count(self) -> int:
        """Get total number of students"""
        students = self.load_students()
        return len(students)
    
    def get_student_emails(self) -> List[str]:
        """Get list of student emails only"""
        students = self.load_students()
        return [s.get('email', '') for s in students if s.get('email')]
    
    def list_uploaded_files(self) -> List[Dict[str, Any]]:
        """List all uploaded email files"""
        files = []
        try:
            for file_path in self.uploaded_files_dir.glob("*.*"):
                if file_path.is_file():
                    try:
                        file_stat = file_path.stat()
                        files.append({
                            "file_name": file_path.name,
                            "file_size": file_stat.st_size,
                            "upload_time": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                            "modified_time": datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                        })
                    except Exception as e:
                        logger.warning(f"Error getting info for {file_path.name}: {str(e)}")
                        continue
            
            # Sort by modified time (newest first)
            files.sort(key=lambda x: x["modified_time"], reverse=True)
            return files
        except Exception as e:
            logger.error(f"Error listing uploaded files: {str(e)}")
            return []
    
    def delete_uploaded_file(self, filename: str) -> Dict[str, Any]:
        """Delete an uploaded email file"""
        try:
            import os
            safe_filename = os.path.basename(filename)
            file_path = self.uploaded_files_dir / safe_filename
            
            if not file_path.exists():
                return {
                    "success": False,
                    "error": "File not found"
                }
            
            file_path.unlink()
            logger.info(f"Deleted uploaded email file: {safe_filename}")
            
            return {
                "success": True,
                "file_name": safe_filename,
                "message": "File deleted successfully"
            }
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_uploaded_file_path(self, filename: str):
        """Get path to uploaded file"""
        import os
        safe_filename = os.path.basename(filename)
        return self.uploaded_files_dir / safe_filename

