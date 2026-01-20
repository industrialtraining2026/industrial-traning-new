"""
Email Sender for sending notification emails to students.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)


class EmailSender:
    """Send notification emails to students"""
    
    def __init__(self):
        # Email configuration (can be set via environment variables)
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", self.smtp_username)
        self.from_name = os.getenv("FROM_NAME", "Industrial Training Office")
    
    def send_notification(
        self,
        to_emails: List[str],
        deadline_date: str,
        deadline_time: Optional[str] = None,
        location: Optional[str] = None,
        submission_items: Optional[List[str]] = None,
        submission_method: Optional[str] = None,
        additional_info: Optional[str] = None,
        reminder_type: str = "general"  # "one_week" or "three_days" or "general"
    ) -> Dict[str, Any]:
        """
        Send notification email to students
        
        Args:
            to_emails: List of recipient email addresses
            deadline_date: Deadline date (YYYY-MM-DD format)
            deadline_time: Deadline time (optional)
            location: Submission location (optional)
            submission_items: List of items to submit (optional)
            submission_method: How to submit (optional)
            additional_info: Additional information (optional)
            reminder_type: Type of reminder
            
        Returns:
            Dictionary with sending results
        """
        if not self.smtp_username or not self.smtp_password:
            return {
                "success": False,
                "error": "SMTP credentials not configured. Please set SMTP_USERNAME and SMTP_PASSWORD environment variables."
            }
        
        if not to_emails:
            return {
                "success": False,
                "error": "No recipient emails provided"
            }
        
        # Generate email content
        subject = self._generate_subject(deadline_date, reminder_type)
        html_body = self._generate_html_body(
            deadline_date, deadline_time, location,
            submission_items, submission_method, additional_info, reminder_type
        )
        text_body = self._generate_text_body(
            deadline_date, deadline_time, location,
            submission_items, submission_method, additional_info, reminder_type
        )
        
        # Send emails
        results = {
            "success": True,
            "total_recipients": len(to_emails),
            "sent": 0,
            "failed": 0,
            "errors": []
        }
        
        for email in to_emails:
            try:
                self._send_single_email(email, subject, html_body, text_body)
                results["sent"] += 1
                logger.info(f"Sent notification email to {email}")
            except Exception as e:
                results["failed"] += 1
                error_msg = f"{email}: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(f"Failed to send email to {email}: {str(e)}")
        
        results["success"] = results["failed"] == 0
        
        return results
    
    def _send_single_email(self, to_email: str, subject: str, html_body: str, text_body: str):
        """Send a single email"""
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = f"{self.from_name} <{self.from_email}>"
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # Add both plain text and HTML versions
        part1 = MIMEText(text_body, 'plain')
        part2 = MIMEText(html_body, 'html')
        
        msg.attach(part1)
        msg.attach(part2)
        
        # Send email
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
    
    def _generate_subject(self, deadline_date: str, reminder_type: str) -> str:
        """Generate email subject"""
        if reminder_type == "one_week":
            return f"[Important] Industrial Training Submission Reminder - 1 Week Before Deadline"
        elif reminder_type == "three_days":
            return f"[Urgent] Industrial Training Submission Reminder - 3 Days Before Deadline"
        else:
            return f"[Important] Industrial Training Submission Reminder"
    
    def _generate_html_body(
        self,
        deadline_date: str,
        deadline_time: Optional[str],
        location: Optional[str],
        submission_items: Optional[List[str]],
        submission_method: Optional[str],
        additional_info: Optional[str],
        reminder_type: str
    ) -> str:
        """Generate HTML email body"""
        # Format date
        try:
            date_obj = datetime.strptime(deadline_date, "%Y-%m-%d")
            formatted_date = date_obj.strftime("%B %d, %Y")
        except:
            formatted_date = deadline_date
        
        deadline_str = formatted_date
        if deadline_time:
            deadline_str += f", {deadline_time}"
        
        # Generate items list
        items_html = ""
        if submission_items:
            items_html = "<ul>"
            for item in submission_items:
                items_html += f"<li>{item}</li>"
            items_html += "</ul>"
        else:
            items_html = "<p>Please refer to the notification PDF for details.</p>"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
        </head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #111827;">Industrial Training Submission Reminder</h2>
                
                <p>Dear Student,</p>
                
                <p>This is a reminder regarding your Industrial Training submission.</p>
                
                <div style="background: #f9fafb; padding: 15px; border-radius: 8px; margin: 20px 0;">
                    <p><strong>Submission Deadline:</strong> {deadline_str}</p>
                    {f'<p><strong>Submission Location:</strong> {location}</p>' if location else ''}
                </div>
                
                <h3 style="color: #111827;">Required Documents:</h3>
                {items_html}
                
                {f'<p><strong>Submission Method:</strong> {submission_method}</p>' if submission_method else ''}
                
                {f'<div style="margin-top: 20px; padding: 15px; background: #fef3c7; border-radius: 8px;"><p><strong>Additional Information:</strong></p><p>{additional_info}</p></div>' if additional_info else ''}
                
                <p style="margin-top: 30px;">Please ensure all documents are submitted before the deadline.</p>
                
                <p>Best regards,<br>
                <strong>{self.from_name}</strong></p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_text_body(
        self,
        deadline_date: str,
        deadline_time: Optional[str],
        location: Optional[str],
        submission_items: Optional[List[str]],
        submission_method: Optional[str],
        additional_info: Optional[str],
        reminder_type: str
    ) -> str:
        """Generate plain text email body"""
        # Format date
        try:
            date_obj = datetime.strptime(deadline_date, "%Y-%m-%d")
            formatted_date = date_obj.strftime("%B %d, %Y")
        except:
            formatted_date = deadline_date
        
        deadline_str = formatted_date
        if deadline_time:
            deadline_str += f", {deadline_time}"
        
        text = f"""Industrial Training Submission Reminder

Dear Student,

This is a reminder regarding your Industrial Training submission.

Submission Deadline: {deadline_str}
"""
        
        if location:
            text += f"Submission Location: {location}\n"
        
        text += "\nRequired Documents:\n"
        if submission_items:
            for item in submission_items:
                text += f"- {item}\n"
        else:
            text += "Please refer to the notification PDF for details.\n"
        
        if submission_method:
            text += f"\nSubmission Method: {submission_method}\n"
        
        if additional_info:
            text += f"\nAdditional Information:\n{additional_info}\n"
        
        text += "\nPlease ensure all documents are submitted before the deadline.\n\n"
        text += f"Best regards,\n{self.from_name}"
        
        return text

