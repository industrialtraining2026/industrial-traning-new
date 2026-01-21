"""
Email Sender for sending notification emails to students (Brevo SMTP).
"""

import smtplib
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


class EmailSender:
    """Send notification emails using Brevo SMTP"""

    def __init__(self):
        # Brevo SMTP configuration
        self.smtp_host = os.getenv("SMTP_HOST", "smtp-relay.brevo.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "apikey")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", "")
        self.from_name = os.getenv("FROM_NAME", "Industrial Training Office")

        logger.info("Email Sender initialized (Brevo)")
        logger.info(f"SMTP Host: {self.smtp_host}")
        logger.info(f"SMTP Port: {self.smtp_port}")
        logger.info(f"From Email: {self.from_email}")
        logger.info(f"From Name: {self.from_name}")
        logger.info(f"Password configured: {'Yes' if self.smtp_password else 'No'}")

    def send_notification(
        self,
        to_emails: List[str],
        deadline_date: str,
        deadline_time: Optional[str] = None,
        location: Optional[str] = None,
        submission_items: Optional[List[str]] = None,
        submission_method: Optional[str] = None,
        additional_info: Optional[str] = None,
        reminder_type: str = "general"
    ) -> Dict[str, Any]:

        if not self.smtp_password:
            return {
                "success": False,
                "error": "Brevo SMTP key not configured"
            }

        if not to_emails:
            return {
                "success": False,
                "error": "No recipient emails provided"
            }

        subject = self._generate_subject(reminder_type)
        html_body = self._generate_html_body(
            deadline_date,
            deadline_time,
            location,
            submission_items,
            submission_method,
            additional_info
        )
        text_body = self._generate_text_body(
            deadline_date,
            deadline_time,
            location,
            submission_items,
            submission_method,
            additional_info
        )

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
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"{email}: {str(e)}")

        results["success"] = results["failed"] == 0
        return results

    def _send_single_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str
    ):
        msg = MIMEMultipart("alternative")
        msg["From"] = f"{self.from_name} <{self.from_email}>"
        msg["To"] = to_email
        msg["Subject"] = subject

        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as server:
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)

    def _generate_subject(self, reminder_type: str) -> str:
        if reminder_type == "one_week":
            return "[Important] Industrial Training Reminder – 1 Week Left"
        elif reminder_type == "three_days":
            return "[Urgent] Industrial Training Reminder – 3 Days Left"
        return "[Important] Industrial Training Submission Reminder"

    def _generate_html_body(
        self,
        deadline_date: str,
        deadline_time: Optional[str],
        location: Optional[str],
        submission_items: Optional[List[str]],
        submission_method: Optional[str],
        additional_info: Optional[str]
    ) -> str:

        try:
            date_obj = datetime.strptime(deadline_date, "%Y-%m-%d")
            deadline_str = date_obj.strftime("%B %d, %Y")
        except ValueError:
            deadline_str = deadline_date

        if deadline_time:
            deadline_str += f", {deadline_time}"

        items_html = "<ul>"
        if submission_items:
            for item in submission_items:
                items_html += f"<li>{item}</li>"
        else:
            items_html += "<li>Please refer to official notice</li>"
        items_html += "</ul>"

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2>Industrial Training Submission Reminder</h2>
            <p>Dear Student,</p>

            <p>This is a reminder regarding your Industrial Training submission.</p>

            <p><strong>Deadline:</strong> {deadline_str}</p>
            {f"<p><strong>Location:</strong> {location}</p>" if location else ""}

            <h4>Required Documents:</h4>
            {items_html}

            {f"<p><strong>Submission Method:</strong> {submission_method}</p>" if submission_method else ""}
            {f"<p><strong>Additional Information:</strong><br>{additional_info}</p>" if additional_info else ""}

            <p>Best regards,<br><strong>{self.from_name}</strong></p>
        </body>
        </html>
        """

    def _generate_text_body(
        self,
        deadline_date: str,
        deadline_time: Optional[str],
        location: Optional[str],
        submission_items: Optional[List[str]],
        submission_method: Optional[str],
        additional_info: Optional[str]
    ) -> str:

        try:
            date_obj = datetime.strptime(deadline_date, "%Y-%m-%d")
            deadline_str = date_obj.strftime("%B %d, %Y")
        except ValueError:
            deadline_str = deadline_date

        if deadline_time:
            deadline_str += f", {deadline_time}"

        text = f"""Industrial Training Submission Reminder

Deadline: {deadline_str}
"""

        if location:
            text += f"Location: {location}\n"

        text += "\nRequired Documents:\n"
        if submission_items:
            for item in submission_items:
                text += f"- {item}\n"

        if submission_method:
            text += f"\nSubmission Method: {submission_method}\n"

        if additional_info:
            text += f"\nAdditional Info:\n{additional_info}\n"

        text += f"\nBest regards,\n{self.from_name}"
        return text
