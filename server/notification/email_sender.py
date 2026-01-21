"""
Email Sender for sending notification emails to students (Brevo SMTP).
"""

import smtplib
import os
import logging
import traceback
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

        logger.info("=== EmailSender Init ===")
        logger.info(f"SMTP Host: {self.smtp_host}")
        logger.info(f"SMTP Port: {self.smtp_port}")
        logger.info(f"SMTP Username: {self.smtp_username}")
        logger.info(f"From Email: {self.from_email}")
        logger.info(f"From Name: {self.from_name}")
        logger.info(f"SMTP Password set: {'YES' if self.smtp_password else 'NO'}")

        if not self.from_email:
            raise RuntimeError("FROM_EMAIL is not set")

        if not self.smtp_password:
            raise RuntimeError("SMTP_PASSWORD is not set")

    # ==========================================================
    # Public API
    # ==========================================================
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

        if not to_emails:
            raise ValueError("No recipient emails provided")

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

        result = {
            "success": True,
            "total": len(to_emails),
            "sent": 0,
            "failed": 0,
            "errors": []
        }

        for email in to_emails:
            try:
                self._send_single_email(
                    to_email=email,
                    subject=subject,
                    html_body=html_body,
                    text_body=text_body
                )
                result["sent"] += 1
            except Exception as e:
                logger.error(f"❌ Failed sending to {email}")
                logger.error(str(e))
                logger.error(traceback.format_exc())
                result["failed"] += 1
                result["errors"].append(f"{email}: {str(e)}")

        result["success"] = result["failed"] == 0
        return result

    # ==========================================================
    # SMTP Core
    # ==========================================================
    def _send_single_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str
    ):
        logger.info("=== Sending Email ===")
        logger.info(f"To: {to_email}")
        logger.info(f"From: {self.from_email}")
        logger.info(f"Subject: {subject}")

        msg = MIMEMultipart("alternative")
        msg["From"] = f"{self.from_name} <{self.from_email}>"
        msg["To"] = to_email
        msg["Subject"] = subject

        msg.attach(MIMEText(text_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as server:
                server.set_debuglevel(1)  # 🔥 SMTP 对话直接打到 Render Logs
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            logger.info(f"✅ Email sent OK to {to_email}")

        except Exception:
            logger.error("🔥 SMTP send failed")
            logger.error(traceback.format_exc())
            raise

    # ==========================================================
    # Helpers
    # ==========================================================
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

        deadline_str = self._format_deadline(deadline_date, deadline_time)

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

        deadline_str = self._format_deadline(deadline_date, deadline_time)

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

    def _format_deadline(self, date_str: str, time_str: Optional[str]) -> str:
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            result = date_obj.strftime("%B %d, %Y")
        except ValueError:
            result = date_str

        if time_str:
            result += f", {time_str}"

        return result
