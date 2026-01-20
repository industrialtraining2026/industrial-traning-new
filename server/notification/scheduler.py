"""
Notification Scheduler for automated email notifications.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .deadline_parser import DeadlineParser
from .student_parser import StudentEmailParser
from .email_sender import EmailSender
from ..config import settings

logger = logging.getLogger(__name__)


class NotificationScheduler:
    """Schedule and manage automated email notifications"""
    
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.deadline_parser = DeadlineParser()
        self.student_parser = StudentEmailParser()
        self.email_sender = EmailSender()
        self.notification_log_file = Path(settings.DATA_FOLDER) / "notifications" / "notification_log.json"
        self.deadline_info_file = Path(settings.DATA_FOLDER) / "notifications" / "deadline_info.json"
        self.notification_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load notification history
        self.notification_history = self._load_notification_history()
    
    def start(self):
        """Start the scheduler"""
        if not self.scheduler.running:
            # Schedule daily check at 1 AM
            self.scheduler.add_job(
                self.check_and_send_notifications,
                trigger=CronTrigger(hour=1, minute=0),
                id='daily_notification_check',
                name='Daily Notification Check',
                replace_existing=True
            )
            self.scheduler.start()
            logger.info("Notification scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Notification scheduler stopped")
    
    def check_and_send_notifications(self):
        """Check deadlines and send notifications if needed"""
        try:
            # Load deadline info
            deadline_info = self._load_deadline_info()
            if not deadline_info or not deadline_info.get("deadline"):
                logger.info("No deadline information found")
                return
            
            deadline_date_str = deadline_info.get("deadline")
            if not deadline_date_str:
                return
            
            # Parse deadline date
            try:
                deadline_date = datetime.strptime(deadline_date_str, "%Y-%m-%d").date()
            except:
                logger.error(f"Invalid deadline date format: {deadline_date_str}")
                return
            
            today = datetime.now().date()
            
            # Calculate notification dates
            one_week_before = deadline_date - timedelta(days=7)
            three_days_before = deadline_date - timedelta(days=3)
            
            # Check if we need to send 1-week reminder
            if today == one_week_before:
                if not self._already_sent(deadline_date_str, "one_week"):
                    logger.info(f"Sending 1-week reminder for deadline {deadline_date_str}")
                    self._send_notification(deadline_info, "one_week")
                    self._log_notification(deadline_date_str, "one_week", "sent")
            
            # Check if we need to send 3-day reminder
            if today == three_days_before:
                if not self._already_sent(deadline_date_str, "three_days"):
                    logger.info(f"Sending 3-day reminder for deadline {deadline_date_str}")
                    self._send_notification(deadline_info, "three_days")
                    self._log_notification(deadline_date_str, "three_days", "sent")
            
        except Exception as e:
            logger.error(f"Error in notification check: {str(e)}")
    
    def _send_notification(self, deadline_info: Dict[str, Any], reminder_type: str):
        """Send notification email"""
        try:
            # Get student emails
            student_emails = self.student_parser.get_student_emails()
            if not student_emails:
                logger.warning("No student emails found")
                return
            
            # Send email
            result = self.email_sender.send_notification(
                to_emails=student_emails,
                deadline_date=deadline_info.get("deadline", ""),
                deadline_time=deadline_info.get("deadline_time"),
                location=deadline_info.get("location"),
                submission_items=deadline_info.get("submission_items"),
                submission_method=deadline_info.get("submission_method"),
                additional_info=deadline_info.get("additional_info"),
                reminder_type=reminder_type
            )
            
            if result.get("success"):
                logger.info(f"Notification sent successfully: {result['sent']} emails sent")
            else:
                logger.error(f"Notification sending failed: {result.get('error')}")
            
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
    
    def _already_sent(self, deadline_date: str, reminder_type: str) -> bool:
        """Check if notification was already sent"""
        for entry in self.notification_history:
            if (entry.get("deadline_date") == deadline_date and 
                entry.get("reminder_type") == reminder_type and
                entry.get("status") == "sent"):
                return True
        return False
    
    def _log_notification(self, deadline_date: str, reminder_type: str, status: str):
        """Log notification to history"""
        entry = {
            "deadline_date": deadline_date,
            "reminder_type": reminder_type,
            "status": status,
            "sent_time": datetime.now().isoformat(),
            "recipient_count": self.student_parser.get_student_count()
        }
        
        self.notification_history.append(entry)
        self._save_notification_history()
    
    def _load_deadline_info(self) -> Optional[Dict[str, Any]]:
        """Load deadline information from file"""
        try:
            if not self.deadline_info_file.exists():
                return None
            
            with open(self.deadline_info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error loading deadline info: {str(e)}")
            return None
    
    def save_deadline_info(self, deadline_info: Dict[str, Any]) -> bool:
        """Save deadline information to file"""
        try:
            with open(self.deadline_info_file, 'w', encoding='utf-8') as f:
                json.dump(deadline_info, f, indent=2, ensure_ascii=False)
            logger.info("Deadline info saved")
            return True
        except Exception as e:
            logger.error(f"Error saving deadline info: {str(e)}")
            return False
    
    def _load_notification_history(self) -> list:
        """Load notification history"""
        try:
            if not self.notification_log_file.exists():
                return []
            
            with open(self.notification_log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("history", [])
                
        except Exception as e:
            logger.error(f"Error loading notification history: {str(e)}")
            return []
    
    def _save_notification_history(self):
        """Save notification history"""
        try:
            data = {
                "history": self.notification_history,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.notification_log_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving notification history: {str(e)}")
    
    def get_notification_status(self) -> Dict[str, Any]:
        """Get current notification status"""
        deadline_info = self._load_deadline_info()
        
        if not deadline_info or not deadline_info.get("deadline"):
            return {
                "has_deadline": False,
                "message": "No deadline information configured"
            }
        
        deadline_date_str = deadline_info.get("deadline")
        try:
            deadline_date = datetime.strptime(deadline_date_str, "%Y-%m-%d").date()
        except:
            return {
                "has_deadline": False,
                "message": "Invalid deadline date format"
            }
        
        today = datetime.now().date()
        one_week_before = deadline_date - timedelta(days=7)
        three_days_before = deadline_date - timedelta(days=3)
        
        return {
            "has_deadline": True,
            "deadline": deadline_date_str,
            "deadline_formatted": deadline_date.strftime("%B %d, %Y"),
            "one_week_reminder": {
                "date": one_week_before.isoformat(),
                "sent": self._already_sent(deadline_date_str, "one_week"),
                "days_until": (one_week_before - today).days
            },
            "three_days_reminder": {
                "date": three_days_before.isoformat(),
                "sent": self._already_sent(deadline_date_str, "three_days"),
                "days_until": (three_days_before - today).days
            },
            "student_count": self.student_parser.get_student_count()
        }
    
    def get_notification_history(self, limit: int = 50) -> list:
        """Get notification history"""
        return self.notification_history[-limit:]
    
    def manual_send_notification(self, reminder_type: str = "general") -> Dict[str, Any]:
        """Manually trigger notification sending"""
        deadline_info = self._load_deadline_info()
        if not deadline_info:
            return {
                "success": False,
                "error": "No deadline information found"
            }
        
        # Get student emails
        student_emails = self.student_parser.get_student_emails()
        if not student_emails:
            return {
                "success": False,
                "error": "No student emails found. Please upload student email list first."
            }
        
        deadline_date = deadline_info.get("deadline", "")
        
        # Send notification
        result = self.email_sender.send_notification(
            to_emails=student_emails,
            deadline_date=deadline_date,
            deadline_time=deadline_info.get("deadline_time"),
            location=deadline_info.get("location"),
            submission_items=deadline_info.get("submission_items"),
            submission_method=deadline_info.get("submission_method"),
            additional_info=deadline_info.get("additional_info"),
            reminder_type=reminder_type
        )
        
        # Log notification to history
        if result.get("success"):
            self._log_notification(deadline_date, reminder_type, "sent")
            logger.info(f"Manual notification sent: {result['sent']} emails sent")
        else:
            self._log_notification(deadline_date, reminder_type, "failed")
            logger.error(f"Manual notification failed: {result.get('error')}")
        
        return result

