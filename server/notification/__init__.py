"""
Notification module for email notifications and deadline management.
"""

from .email_sender import EmailSender
from .scheduler import NotificationScheduler
from .student_parser import StudentEmailParser
from .deadline_parser import DeadlineParser

__all__ = ['EmailSender', 'NotificationScheduler', 'StudentEmailParser', 'DeadlineParser']

