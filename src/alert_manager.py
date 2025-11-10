"""
Alert Manager - Sends alerts via Slack and email when events are detected
"""

import os
import logging
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
import requests
from dotenv import load_dotenv

load_dotenv('configs/credentials.env')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertManager:
    """Sends notifications to Slack and email"""

    def __init__(self):
        self.slack_webhook = os.getenv('SLACK_WEBHOOK_URL', '')

        self.smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = os.getenv('SMTP_USER', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.alert_email = os.getenv('ALERT_EMAIL', '')

        self.min_severity = os.getenv('MIN_ALERT_SEVERITY', 'medium')

        self.enabled_channels = []
        if self.slack_webhook:
            self.enabled_channels.append('slack')
        if self.smtp_user and self.alert_email:
            self.enabled_channels.append('email')

        logger.info(f"Alert manager initialized with channels: {self.enabled_channels}")
    
    def should_alert(self, severity: str) -> bool:
        """Check if severity is high enough"""
        severity_levels = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}

        min_level = severity_levels.get(self.min_severity, 1)
        event_level = severity_levels.get(severity, 0)

        return event_level >= min_level

    def send_slack_alert(self, message: str, severity: str = 'medium') -> bool:
        """Post message to Slack webhook"""
        if not self.slack_webhook:
            logger.warning("Slack webhook not configured")
            return False
        
        try:
            colors = {
                'low': '#36a64f',
                'medium': '#ff9900',
                'high': '#ff0000',
                'critical': '#8b0000'
            }

            color = colors.get(severity, '#808080')

            payload = {
                'attachments': [{
                    'color': color,
                    'title': f'🚨 Event Alert - {severity.upper()}',
                    'text': message,
                    'footer': 'Real-Time Event Pipeline',
                    'ts': int(datetime.utcnow().timestamp())
                }]
            }

            response = requests.post(
                self.slack_webhook,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                logger.info("Slack alert sent successfully")
                return True
            else:
                logger.error(f"Slack alert failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
