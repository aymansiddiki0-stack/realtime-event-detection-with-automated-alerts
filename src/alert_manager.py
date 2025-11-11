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

    def send_email_alert(self, subject: str, message: str,
                        severity: str = 'medium') -> bool:
        """Send formatted email alert"""
        if not self.smtp_user or not self.alert_email:
            logger.warning("Email not configured")
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{severity.upper()}] {subject}"
            msg['From'] = self.smtp_user
            msg['To'] = self.alert_email

            html = f"""
            <html>
            <head></head>
            <body>
                <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                    <div style="background-color: {'#ff0000' if severity == 'critical' else '#ff9900' if severity == 'high' else '#36a64f'}; 
                                color: white; padding: 20px; text-align: center;">
                        <h2>🚨 Event Alert - {severity.upper()}</h2>
                    </div>
                    <div style="padding: 20px; background-color: #f5f5f5;">
                        <p style="font-size: 16px; line-height: 1.6;">
                            {message.replace('\n', '<br>')}
                        </p>
                    </div>
                    <div style="padding: 10px; text-align: center; font-size: 12px; color: #666;">
                        Real-Time Event Detection Pipeline
                    </div>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(message, 'plain'))
            msg.attach(MIMEText(html, 'html'))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info("Email alert sent successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def format_keyword_spike_alert(self, spike: Dict) -> str:
        """Format message for keyword spike"""
        return f"""
📊 KEYWORD SPIKE DETECTED

Category: {spike.get('category', 'unknown').upper()}
Keyword: "{spike.get('keyword', 'N/A')}"
Mentions: {spike.get('count', 0)}
Baseline: {spike.get('baseline', 0):.1f}
Spike Ratio: {spike.get('spike_ratio', 0):.2f}x
Severity: {spike.get('severity', 'unknown').upper()}

This keyword is being mentioned {spike.get('spike_ratio', 0):.1f}x more than normal.
Dashboard: http://localhost:8501
        """.strip()
    
    def format_location_cluster_alert(self, cluster: Dict) -> str:
        """Format message for location cluster"""
        return f"""
🌍 LOCATION CLUSTER DETECTED

Location: {cluster.get('location', 'Unknown')}
Event Count: {cluster.get('event_count', 0)}
Category: {cluster.get('category', 'unknown').upper()}
Average Severity: {cluster.get('avg_severity', 0):.2f}
Severity Level: {cluster.get('severity', 'unknown').upper()}

Multiple events detected in this location.
Dashboard: http://localhost:8501
        """.strip()

    def format_topic_cluster_alert(self, cluster: Dict) -> str:
        """Format message for topic cluster"""
        keywords = cluster.get('keywords', [])
        keywords_str = ', '.join(keywords[:5])

        return f"""
📰 TOPIC CLUSTER DETECTED

Cluster ID: {cluster.get('cluster_id', 'unknown')}
Event Count: {cluster.get('event_count', 0)}
Category: {cluster.get('category', 'unknown').upper()}
Keywords: {keywords_str}
Severity: {cluster.get('severity', 'unknown').upper()}

Related events detected on similar topics.
Dashboard: http://localhost:8501
        """.strip()
