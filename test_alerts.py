"""
Quick test script to check if Slack/Email alerts work
"""

import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

load_dotenv('configs/credentials.env')

from alert_manager import get_alert_manager

def test_alerts():
    manager = get_alert_manager()

    print(f"Enabled channels: {manager.enabled_channels}")
    print(f"Min severity: {manager.min_severity}")

    test_spike = {
        'type': 'keyword_spike',
        'category': 'disaster',
        'keyword': 'earthquake',
        'count': 150,
        'baseline': 20.0,
        'spike_ratio': 7.5,
        'severity': 'critical'
    }

    print(f"\nSending test alert for {test_spike['keyword']} ({test_spike['severity']})")
    results = manager.send_alert(test_spike)

    for channel, success in results.items():
        status = "sent" if success else "failed"
        print(f"{channel}: {status}")

if __name__ == '__main__':
    test_alerts()
