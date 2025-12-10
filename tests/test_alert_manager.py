"""
Tests for Alert Manager
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from alert_manager import AlertManager


@pytest.fixture
def alert_manager(monkeypatch):
    """Alert manager with no channels configured, so nothing external is contacted"""
    for var in ('SLACK_WEBHOOK_URL', 'SMTP_USER', 'SMTP_PASSWORD', 'ALERT_EMAIL'):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv('MIN_ALERT_SEVERITY', 'medium')
    return AlertManager()


@pytest.fixture
def sample_spike():
    return {
        'type': 'keyword_spike',
        'category': 'natural_disaster',
        'keyword': 'earthquake',
        'count': 150,
        'baseline': 20.0,
        'spike_ratio': 7.5,
        'severity': 'critical'
    }


def test_no_channels_without_config(alert_manager):
    assert alert_manager.enabled_channels == []


def test_should_alert_thresholds(alert_manager):
    assert alert_manager.should_alert('critical')
    assert alert_manager.should_alert('high')
    assert alert_manager.should_alert('medium')
    assert not alert_manager.should_alert('low')
    assert not alert_manager.should_alert('unknown')


def test_slack_alert_unconfigured_returns_false(alert_manager):
    assert alert_manager.send_slack_alert('test message', 'high') is False


def test_email_alert_unconfigured_returns_false(alert_manager):
    assert alert_manager.send_email_alert('subject', 'test message', 'high') is False


def test_email_alert_with_config_builds_html(monkeypatch):
    """Multi-line messages must be converted to <br> in the HTML body"""
    monkeypatch.delenv('SLACK_WEBHOOK_URL', raising=False)
    monkeypatch.setenv('SMTP_USER', 'sender@example.com')
    monkeypatch.setenv('SMTP_PASSWORD', 'password')
    monkeypatch.setenv('ALERT_EMAIL', 'alerts@example.com')

    manager = AlertManager()
    sent_messages = []

    class FakeSMTP:
        def __init__(self, host, port):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def starttls(self):
            pass

        def login(self, user, password):
            pass

        def send_message(self, msg):
            sent_messages.append(msg)

    monkeypatch.setattr('alert_manager.smtplib.SMTP', FakeSMTP)

    result = manager.send_email_alert('Test', 'line one\nline two', 'high')

    assert result is True
    assert len(sent_messages) == 1

    html_part = sent_messages[0].get_payload()[1].get_payload(decode=True).decode('utf-8')
    assert 'line one<br>line two' in html_part


def test_format_keyword_spike_alert(alert_manager, sample_spike):
    message = alert_manager.format_keyword_spike_alert(sample_spike)

    assert 'earthquake' in message
    assert 'NATURAL_DISASTER' in message
    assert '150' in message
    assert '7.5' in message


def test_format_location_cluster_alert(alert_manager):
    cluster = {
        'type': 'location_cluster',
        'location': 'California',
        'event_count': 5,
        'category': 'natural_disaster',
        'avg_severity': 0.8,
        'severity': 'high'
    }
    message = alert_manager.format_location_cluster_alert(cluster)

    assert 'California' in message
    assert '5' in message


def test_send_alert_below_threshold_skips(alert_manager, sample_spike):
    sample_spike['severity'] = 'low'
    results = alert_manager.send_alert(sample_spike)

    assert results == {'slack': False, 'email': False}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
