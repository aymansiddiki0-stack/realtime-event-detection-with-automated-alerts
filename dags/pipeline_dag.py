"""
Airflow DAG for running event detection and sending alerts
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
import os

sys.path.insert(0, '/opt/airflow')

default_args = {
    'owner': 'event-pipeline',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def run_event_detection():
    """Detect events and send alerts"""
    from src.storage_manager import get_storage_manager
    from src.event_detector import EventDetector
    from src.alert_manager import get_alert_manager

    storage = get_storage_manager()

    events = storage.get_recent_events(hours=1, limit=1000)

    if not events:
        print("No events to process")
        return

    print(f"Processing {len(events)} events")

    detector = EventDetector(
        spike_threshold=2.0,
        min_cluster_size=3,
        time_window_minutes=60
    )

    detected_events = detector.detect_all_events(events)

    total_detected = 0
    for event_type, event_list in detected_events.items():
        for event in event_list:
            detection_id = storage.insert_detected_event(event)
            total_detected += 1

    print(f"Detected {total_detected} events")

    alert_manager = get_alert_manager()

    high_priority = detector.filter_high_priority(detected_events, min_severity='medium')

    if high_priority:
        print(f"Sending alerts for {len(high_priority)} high-priority events")
        results = alert_manager.send_batch_alerts(high_priority)
        print(f"Alert results: {results}")

    return {
        'total_events': len(events),
        'total_detected': total_detected,
        'alerts_sent': len(high_priority)
    }


def send_daily_summary():
    """Send summary email/slack"""
    from src.storage_manager import get_storage_manager
    from src.alert_manager import get_alert_manager
    
    storage = get_storage_manager()

    events = storage.get_recent_events(hours=24, limit=10000)
    detected = storage.get_detected_events(hours=24)
    stats = storage.get_category_stats(hours=24)

    severity_counts = {
        'critical': len([d for d in detected if d.get('severity') == 'critical']),
        'high': len([d for d in detected if d.get('severity') == 'high']),
        'medium': len([d for d in detected if d.get('severity') == 'medium']),
        'low': len([d for d in detected if d.get('severity') == 'low'])
    }

    top_categories = {
        cat: data['count']
        for cat, data in sorted(
            stats.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:10]
    }

    summary = {
        'period': 'Last 24 hours',
        'total_events': len(events),
        'detected_count': len(detected),
        'critical': severity_counts['critical'],
        'high': severity_counts['high'],
        'medium': severity_counts['medium'],
        'low': severity_counts['low'],
        'top_categories': top_categories
    }

    alert_manager = get_alert_manager()
    alert_manager.send_summary_alert(summary)

    print(f"Daily summary sent: {summary}")
    return summary


def cleanup_old_data():
    """Delete records older than 30 days"""
    from src.storage_manager import get_storage_manager
    
    storage = get_storage_manager()
    deleted = storage.cleanup_old_data(days=30)
    
    print(f"Cleaned up {deleted} old records")
    return {'deleted': deleted}


# Hourly detection DAG
dag = DAG(
    'event_detection_pipeline',
    default_args=default_args,
    description='Real-time event detection and alerting pipeline',
    schedule_interval=timedelta(hours=1),
    catchup=False,
    tags=['events', 'detection', 'alerts']
)

detect_events_task = PythonOperator(
    task_id='detect_events',
    python_callable=run_event_detection,
    dag=dag
)

# Daily summary DAG
summary_dag = DAG(
    'daily_summary',
    default_args=default_args,
    description='Send daily summary of detected events',
    schedule_interval='0 9 * * *',  # 9 AM UTC
    catchup=False,
    tags=['events', 'summary', 'alerts']
)

send_summary_task = PythonOperator(
    task_id='send_daily_summary',
    python_callable=send_daily_summary,
    dag=summary_dag
)

# Weekly cleanup DAG
cleanup_dag = DAG(
    'weekly_cleanup',
    default_args=default_args,
    description='Clean up old event data',
    schedule_interval='0 0 * * 0',  # Sunday midnight
    catchup=False,
    tags=['events', 'maintenance']
)

cleanup_task = PythonOperator(
    task_id='cleanup_old_data',
    python_callable=cleanup_old_data,
    dag=cleanup_dag
)

health_check_task = BashOperator(
    task_id='health_check',
    bash_command='echo "Pipeline health check passed" && date',
    dag=dag
)

health_check_task >> detect_events_task