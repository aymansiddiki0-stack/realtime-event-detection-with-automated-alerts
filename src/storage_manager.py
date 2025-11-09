"""
Database operations - handles all Postgres reads/writes
"""

import os
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StorageManager:
    """Handles all database interactions"""

    def __init__(self):
        self.host = os.getenv('POSTGRES_HOST', 'localhost')
        self.port = os.getenv('POSTGRES_PORT', '5432')
        self.database = os.getenv('POSTGRES_DB', 'events_db')
        self.user = os.getenv('POSTGRES_USER', 'eventpipeline')
        self.password = os.getenv('POSTGRES_PASSWORD', 'pipeline_secret_2024')

        self._ensure_tables_exist()
        logger.info("Storage manager initialized")
    
    @contextmanager
    def get_connection(self):
        """Get a DB connection with auto-commit/rollback"""
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def _ensure_tables_exist(self):
        """Create tables and indexes if needed"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id SERIAL PRIMARY KEY,
                    event_id VARCHAR(255) UNIQUE NOT NULL,
                    source VARCHAR(50),
                    source_type VARCHAR(50),
                    title TEXT,
                    description TEXT,
                    content TEXT,
                    url TEXT,
                    published_at TIMESTAMP,
                    timestamp TIMESTAMP,
                    category VARCHAR(100),
                    category_confidence FLOAT,
                    crisis_level VARCHAR(20),
                    severity_score FLOAT,
                    persons TEXT,
                    organizations TEXT,
                    locations TEXT,
                    word_count INTEGER,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_events_category ON events(category);
                CREATE INDEX IF NOT EXISTS idx_events_crisis_level ON events(crisis_level);
                CREATE INDEX IF NOT EXISTS idx_events_processed_at ON events(processed_at);
            """)
            
            # Detected events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detected_events (
                    id SERIAL PRIMARY KEY,
                    detection_type VARCHAR(50),
                    category VARCHAR(100),
                    severity VARCHAR(20),
                    event_count INTEGER,
                    details JSONB,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    alert_sent BOOLEAN DEFAULT FALSE
                );
                
                CREATE INDEX IF NOT EXISTS idx_detected_events_severity ON detected_events(severity);
                CREATE INDEX IF NOT EXISTS idx_detected_events_detected_at ON detected_events(detected_at);
            """)
            
            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id SERIAL PRIMARY KEY,
                    detection_id INTEGER REFERENCES detected_events(id),
                    alert_type VARCHAR(50),
                    message TEXT,
                    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(20)
                );
            """)
            
            logger.info("Database tables verified")
    
    def insert_events(self, events: List[Dict]) -> int:
        """Batch insert events into DB"""
        if not events:
            return 0

        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                INSERT INTO events (
                    event_id, source, source_type, title, description, content,
                    url, published_at, timestamp, category, category_confidence,
                    crisis_level, severity_score, persons, organizations, locations,
                    word_count
                )
                VALUES %s
                ON CONFLICT (event_id) DO NOTHING
            """

            values = []
            for event in events:
                nlp_data = event.get('nlp_data', {})
                entities = nlp_data.get('entities', {})

                values.append((
                    event.get('event_id'),
                    event.get('source'),
                    event.get('source_type'),
                    event.get('title', '')[:500],
                    event.get('description', '')[:1000],
                    event.get('content', '')[:2000],
                    event.get('url', '')[:500],
                    event.get('published_at'),
                    event.get('timestamp'),
                    nlp_data.get('category', 'unknown'),
                    nlp_data.get('category_confidence', 0.0),
                    nlp_data.get('crisis_level', 'low'),
                    nlp_data.get('severity_score', 0.0),
                    str(entities.get('persons', [])),
                    str(entities.get('organizations', [])),
                    str(entities.get('locations', [])),
                    nlp_data.get('word_count', 0)
                ))

            execute_values(cursor, query, values)
            inserted = cursor.rowcount

            logger.info(f"Inserted {inserted} events")
            return inserted

    def insert_detected_event(self, detected_event: Dict) -> int:
        """Store a detected event (spike, cluster, etc)"""
        import json
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO detected_events (
                    detection_type, category, severity, event_count, details
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (
                detected_event.get('type'),
                detected_event.get('category'),
                detected_event.get('severity'),
                detected_event.get('event_count', 1),
                json.dumps(detected_event)
            ))
            
            detection_id = cursor.fetchone()[0]
            logger.info(f"Inserted detected event with ID: {detection_id}")
            return detection_id
    
    def insert_alert(self, detection_id: int, alert_type: str,
                    message: str, status: str = 'sent') -> int:
        """Record an alert that was sent"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO alerts (detection_id, alert_type, message, status)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (detection_id, alert_type, message, status))

            alert_id = cursor.fetchone()[0]

            cursor.execute("""
                UPDATE detected_events SET alert_sent = TRUE WHERE id = %s
            """, (detection_id,))

            return alert_id

    def get_recent_events(self, hours: int = 24, limit: int = 100) -> List[Dict]:
        """Fetch recent events from DB"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM events
                WHERE processed_at > NOW() - INTERVAL '%s hours'
                ORDER BY processed_at DESC
                LIMIT %s
            """, (hours, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_events_by_category(self, category: str, hours: int = 24) -> List[Dict]:
        """Filter events by category"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT * FROM events
                WHERE category = %s
                AND processed_at > NOW() - INTERVAL '%s hours'
                ORDER BY severity_score DESC, processed_at DESC
            """, (category, hours))

            return [dict(row) for row in cursor.fetchall()]

    def get_high_severity_events(self, min_severity: float = 0.7,
                                 hours: int = 24) -> List[Dict]:
        """Get events above severity threshold"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT * FROM events
                WHERE severity_score >= %s
                AND processed_at > NOW() - INTERVAL '%s hours'
                ORDER BY severity_score DESC, processed_at DESC
            """, (min_severity, hours))

            return [dict(row) for row in cursor.fetchall()]

    def get_category_stats(self, hours: int = 24) -> Dict:
        """Get aggregate stats per category"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    category,
                    COUNT(*) as count,
                    AVG(severity_score) as avg_severity,
                    MAX(severity_score) as max_severity
                FROM events
                WHERE processed_at > NOW() - INTERVAL '%s hours'
                GROUP BY category
                ORDER BY count DESC
            """, (hours,))
            
            return {row['category']: dict(row) for row in cursor.fetchall()}
    
    def get_detected_events(self, hours: int = 24,
                           min_severity: str = 'low') -> List[Dict]:
        """Get detected events with severity filter"""
        severity_order = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        min_level = severity_order.get(min_severity, 0)

        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT * FROM detected_events
                WHERE detected_at > NOW() - INTERVAL '%s hours'
                ORDER BY detected_at DESC
            """, (hours,))

            events = [dict(row) for row in cursor.fetchall()]

            filtered = [
                e for e in events
                if severity_order.get(e.get('severity', 'low'), 0) >= min_level
            ]

            return filtered

    def cleanup_old_data(self, days: int = 30):
        """Delete old records to save space"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM events
                WHERE processed_at < NOW() - INTERVAL '%s days'
            """, (days,))

            events_deleted = cursor.rowcount

            cursor.execute("""
                DELETE FROM detected_events
                WHERE detected_at < NOW() - INTERVAL '%s days'
            """, (days,))

            detected_deleted = cursor.rowcount

            logger.info(f"Cleaned up {events_deleted} events and {detected_deleted} detected events")
            return events_deleted + detected_deleted


# singleton pattern
_storage_manager = None


def get_storage_manager() -> StorageManager:
    """Get the storage manager instance"""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StorageManager()
    return _storage_manager


if __name__ == '__main__':
    # quick test
    storage = StorageManager()

    events = storage.get_recent_events(hours=1)
    print(f"Recent events: {len(events)}")

    stats = storage.get_category_stats(hours=24)
    print("Category stats:", stats)