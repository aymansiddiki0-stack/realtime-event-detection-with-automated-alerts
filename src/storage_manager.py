"""
Database operations - handles all Postgres reads/writes
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values, RealDictCursor
from contextlib import contextmanager

from embeddings import to_pgvector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Single source of truth for the schema. Postgres runs it on first boot via
# docker-entrypoint-initdb.d; StorageManager reruns it on startup so an
# existing volume also picks up new objects. Every statement must stay
# idempotent (IF NOT EXISTS / OR REPLACE).
SCHEMA_FILE = Path(__file__).resolve().parent.parent / 'sql' / 'init.sql'


class StorageManager:
    """Handles all database interactions"""

    def __init__(self):
        self.host = os.getenv('POSTGRES_HOST', 'localhost')
        self.port = os.getenv('POSTGRES_PORT', '5432')
        self.database = os.getenv('POSTGRES_DB', 'events_db')
        self.user = os.getenv('POSTGRES_USER', 'eventpipeline')
        self.password = os.getenv('POSTGRES_PASSWORD')

        # Streamlit reruns the whole script on every widget interaction, and
        # each rerun issues several queries; a pool keeps that from opening a
        # new TCP connection and re-authenticating every time.
        self._pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=int(os.getenv('POSTGRES_POOL_SIZE', 10)),
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )

        self._ensure_tables_exist()
        logger.info("Storage manager initialized")

    @contextmanager
    def get_connection(self):
        """Borrow a pooled connection, committing or rolling back on exit"""
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            self._pool.putconn(conn)

    def close(self):
        """Release every pooled connection"""
        if self._pool and not self._pool.closed:
            self._pool.closeall()
            logger.info("Connection pool closed")

    def _ensure_tables_exist(self):
        """Apply the schema file so the database matches sql/init.sql"""
        if not SCHEMA_FILE.exists():
            logger.warning(f"Schema file not found at {SCHEMA_FILE}, skipping")
            return

        schema_sql = SCHEMA_FILE.read_text(encoding='utf-8')

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(schema_sql)

        logger.info("Database schema applied")
    
    def insert_events(self, events: List[Dict],
                      embeddings: Optional[List[List[float]]] = None) -> int:
        """Batch insert events into DB, optionally with their vectors.

        Embeddings are positional: embeddings[i] belongs to events[i]. Passing
        None stores NULL, which keeps this usable from callers that have no
        model loaded.
        """
        if not events:
            return 0

        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                INSERT INTO events (
                    event_id, source, source_type, title, description, content,
                    url, published_at, timestamp, category, category_confidence,
                    crisis_level, severity_score, persons, organizations, locations,
                    word_count, embedding
                )
                VALUES %s
                ON CONFLICT (event_id) DO NOTHING
            """

            values = []
            for index, event in enumerate(events):
                nlp_data = event.get('nlp_data', {})
                entities = nlp_data.get('entities', {})

                vector = None
                if embeddings is not None and index < len(embeddings):
                    vector = to_pgvector(embeddings[index])

                values.append((
                    event.get('event_id'),
                    event.get('source'),
                    event.get('source_type'),
                    (event.get('title') or '')[:500],
                    (event.get('description') or '')[:1000],
                    (event.get('content') or '')[:2000],
                    (event.get('url') or '')[:500],
                    event.get('published_at') or None,
                    event.get('timestamp') or None,
                    nlp_data.get('category', 'unknown'),
                    nlp_data.get('category_confidence', 0.0),
                    nlp_data.get('crisis_level', 'low'),
                    nlp_data.get('severity_score', 0.0),
                    # JSON, not repr(): the dashboard parses these with json.loads
                    json.dumps(entities.get('persons', [])),
                    json.dumps(entities.get('organizations', [])),
                    json.dumps(entities.get('locations', [])),
                    nlp_data.get('word_count', 0),
                    vector
                ))

            execute_values(cursor, query, values)
            inserted = cursor.rowcount

            logger.info(f"Inserted {inserted} events")
            return inserted

    def insert_detected_event(self, detected_event: Dict) -> int:
        """Store a detected event (spike, cluster, etc)"""
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

    def get_keyword_baselines(self, category: str, keywords: List[str],
                              window_hours: int = 168) -> Dict[str, Dict]:
        """Median historical mention count per keyword, with sample size"""
        if not keywords:
            return {}

        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT
                    keyword,
                    percentile_cont(0.5) WITHIN GROUP (ORDER BY mention_count) AS median_count,
                    COUNT(*) AS observations
                FROM keyword_observations
                WHERE category = %s
                  AND keyword = ANY(%s)
                  AND observed_at > NOW() - make_interval(hours => %s)
                GROUP BY keyword
            """, (category, list(keywords), window_hours))

            return {
                row['keyword']: {
                    'baseline': float(row['median_count']),
                    'observations': int(row['observations'])
                }
                for row in cursor.fetchall()
            }

    def record_keyword_observations(self, category: str, counts: Dict[str, int]) -> int:
        """Store this run's mention counts so future runs have a baseline"""
        if not counts:
            return 0

        with self.get_connection() as conn:
            cursor = conn.cursor()

            execute_values(cursor, """
                INSERT INTO keyword_observations (category, keyword, mention_count)
                VALUES %s
            """, [(category, keyword, count) for keyword, count in counts.items()])

            return cursor.rowcount

    def search_events_by_vector(self, query_vector: List[float],
                                limit: int = 5) -> List[Dict]:
        """Return the events most semantically similar to a query vector.

        Rows with no embedding are excluded rather than ranked last: a NULL
        embedding means "not yet indexed", which is a different thing from
        "indexed and unrelated", and ordering them in would put unindexed
        events above genuinely weak matches.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT
                    event_id, title, description, url, source,
                    published_at, category, crisis_level, severity_score,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM events
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (to_pgvector(query_vector), to_pgvector(query_vector), limit))

            return [dict(row) for row in cursor.fetchall()]

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

            # Baselines only look back baseline_window_hours, so anything
            # older than the retention window is unreachable history.
            cursor.execute("""
                DELETE FROM keyword_observations
                WHERE observed_at < NOW() - make_interval(days => %s)
            """, (days,))

            observations_deleted = cursor.rowcount

            logger.info(
                f"Cleaned up {events_deleted} events, {detected_deleted} detected events, "
                f"{observations_deleted} keyword observations"
            )
            return events_deleted + detected_deleted + observations_deleted


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