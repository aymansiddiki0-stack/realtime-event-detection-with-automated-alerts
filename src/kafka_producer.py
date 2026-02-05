"""
Data ingestion from news APIs - Kafka Pulls from NewsAPI, Reddit, and GDELT every minute
"""

import os
import json
import time
import random
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from kafka import KafkaProducer
from kafka.errors import KafkaError
import requests
from dotenv import load_dotenv
import praw
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import threading

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/kafka_producer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Track metrics for monitoring
messages_sent_total = Counter('kafka_messages_sent_total', 'Total messages sent to Kafka', ['source'])
messages_failed_total = Counter('kafka_messages_failed_total', 'Total failed messages', ['source'])
events_fetched_total = Counter('events_fetched_total', 'Total events fetched from sources', ['source'])
events_skipped_total = Counter('events_skipped_total', 'Events dropped for lacking a usable identity', ['source'])
rate_limited_total = Counter('source_rate_limited_total', 'Rate-limit responses from sources', ['source'])
source_backoff_seconds = Gauge('source_backoff_seconds', 'Current backoff delay per source', ['source'])
fetch_duration_seconds = Histogram('fetch_duration_seconds', 'Time spent fetching from sources', ['source'])
active_sources = Gauge('active_data_sources', 'Number of active data sources')


# Analytics parameters that vary between fetches of the same article and
# would otherwise produce distinct identities for identical content.
TRACKING_PARAMS = {
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
    'utm_id', 'utm_name', 'utm_reader', 'fbclid', 'gclid', 'msclkid',
    'mc_cid', 'mc_eid', 'ref', 'ref_src', 'ocid', 'cmpid', 'icid',
    'igshid', 'spm', '_ga',
}


def normalize_url(url: str) -> str:
    """Reduce a URL to a canonical form for identity purposes"""
    if not url:
        return ''

    parts = urlsplit(url.strip())

    scheme = parts.scheme.lower() or 'https'
    netloc = parts.netloc.lower()

    # Same host, same content: strip the www prefix and default ports.
    if netloc.startswith('www.'):
        netloc = netloc[4:]
    if netloc.endswith(':80') or netloc.endswith(':443'):
        netloc = netloc.rsplit(':', 1)[0]

    path = parts.path.rstrip('/') or '/'

    query = urlencode(sorted(
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in TRACKING_PARAMS
    ))

    # Fragments never identify a distinct article.
    return urlunsplit((scheme, netloc, path, query, ''))


def make_event_id(source: str, identifier: str) -> str:
    """Build a stable event ID from a source-scoped identifier.

    Deterministic across processes and fetch cycles: Python's hash() is
    randomized per process, and anything time-based defeats deduplication.
    """
    digest = hashlib.sha256(identifier.encode('utf-8')).hexdigest()[:32]
    return f"{source}_{digest}"


def url_event_id(source: str, url: str) -> str:
    """Event ID derived from a normalized article URL"""
    return make_event_id(source, normalize_url(url))


# Sources are polled on their own cadence. GDELT publishes roughly every 15
# minutes and returned sustained 429s under a shared 60s interval; NewsAPI's
# free tier allows 100 requests a day, which 60s polling exceeds twelve-fold.
DEFAULT_INTERVALS = {
    'newsapi': 900,
    'reddit': 300,
    'gdelt': 900,
}

MAX_BACKOFF_SECONDS = 3600


class RateLimited(Exception):
    """A source asked us to slow down"""

    def __init__(self, source: str, retry_after: Optional[float] = None):
        super().__init__(f"{source} rate limited")
        self.source = source
        self.retry_after = retry_after


class SourceSchedule:
    """Decides when a source may next be polled"""

    def __init__(self, name: str, interval: float):
        self.name = name
        self.interval = interval
        self.next_allowed = 0.0
        self.backoff = 0.0

    def ready(self, now: float) -> bool:
        return now >= self.next_allowed

    def record_success(self, now: float):
        self.backoff = 0.0
        source_backoff_seconds.labels(source=self.name).set(0)
        self.next_allowed = now + self.interval

    def record_rate_limit(self, now: float, retry_after: Optional[float] = None) -> float:
        if retry_after is not None:
            delay = retry_after
        else:
            self.backoff = min(max(self.backoff * 2, self.interval), MAX_BACKOFF_SECONDS)
            delay = self.backoff

        # Jitter keeps several sources from retrying in lockstep after a
        # shared outage.
        delay *= random.uniform(1.0, 1.25)

        self.next_allowed = now + delay
        source_backoff_seconds.labels(source=self.name).set(delay)
        return delay


def parse_retry_after(response) -> Optional[float]:
    """Read a Retry-After header, which may be seconds or an HTTP date"""
    value = response.headers.get('Retry-After') if response is not None else None
    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        pass

    try:
        from email.utils import parsedate_to_datetime
        retry_at = parsedate_to_datetime(value)
        return max(0.0, (retry_at - datetime.now(retry_at.tzinfo)).total_seconds())
    except (TypeError, ValueError):
        return None


class EventProducer:
    """Fetches events from news sources and sends them to Kafka"""

    def __init__(self):
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.producer = self._create_producer()
        self.topic = 'raw-events'

        self.newsapi_key = os.getenv('NEWSAPI_KEY', '')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID', '')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET', '')
        self.reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'EventPipeline/1.0')

        # Set up Reddit if we have creds
        self.reddit = None
        if self.reddit_client_id and self.reddit_client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    user_agent=self.reddit_user_agent
                )
                logger.info("Reddit API initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Reddit API: {e}")

        self.schedules = {
            name: SourceSchedule(name, float(os.getenv(f'{name.upper()}_INTERVAL', default)))
            for name, default in DEFAULT_INTERVALS.items()
        }

        # Metrics server runs in background
        metrics_thread = threading.Thread(target=self._start_metrics_server, daemon=True)
        metrics_thread.start()

        # Count how many sources we're actually using
        active_count = sum([
            1 if self.newsapi_key else 0,
            1 if self.reddit else 0,
            1  # GDELT doesn't need auth
        ])
        active_sources.set(active_count)

    def _start_metrics_server(self):
        """Start Prometheus metrics HTTP server"""
        try:
            start_http_server(8000)
            logger.info("Prometheus metrics server started on port 8000")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    def _create_producer(self) -> KafkaProducer:
        """Connect to Kafka with retries"""
        max_retries = 5
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                producer = KafkaProducer(
                    bootstrap_servers=self.kafka_servers.split(','),
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    key_serializer=lambda k: k.encode('utf-8') if k else None,
                    acks='all',
                    retries=3,
                    max_in_flight_requests_per_connection=1
                )
                logger.info(f"Connected to Kafka at {self.kafka_servers}")
                return producer
            except KafkaError as e:
                logger.error(f"Kafka connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise
    
    def _get(self, source: str, url: str, params: Dict, timeout: int):
        """HTTP GET that turns a rate-limit response into RateLimited"""
        response = requests.get(url, params=params, timeout=timeout)

        if response.status_code == 429:
            rate_limited_total.labels(source=source).inc()
            raise RateLimited(source, parse_retry_after(response))

        response.raise_for_status()
        return response

    def fetch_newsapi(self) -> List[Dict]:
        """Fetch latest news from NewsAPI"""
        if not self.newsapi_key:
            logger.warning("NewsAPI key not configured, using mock data")
            return self._generate_mock_news()

        with fetch_duration_seconds.labels(source='newsapi').time():
            try:
                url = 'https://newsapi.org/v2/top-headlines'
                params = {
                    'apiKey': self.newsapi_key,
                    'language': 'en',
                    'pageSize': 100,
                    'category': 'general'
                }

                response = self._get('newsapi', url, params, timeout=10)

                articles = response.json().get('articles', [])
                events = []

                for article in articles:
                    url = article.get('url', '')
                    if not url:
                        events_skipped_total.labels(source='newsapi').inc()
                        continue

                    event = {
                        'source': 'newsapi',
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'url': url,
                        'published_at': article.get('publishedAt', ''),
                        'source_name': article.get('source', {}).get('name', ''),
                        'timestamp': datetime.utcnow().isoformat(),
                        'event_id': url_event_id('news', url)
                    }
                    events.append(event)

                events_fetched_total.labels(source='newsapi').inc(len(events))
                logger.info(f"Fetched {len(events)} articles from NewsAPI")
                return events

            except requests.RequestException as e:
                logger.error(f"Error fetching from NewsAPI: {e}")
                return []
    
    def fetch_reddit(self) -> List[Dict]:
        """Fetch posts from relevant subreddits"""
        if not self.reddit:
            logger.warning("Reddit API not configured, using mock data")
            return self._generate_mock_reddit()

        with fetch_duration_seconds.labels(source='reddit').time():
            try:
                subreddits = ['worldnews', 'news', 'politics', 'technology']
                events = []

                for subreddit_name in subreddits:
                    subreddit = self.reddit.subreddit(subreddit_name)

                    # Get hot posts
                    for post in subreddit.hot(limit=25):
                        event = {
                            'source': 'reddit',
                            'title': post.title,
                            'content': post.selftext,
                            'url': f"https://reddit.com{post.permalink}",
                            'subreddit': subreddit_name,
                            'score': post.score,
                            'num_comments': post.num_comments,
                            'created_utc': datetime.fromtimestamp(post.created_utc).isoformat(),
                            'timestamp': datetime.utcnow().isoformat(),
                            # Reddit post IDs are already stable and unique.
                            'event_id': make_event_id('reddit', post.id)
                        }
                        events.append(event)

                events_fetched_total.labels(source='reddit').inc(len(events))
                logger.info(f"Fetched {len(events)} posts from Reddit")
                return events

            except Exception as e:
                logger.error(f"Error fetching from Reddit: {e}")
                return []
    
    def fetch_gdelt(self) -> List[Dict]:
        """Fetch events from GDELT API"""
        with fetch_duration_seconds.labels(source='gdelt').time():
            try:
                # GDELT GKG API - last 15 minutes
                url = 'https://api.gdeltproject.org/api/v2/doc/doc'
                params = {
                    'query': 'sourcecountry:US',
                    'mode': 'artlist',
                    'maxrecords': 250,
                    'format': 'json'
                }

                response = self._get('gdelt', url, params, timeout=15)

                articles = response.json().get('articles', [])
                events = []

                for article in articles:
                    url = article.get('url', '')
                    if not url:
                        events_skipped_total.labels(source='gdelt').inc()
                        continue

                    event = {
                        'source': 'gdelt',
                        'title': article.get('title', ''),
                        'url': url,
                        'language': article.get('language', ''),
                        'seendate': article.get('seendate', ''),
                        'timestamp': datetime.utcnow().isoformat(),
                        'event_id': url_event_id('gdelt', url)
                    }
                    events.append(event)

                events_fetched_total.labels(source='gdelt').inc(len(events))
                logger.info(f"Fetched {len(events)} articles from GDELT")
                return events

            except requests.RequestException as e:
                logger.error(f"Error fetching from GDELT: {e}")
                return []
    
    def _generate_mock_news(self) -> List[Dict]:
        """Mock data when API key isn't set"""
        mock_events = [
            {
                'source': 'newsapi',
                'title': 'Major earthquake strikes Pacific region',
                'description': 'A 7.2 magnitude earthquake hit the Pacific coast',
                'content': 'Emergency services are responding to a major seismic event...',
                'url': 'https://example.com/earthquake-1',
                'published_at': datetime.utcnow().isoformat(),
                'source_name': 'Mock News',
                'timestamp': datetime.utcnow().isoformat(),
                'event_id': url_event_id('news', 'https://example.com/earthquake-1')
            },
            {
                'source': 'newsapi',
                'title': 'Tech company announces breakthrough in AI',
                'description': 'New language model shows unprecedented capabilities',
                'content': 'A leading tech company has unveiled their latest AI system...',
                'url': 'https://example.com/ai-breakthrough',
                'published_at': datetime.utcnow().isoformat(),
                'source_name': 'Mock Tech News',
                'timestamp': datetime.utcnow().isoformat(),
                'event_id': url_event_id('news', 'https://example.com/ai-breakthrough')
            }
        ]
        return mock_events
    
    def _generate_mock_reddit(self) -> List[Dict]:
        """Mock Reddit posts for testing"""
        mock_posts = [
            {
                'source': 'reddit',
                'title': 'Discussion: Climate change impacts',
                'content': 'What are the most pressing climate issues?',
                'url': 'https://reddit.com/r/worldnews/mock1',
                'subreddit': 'worldnews',
                'score': 1500,
                'num_comments': 234,
                'created_utc': datetime.utcnow().isoformat(),
                'timestamp': datetime.utcnow().isoformat(),
                'event_id': make_event_id('reddit', 'mock1')
            }
        ]
        return mock_posts
    
    def send_to_kafka(self, events: List[Dict], source: str):
        """Push events to Kafka topic"""
        success_count = 0

        for event in events:
            try:
                event['ingestion_time'] = datetime.utcnow().isoformat()
                event['source_type'] = source

                future = self.producer.send(
                    self.topic,
                    key=event['event_id'],
                    value=event
                )

                record_metadata = future.get(timeout=10)
                success_count += 1
                messages_sent_total.labels(source=source).inc()

            except KafkaError as e:
                logger.error(f"Failed to send event {event.get('event_id')}: {e}")
                messages_failed_total.labels(source=source).inc()
            except Exception as e:
                logger.error(f"Unexpected error sending event: {e}")
                messages_failed_total.labels(source=source).inc()

        logger.info(f"Sent {success_count}/{len(events)} events from {source} to Kafka")
    
    def poll_once(self, now: Optional[float] = None) -> Dict[str, int]:
        """Poll every source whose interval has elapsed"""
        now = time.monotonic() if now is None else now
        fetchers = {
            'newsapi': self.fetch_newsapi,
            'reddit': self.fetch_reddit,
            'gdelt': self.fetch_gdelt,
        }

        sent = {}

        for source, fetch in fetchers.items():
            schedule = self.schedules[source]
            if not schedule.ready(now):
                continue

            try:
                events = fetch()
            except RateLimited as e:
                delay = schedule.record_rate_limit(now, e.retry_after)
                logger.warning(
                    f"{source} rate limited, next attempt in {delay:.0f}s"
                    + (" (Retry-After)" if e.retry_after is not None else " (backoff)")
                )
                continue

            schedule.record_success(now)

            if events:
                self.send_to_kafka(events, source)
                sent[source] = len(events)

        return sent

    def run(self, tick_seconds: int = 15):
        """Main loop - polls each source on its own schedule"""
        intervals = {name: s.interval for name, s in self.schedules.items()}
        logger.info(f"Starting event producer (intervals: {intervals})")

        try:
            while True:
                self.poll_once()
                self.producer.flush()
                time.sleep(tick_seconds)

        except KeyboardInterrupt:
            logger.info("Shutting down producer")
        finally:
            self.producer.close()
            logger.info("Producer closed")


def main():
    producer = EventProducer()

    # How often schedules are checked, not how often a source is polled.
    producer.run(tick_seconds=int(os.getenv('PRODUCER_TICK_SECONDS', 15)))


if __name__ == '__main__':
    main()
