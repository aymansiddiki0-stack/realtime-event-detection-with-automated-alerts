"""
Data ingestion from news APIs - Kafka Pulls from NewsAPI, Reddit, and GDELT every minute
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from kafka import KafkaProducer
from kafka.errors import KafkaError
import requests
from dotenv import load_dotenv
import praw
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import threading

load_dotenv('configs/credentials.env')

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
fetch_duration_seconds = Histogram('fetch_duration_seconds', 'Time spent fetching from sources', ['source'])
active_sources = Gauge('active_data_sources', 'Number of active data sources')


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

        self.last_fetch = {
            'newsapi': None,
            'reddit': None,
            'gdelt': None
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

                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()

                articles = response.json().get('articles', [])
                events = []

                for article in articles:
                    event = {
                        'source': 'newsapi',
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'url': article.get('url', ''),
                        'published_at': article.get('publishedAt', ''),
                        'source_name': article.get('source', {}).get('name', ''),
                        'timestamp': datetime.utcnow().isoformat(),
                        'event_id': f"news_{hash(article.get('url', ''))}_{int(time.time())}"
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
                            'event_id': f"reddit_{post.id}_{int(time.time())}"
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

                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()

                articles = response.json().get('articles', [])
                events = []

                for article in articles:
                    event = {
                        'source': 'gdelt',
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'language': article.get('language', ''),
                        'seendate': article.get('seendate', ''),
                        'timestamp': datetime.utcnow().isoformat(),
                        'event_id': f"gdelt_{hash(article.get('url', ''))}_{int(time.time())}"
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
                'event_id': f"mock_news_{int(time.time())}_1"
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
                'event_id': f"mock_news_{int(time.time())}_2"
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
                'event_id': f"mock_reddit_{int(time.time())}_1"
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
    
    def run(self, fetch_interval: int = 60):
        """Main loop - fetches data and sends to Kafka"""
        logger.info(f"Starting event producer (fetch interval: {fetch_interval}s)")

        try:
            while True:
                news_events = self.fetch_newsapi()
                if news_events:
                    self.send_to_kafka(news_events, 'newsapi')

                reddit_events = self.fetch_reddit()
                if reddit_events:
                    self.send_to_kafka(reddit_events, 'reddit')

                gdelt_events = self.fetch_gdelt()
                if gdelt_events:
                    self.send_to_kafka(gdelt_events, 'gdelt')

                self.producer.flush()

                logger.info(f"Cycle complete. Waiting {fetch_interval}s")
                time.sleep(fetch_interval)

        except KeyboardInterrupt:
            logger.info("Shutting down producer")
        finally:
            self.producer.close()
            logger.info("Producer closed")


def main():
    producer = EventProducer()

    # Default is 60s, can be changed via env var
    fetch_interval = int(os.getenv('FETCH_INTERVAL', 60))

    producer.run(fetch_interval=fetch_interval)


if __name__ == '__main__':
    main()
