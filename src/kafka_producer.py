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
