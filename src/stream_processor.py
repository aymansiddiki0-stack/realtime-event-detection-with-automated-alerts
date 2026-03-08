"""
Event processor - consumes raw events from Kafka, runs NLP, writes to Postgres
"""

import os
import json
import time
import signal
import logging
import threading
from typing import Dict, List, Optional
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from prometheus_client import Counter, Histogram, start_http_server

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics for tracking performance
batches_processed = Counter('processor_batches_total', 'Total batches processed')
batches_failed = Counter('processor_batches_failed_total', 'Total batches failed')
events_consumed = Counter('processor_events_consumed_total', 'Total events consumed from Kafka')
events_written_db = Counter('processor_events_written_total', 'Total events written to database')
duplicates_skipped = Counter('processor_duplicates_skipped_total', 'Events skipped as duplicates on insert')
messages_invalid = Counter('processor_messages_invalid_total', 'Messages dropped as undecodable')
batch_processing_duration = Histogram('processor_batch_seconds', 'Time spent processing batches')
nlp_processing_duration = Histogram('nlp_processing_seconds', 'Time spent on NLP processing')
db_write_duration = Histogram('db_write_seconds', 'Time spent writing to database')
embedding_duration = Histogram('embedding_seconds', 'Time spent embedding events for retrieval')
events_embedded = Counter('processor_events_embedded_total', 'Events embedded for semantic retrieval')
events_embedding_failed = Counter('processor_events_embedding_failed_total', 'Events stored without a vector')


class EventConsumer:
    """Consumes raw events, enriches them with NLP, and persists them"""

    def __init__(self, nlp_processor=None, storage=None, embedder=None):
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.topic = 'raw-events'
        self.group_id = 'event-processor'

        # Injectable for tests; loaded lazily otherwise because the NLP
        # stack (spaCy + transformers) takes tens of seconds to import.
        self._nlp = nlp_processor
        self._storage = storage
        self._embedder = embedder

        self._running = True

        metrics_thread = threading.Thread(target=self._start_metrics_server, daemon=True)
        metrics_thread.start()

        logger.info("Event consumer initialized")

    @property
    def nlp(self):
        if self._nlp is None:
            from nlp_module import get_nlp_processor
            self._nlp = get_nlp_processor()
        return self._nlp

    @property
    def storage(self):
        if self._storage is None:
            from storage_manager import get_storage_manager
            self._storage = get_storage_manager()
        return self._storage

    @property
    def embedder(self):
        if self._embedder is None:
            from embeddings import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def _start_metrics_server(self):
        """Start Prometheus metrics HTTP server"""
        try:
            start_http_server(8000)
            logger.info("Prometheus metrics server started on port 8000")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")

    def _create_consumer(self) -> KafkaConsumer:
        """Connect to Kafka with retries"""
        max_retries = 5
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                consumer = KafkaConsumer(
                    self.topic,
                    bootstrap_servers=self.kafka_servers.split(','),
                    group_id=self.group_id,
                    # Offsets are committed manually after the database write,
                    # so a crash mid-batch replays the batch instead of losing it.
                    # The ON CONFLICT insert makes replays idempotent.
                    enable_auto_commit=False,
                    auto_offset_reset='earliest',
                    # Transformer inference is slow on CPU; keep polls small and
                    # the poll interval generous so the broker doesn't evict us
                    # from the group while a batch is still being classified.
                    max_poll_records=20,
                    max_poll_interval_ms=600000,
                )
                logger.info(f"Connected to Kafka at {self.kafka_servers}")
                return consumer
            except KafkaError as e:
                logger.error(f"Kafka connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise

    def _decode(self, raw_value: bytes) -> Optional[Dict]:
        """Decode a Kafka message, dropping anything unparseable"""
        try:
            return json.loads(raw_value.decode('utf-8'))
        except (ValueError, UnicodeDecodeError) as e:
            messages_invalid.inc()
            logger.warning(f"Dropping undecodable message: {e}")
            return None

    def _embed(self, events: List[Dict]) -> Optional[List[List[float]]]:
        """Vectors for a batch, or None if the model could not produce them.

        Embedding failure must not cost us the events themselves, so this
        degrades to None and the rows are stored without vectors rather than
        the batch being lost. They are simply not retrievable semantically
        until re-embedded.
        """
        try:
            with embedding_duration.time():
                return self.embedder.embed_events(events)
        except Exception as e:
            events_embedding_failed.inc(len(events))
            logger.warning(f"Embedding failed for {len(events)} events, storing without vectors: {e}")
            return None

    def handle_batch(self, events: List[Dict]) -> int:
        """Run NLP on a batch and persist it; returns rows actually inserted"""
        with nlp_processing_duration.time():
            processed_events = self.nlp.batch_process(events)

        embeddings = self._embed(processed_events)

        with db_write_duration.time():
            inserted = self.storage.insert_events(processed_events, embeddings)

        if embeddings is not None:
            events_embedded.inc(inserted)

        skipped = len(processed_events) - inserted
        events_written_db.inc(inserted)
        if skipped > 0:
            duplicates_skipped.inc(skipped)

        logger.info(f"Batch done: {inserted} inserted, {skipped} duplicates skipped")
        return inserted

    def stop(self, *args):
        logger.info("Shutdown requested")
        self._running = False

    def run(self):
        """Main loop: poll, enrich, persist, commit"""
        consumer = self._create_consumer()

        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)

        logger.info(f"Consuming from '{self.topic}' as group '{self.group_id}'")

        try:
            while self._running:
                records = consumer.poll(timeout_ms=5000)
                if not records:
                    continue

                events = []
                for partition_records in records.values():
                    for record in partition_records:
                        event = self._decode(record.value)
                        if event is not None:
                            events.append(event)

                events_consumed.inc(sum(len(v) for v in records.values()))

                if not events:
                    consumer.commit()
                    continue

                try:
                    with batch_processing_duration.time():
                        self.handle_batch(events)
                    consumer.commit()
                    batches_processed.inc()
                except Exception as e:
                    # Offsets stay uncommitted so the batch is redelivered;
                    # duplicates from partial writes are absorbed by ON CONFLICT.
                    batches_failed.inc()
                    logger.error(f"Batch failed, will retry after redelivery: {e}")
                    time.sleep(5)
        finally:
            consumer.close()
            logger.info("Consumer closed")


def main():
    processor = EventConsumer()
    processor.run()


if __name__ == '__main__':
    main()
