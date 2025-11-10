"""
Spark streaming job - reads from Kafka, runs NLP, writes to Postgres
"""

import os
import json
import logging
import threading
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, udf, current_timestamp, to_timestamp, lit
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, TimestampType, MapType, ArrayType
)
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics for tracking performance
batches_processed = Counter('spark_batches_processed_total', 'Total batches processed')
batches_failed = Counter('spark_batches_failed_total', 'Total batches failed')
events_processed = Counter('spark_events_processed_total', 'Total events processed')
events_written_db = Counter('spark_events_written_db_total', 'Total events written to database')
batch_processing_duration = Histogram('spark_batch_processing_seconds', 'Time spent processing batches')
nlp_processing_duration = Histogram('nlp_processing_seconds', 'Time spent on NLP processing')
db_write_duration = Histogram('db_write_seconds', 'Time spent writing to database')
stream_lag = Gauge('spark_stream_lag_seconds', 'Stream processing lag')


class StreamProcessor:
    """Handles streaming from Kafka to Postgres with NLP processing"""

    def __init__(self):
        self.spark = self._create_spark_session()
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.postgres_url = self._get_postgres_url()

        # Metrics server runs in the background
        metrics_thread = threading.Thread(target=self._start_metrics_server, daemon=True)
        metrics_thread.start()

        logger.info("Stream processor initialized")

    def _start_metrics_server(self):
        """Start Prometheus metrics HTTP server"""
        try:
            start_http_server(8000)
            logger.info("Prometheus metrics server started on port 8000")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    def _create_spark_session(self) -> SparkSession:
        """Set up Spark with streaming configs"""
        spark = SparkSession.builder \
            .appName("RealTimeEventProcessor") \
            .config("spark.sql.streaming.checkpointLocation", "/tmp/checkpoint") \
            .config("spark.sql.shuffle.partitions", "4") \
            .config("spark.streaming.stopGracefullyOnShutdown", "true") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("WARN")
        logger.info("Spark session created")

        return spark

    def _get_postgres_url(self) -> str:
        """Build JDBC connection string"""
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        database = os.getenv('POSTGRES_DB', 'events_db')
        user = os.getenv('POSTGRES_USER', 'eventpipeline')
        password = os.getenv('POSTGRES_PASSWORD', 'pipeline_secret_2024')
        
        return f"jdbc:postgresql://{host}:{port}/{database}?user={user}&password={password}"
    
    def define_schema(self) -> StructType:
        """Schema for raw Kafka events"""
        return StructType([
            StructField("event_id", StringType(), False),
            StructField("source", StringType(), True),
            StructField("source_type", StringType(), True),
            StructField("title", StringType(), True),
            StructField("description", StringType(), True),
            StructField("content", StringType(), True),
            StructField("url", StringType(), True),
            StructField("published_at", StringType(), True),
            StructField("timestamp", StringType(), True),
            StructField("ingestion_time", StringType(), True),
            StructField("source_name", StringType(), True),
            StructField("subreddit", StringType(), True),
            StructField("score", IntegerType(), True),
            StructField("num_comments", IntegerType(), True),
            StructField("language", StringType(), True),
            StructField("seendate", StringType(), True),
        ])

    def define_output_schema(self) -> StructType:
        """Schema for processed events with NLP fields"""
        return StructType([
            StructField("event_id", StringType(), True),
            StructField("source", StringType(), True),
            StructField("source_type", StringType(), True),
            StructField("title", StringType(), True),
            StructField("description", StringType(), True),
            StructField("content", StringType(), True),
            StructField("url", StringType(), True),
            StructField("published_at", StringType(), True),
            StructField("timestamp", StringType(), True),
            StructField("category", StringType(), True),
            StructField("category_confidence", DoubleType(), True),
            StructField("crisis_level", StringType(), True),
            StructField("severity_score", DoubleType(), True),
            StructField("persons", StringType(), True),
            StructField("organizations", StringType(), True),
            StructField("locations", StringType(), True),
            StructField("word_count", IntegerType(), True),
        ])
    
    def read_from_kafka(self):
        """Stream events from Kafka topic"""
        logger.info(f"Reading from Kafka: {self.kafka_servers}")

        raw_stream = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_servers) \
            .option("subscribe", "raw-events") \
            .option("startingOffsets", "latest") \
            .option("maxOffsetsPerTrigger", "20") \
            .load()

        schema = self.define_schema()

        parsed_stream = raw_stream.select(
            from_json(col("value").cast("string"), schema).alias("data")
        ).select("data.*")

        logger.info("Kafka stream configured")
        return parsed_stream

    def process_nlp(self, batch_df, batch_id):
        """Process each batch with NLP and write to DB"""
        if batch_df.isEmpty():
            return

        with batch_processing_duration.time():
            record_count = batch_df.count()
            logger.info(f"Processing batch {batch_id} with {record_count} records")
            events_processed.inc(record_count)

            try:
                # Import here to avoid Spark serialization issues
                from nlp_module import get_nlp_processor

                events = [row.asDict() for row in batch_df.collect()]

                # Run NLP on the batch
                with nlp_processing_duration.time():
                    nlp_processor = get_nlp_processor()
                    processed_events = nlp_processor.batch_process(events)

                if processed_events:
                    # Flatten the nested NLP results for DB storage
                    flattened_events = []
                    for event in processed_events:
                        nlp_data = event.get('nlp_data', {})
                        entities = nlp_data.get('entities', {})

                        flattened = {
                            'event_id': event.get('event_id'),
                            'source': event.get('source'),
                            'source_type': event.get('source_type'),
                            'title': (event.get('title') or '')[:500],
                            'description': (event.get('description') or '')[:1000],
                            'content': (event.get('content') or '')[:2000],
                            'url': (event.get('url') or '')[:500],
                            'published_at': event.get('published_at'),
                            'timestamp': event.get('timestamp'),
                            'category': nlp_data.get('category', 'unknown'),
                            'category_confidence': nlp_data.get('category_confidence', 0.0),
                            'crisis_level': nlp_data.get('crisis_level', 'low'),
                            'severity_score': nlp_data.get('severity_score', 0.0),
                            'persons': json.dumps(entities.get('persons', [])),
                            'organizations': json.dumps(entities.get('organizations', [])),
                            'locations': json.dumps(entities.get('locations', [])),
                            'word_count': nlp_data.get('word_count', 0),
                        }
                        flattened_events.append(flattened)

                    output_schema = self.define_output_schema()
                    processed_df = self.spark.createDataFrame(flattened_events, schema=output_schema)

                    # Convert string timestamps to proper timestamp type
                    processed_df = processed_df \
                        .withColumn("published_at", to_timestamp(col("published_at"))) \
                        .withColumn("timestamp", to_timestamp(col("timestamp"))) \
                        .withColumn("processed_at", current_timestamp())

                    # Write to Postgres
                    with db_write_duration.time():
                        processed_df.write \
                            .format("jdbc") \
                            .option("url", self.postgres_url) \
                            .option("dbtable", "events") \
                            .option("driver", "org.postgresql.Driver") \
                            .mode("append") \
                            .save()

                        events_written_db.inc(len(flattened_events))
                        logger.info(f"Batch {batch_id}: Wrote {len(flattened_events)} events to database")

                batches_processed.inc()

            except Exception as e:
                logger.error(f"Failed to write batch {batch_id} to database: {e}")
                batches_failed.inc()
    
    def write_to_postgres(self, processed_stream):
        """Set up streaming write to Postgres"""
        query = processed_stream.writeStream \
            .foreachBatch(self.process_nlp) \
            .outputMode("append") \
            .trigger(processingTime='30 seconds') \
            .option("checkpointLocation", "/tmp/checkpoint/events") \
            .start()

        logger.info("Started writing to PostgreSQL")
        return query

    def run(self):
        """Start the streaming pipeline"""
        logger.info("Starting stream processor.")

        try:
            stream = self.read_from_kafka()

            query = self.write_to_postgres(stream)

            logger.info("Stream processor running. Press Ctrl+C to stop.")
            query.awaitTermination()

        except KeyboardInterrupt:
            logger.info("Shutting down stream processor.")
        except Exception as e:
            logger.error(f"Stream processor error: {e}")
            raise
        finally:
            self.spark.stop()
            logger.info("Stream processor stopped")


def main():
    processor = StreamProcessor()
    processor.run()


if __name__ == '__main__':
    main()