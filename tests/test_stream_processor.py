"""
Tests for the Kafka event consumer
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stream_processor import EventConsumer


class FakeNLP:
    def __init__(self):
        self.batches = []

    def batch_process(self, events):
        self.batches.append(events)
        for event in events:
            event['nlp_data'] = {'category': 'other', 'category_confidence': 0.5}
        return events


class FakeStorage:
    def __init__(self, inserted_per_call=None):
        self.calls = []
        self.embeddings = []
        self.inserted_per_call = inserted_per_call

    def insert_events(self, events, embeddings=None):
        self.calls.append(events)
        self.embeddings.append(embeddings)
        if self.inserted_per_call is not None:
            return self.inserted_per_call
        return len(events)


class FakeEmbedder:
    def __init__(self, explode=False):
        self.explode = explode
        self.batches = []

    def embed_events(self, events):
        if self.explode:
            raise RuntimeError('model unavailable')
        self.batches.append(events)
        return [[0.1] * 4 for _ in events]


def make_consumer(storage=None, embedder=None, **storage_kw):
    return EventConsumer(
        nlp_processor=FakeNLP(),
        storage=storage if storage is not None else FakeStorage(**storage_kw),
        embedder=embedder if embedder is not None else FakeEmbedder(),
    )


@pytest.fixture
def consumer():
    return make_consumer()


def test_handle_batch_runs_nlp_then_inserts(consumer):
    events = [{'event_id': 'a', 'title': 'one'}, {'event_id': 'b', 'title': 'two'}]

    inserted = consumer.handle_batch(events)

    assert inserted == 2
    assert consumer._nlp.batches == [events]
    stored = consumer._storage.calls[0]
    assert all('nlp_data' in e for e in stored)


def test_handle_batch_reports_duplicates():
    consumer = make_consumer(inserted_per_call=1)
    events = [{'event_id': 'a'}, {'event_id': 'a'}, {'event_id': 'b'}]

    inserted = consumer.handle_batch(events)

    # storage said only 1 row landed; the other 2 were conflict-skipped
    assert inserted == 1


def test_decode_valid_message(consumer):
    raw = b'{"event_id": "x", "title": "hello"}'

    assert consumer._decode(raw) == {'event_id': 'x', 'title': 'hello'}


def test_decode_invalid_json_returns_none(consumer):
    assert consumer._decode(b'not json at all') is None


def test_decode_invalid_bytes_returns_none(consumer):
    assert consumer._decode(b'\xff\xfe\x00broken') is None


# --- embedding at ingest ----------------------------------------------------

def test_events_are_embedded_in_the_batch_that_stores_them(consumer):
    """Without this an event is stored but not semantically retrievable"""
    events = [{'event_id': 'a', 'title': 'one'}, {'event_id': 'b', 'title': 'two'}]

    consumer.handle_batch(events)

    assert consumer._embedder.batches == [events]
    assert len(consumer._storage.embeddings[0]) == 2


def test_an_embedding_failure_does_not_lose_the_batch():
    """The events matter more than the vectors; they store without them"""
    consumer = make_consumer(embedder=FakeEmbedder(explode=True))

    inserted = consumer.handle_batch([{'event_id': 'a', 'title': 'one'}])

    assert inserted == 1
    assert consumer._storage.calls
    assert consumer._storage.embeddings[0] is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
