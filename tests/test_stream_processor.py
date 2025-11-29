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
        self.inserted_per_call = inserted_per_call

    def insert_events(self, events):
        self.calls.append(events)
        if self.inserted_per_call is not None:
            return self.inserted_per_call
        return len(events)


@pytest.fixture
def consumer():
    return EventConsumer(nlp_processor=FakeNLP(), storage=FakeStorage())


def test_handle_batch_runs_nlp_then_inserts(consumer):
    events = [{'event_id': 'a', 'title': 'one'}, {'event_id': 'b', 'title': 'two'}]

    inserted = consumer.handle_batch(events)

    assert inserted == 2
    assert consumer._nlp.batches == [events]
    stored = consumer._storage.calls[0]
    assert all('nlp_data' in e for e in stored)


def test_handle_batch_reports_duplicates():
    consumer = EventConsumer(nlp_processor=FakeNLP(), storage=FakeStorage(inserted_per_call=1))
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
