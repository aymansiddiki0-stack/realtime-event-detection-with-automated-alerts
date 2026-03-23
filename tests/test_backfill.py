"""
Tests for embedding backfill
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backfill import backfill, DEFAULT_BATCH_SIZE


class FakeStorage:
    """Holds events, some with vectors, and serves the ones still missing"""

    def __init__(self, unembedded=0, fail_update=False):
        self.pending = [
            {'event_id': f'e{i}', 'title': f'title {i}',
             'description': '', 'content': ''}
            for i in range(unembedded)
        ]
        self.updated = {}
        self.fail_update = fail_update
        self.select_calls = []

    def get_events_without_embeddings(self, limit=200):
        self.select_calls.append(limit)
        return self.pending[:limit]

    def update_embeddings(self, vectors):
        if self.fail_update:
            raise RuntimeError('database unreachable')
        self.updated.update(vectors)
        done = set(vectors)
        self.pending = [e for e in self.pending if e['event_id'] not in done]
        return len(vectors)


class FakeEmbedder:
    def __init__(self, explode=False):
        self.explode = explode
        self.batches = []

    def embed_events(self, events):
        if self.explode:
            raise RuntimeError('model unavailable')
        self.batches.append(len(events))
        return [[0.1] * 4 for _ in events]


def test_every_unembedded_event_is_embedded():
    storage = FakeStorage(unembedded=250)

    totals = backfill(storage, FakeEmbedder(), batch_size=100)

    assert totals['embedded'] == 250
    assert len(storage.updated) == 250


def test_work_is_done_in_batches_rather_than_all_at_once():
    """A single query for tens of thousands of rows is not a plan"""
    storage = FakeStorage(unembedded=250)
    embedder = FakeEmbedder()

    backfill(storage, embedder, batch_size=100)

    assert embedder.batches == [100, 100, 50]


def test_an_empty_backlog_does_no_work():
    storage = FakeStorage(unembedded=0)
    embedder = FakeEmbedder()

    totals = backfill(storage, embedder)

    assert totals == {'embedded': 0, 'failed': 0, 'batches': 0}
    assert embedder.batches == []


def test_the_run_can_be_bounded_for_a_trial():
    storage = FakeStorage(unembedded=500)

    totals = backfill(storage, FakeEmbedder(), batch_size=100, max_batches=2)

    assert totals['embedded'] == 200
    assert totals['batches'] == 2


def test_an_embedding_failure_stops_rather_than_spinning():
    """The same rows would be selected again, so continuing would loop"""
    storage = FakeStorage(unembedded=300)

    totals = backfill(storage, FakeEmbedder(explode=True), batch_size=100)

    assert totals['embedded'] == 0
    assert totals['failed'] == 100
    assert totals['batches'] == 0


def test_a_write_failure_stops_too():
    storage = FakeStorage(unembedded=300, fail_update=True)

    totals = backfill(storage, FakeEmbedder(), batch_size=100)

    assert totals['embedded'] == 0
    assert totals['failed'] == 100


def test_progress_survives_a_failure_partway_through():
    """Whatever committed before the failure stays committed"""
    storage = FakeStorage(unembedded=300)
    embedder = FakeEmbedder()

    backfill(storage, embedder, batch_size=100, max_batches=1)
    assert len(storage.updated) == 100

    # A later run picks up only what is still missing
    backfill(storage, embedder, batch_size=100)
    assert len(storage.updated) == 300


def test_the_batch_size_reaches_the_query():
    storage = FakeStorage(unembedded=10)

    backfill(storage, FakeEmbedder(), batch_size=25)

    assert storage.select_calls[0] == 25


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
