"""
Tests for connection pooling behaviour in StorageManager
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import storage_manager
from storage_manager import StorageManager


class FakeConnection:
    def __init__(self):
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class FakePool:
    """Stands in for psycopg2's ThreadedConnectionPool"""

    def __init__(self, minconn, maxconn, **kwargs):
        self.minconn = minconn
        self.maxconn = maxconn
        self.kwargs = kwargs
        self.closed = False
        self.connections = [FakeConnection() for _ in range(maxconn)]
        self.checked_out = []

    def getconn(self):
        if not self.connections:
            raise RuntimeError("pool exhausted")
        conn = self.connections.pop()
        self.checked_out.append(conn)
        return conn

    def putconn(self, conn):
        self.checked_out.remove(conn)
        self.connections.append(conn)

    def closeall(self):
        self.closed = True


@pytest.fixture
def storage(monkeypatch):
    monkeypatch.setattr(storage_manager.pool, 'ThreadedConnectionPool', FakePool)
    monkeypatch.setattr(StorageManager, '_ensure_tables_exist', lambda self: None)
    monkeypatch.setenv('POSTGRES_POOL_SIZE', '3')
    return StorageManager()


def test_pool_is_created_with_configured_size(storage):
    assert storage._pool.maxconn == 3
    assert storage._pool.minconn == 1


def test_connection_is_returned_to_the_pool(storage):
    available_before = len(storage._pool.connections)

    with storage.get_connection() as conn:
        assert conn in storage._pool.checked_out

    assert storage._pool.checked_out == []
    assert len(storage._pool.connections) == available_before


def test_successful_block_commits(storage):
    with storage.get_connection() as conn:
        pass

    assert conn.commits == 1
    assert conn.rollbacks == 0


def test_failing_block_rolls_back_and_returns_connection(storage):
    with pytest.raises(ValueError):
        with storage.get_connection() as conn:
            raise ValueError("query blew up")

    assert conn.rollbacks == 1
    assert conn.commits == 0
    assert storage._pool.checked_out == []


def test_connections_are_reused_across_operations(storage):
    seen = []
    for _ in range(5):
        with storage.get_connection() as conn:
            seen.append(id(conn))

    # Sequential borrows return the same connection rather than new ones
    assert len(set(seen)) == 1


def test_concurrent_borrows_get_distinct_connections(storage):
    with storage.get_connection() as first:
        with storage.get_connection() as second:
            assert first is not second
            assert len(storage._pool.checked_out) == 2

    assert storage._pool.checked_out == []


def test_close_releases_the_pool(storage):
    storage.close()

    assert storage._pool.closed is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
