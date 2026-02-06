"""
Tests for per-source polling intervals and rate-limit backoff
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kafka_producer import (
    SourceSchedule, RateLimited, parse_retry_after, MAX_BACKOFF_SECONDS
)


class FakeResponse:
    def __init__(self, headers=None):
        self.headers = headers or {}


def test_schedule_is_ready_initially():
    schedule = SourceSchedule('gdelt', 900)

    assert schedule.ready(now=0.0)


def test_schedule_waits_its_interval_after_success():
    schedule = SourceSchedule('gdelt', 900)

    schedule.record_success(now=100.0)

    assert not schedule.ready(now=999.0)
    assert schedule.ready(now=1000.0)


def test_rate_limit_backs_off_at_least_one_interval():
    schedule = SourceSchedule('gdelt', 900)

    delay = schedule.record_rate_limit(now=0.0)

    assert delay >= 900
    assert not schedule.ready(now=899.0)


def test_repeated_rate_limits_grow_the_delay():
    schedule = SourceSchedule('gdelt', 900)

    first = schedule.record_rate_limit(now=0.0)
    second = schedule.record_rate_limit(now=0.0)
    third = schedule.record_rate_limit(now=0.0)

    assert second > first
    assert third > second


def test_backoff_is_capped():
    schedule = SourceSchedule('gdelt', 900)

    for _ in range(20):
        schedule.record_rate_limit(now=0.0)

    assert schedule.backoff <= MAX_BACKOFF_SECONDS


def test_success_clears_backoff():
    schedule = SourceSchedule('gdelt', 900)
    schedule.record_rate_limit(now=0.0)
    schedule.record_rate_limit(now=0.0)

    schedule.record_success(now=10000.0)

    assert schedule.backoff == 0.0
    assert schedule.ready(now=10000.0 + 900)


def test_retry_after_header_is_honoured():
    schedule = SourceSchedule('gdelt', 900)

    delay = schedule.record_rate_limit(now=0.0, retry_after=30.0)

    # Jitter only ever extends the wait, and never to a full backoff interval
    assert 30.0 <= delay < 900


def test_parse_retry_after_seconds():
    assert parse_retry_after(FakeResponse({'Retry-After': '120'})) == 120.0


def test_parse_retry_after_http_date():
    response = FakeResponse({'Retry-After': 'Wed, 21 Oct 2099 07:28:00 GMT'})

    assert parse_retry_after(response) > 0


def test_parse_retry_after_absent_or_junk():
    assert parse_retry_after(FakeResponse()) is None
    assert parse_retry_after(FakeResponse({'Retry-After': 'soon'})) is None


def test_rate_limited_carries_source_and_delay():
    error = RateLimited('gdelt', retry_after=45.0)

    assert error.source == 'gdelt'
    assert error.retry_after == 45.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
