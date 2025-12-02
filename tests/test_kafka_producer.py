"""
Tests for event identity in the producer
"""

import pytest
import subprocess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kafka_producer import normalize_url, make_event_id, url_event_id


@pytest.mark.parametrize('url,expected', [
    ('https://example.com/story', 'https://example.com/story'),
    ('https://www.example.com/story', 'https://example.com/story'),
    ('https://EXAMPLE.com/story', 'https://example.com/story'),
    ('https://example.com/story/', 'https://example.com/story'),
    ('https://example.com/story#section-2', 'https://example.com/story'),
    ('https://example.com:443/story', 'https://example.com/story'),
    ('  https://example.com/story  ', 'https://example.com/story'),
    ('https://example.com/story?utm_source=twitter', 'https://example.com/story'),
    ('https://example.com/story?fbclid=abc123', 'https://example.com/story'),
    ('https://example.com/story?id=7&utm_campaign=x', 'https://example.com/story?id=7'),
    ('https://example.com/story?b=2&a=1', 'https://example.com/story?a=1&b=2'),
    ('', ''),
])
def test_normalize_url(url, expected):
    assert normalize_url(url) == expected


def test_normalize_url_keeps_meaningful_query_params():
    normalized = normalize_url('https://example.com/article?id=123&page=2')

    assert 'id=123' in normalized
    assert 'page=2' in normalized


def test_normalize_url_distinguishes_different_articles():
    assert normalize_url('https://example.com/a') != normalize_url('https://example.com/b')


def test_event_id_is_stable_across_calls():
    url = 'https://example.com/quake'

    assert url_event_id('news', url) == url_event_id('news', url)


def test_event_id_is_stable_across_url_variants():
    """The same article fetched with different tracking decoration is one event"""
    plain = url_event_id('news', 'https://example.com/quake')
    decorated = url_event_id('news', 'https://www.example.com/quake/?utm_source=rss#top')

    assert plain == decorated


def test_event_id_differs_by_source():
    url = 'https://example.com/quake'

    assert url_event_id('news', url) != url_event_id('gdelt', url)


def test_event_id_differs_by_article():
    assert url_event_id('news', 'https://example.com/a') != url_event_id('news', 'https://example.com/b')


def test_event_id_carries_source_prefix():
    event_id = url_event_id('gdelt', 'https://example.com/quake')

    assert event_id.startswith('gdelt_')
    assert len(event_id) <= 255


def test_event_id_stable_across_processes():
    """Python's hash() is per-process randomized; the ID must not be"""
    script = (
        'import sys; sys.path.insert(0, %r);'
        'from kafka_producer import url_event_id;'
        'print(url_event_id("news", "https://example.com/quake"))'
        % os.path.join(os.path.dirname(__file__), '..', 'src')
    )

    runs = {
        subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True, text=True, check=True,
            env={**os.environ, 'PYTHONHASHSEED': str(seed)},
        ).stdout.strip()
        for seed in (0, 1, 2)
    }

    assert len(runs) == 1


def test_make_event_id_uses_source_native_identifier():
    assert make_event_id('reddit', 'abc123') == make_event_id('reddit', 'abc123')
    assert make_event_id('reddit', 'abc123') != make_event_id('reddit', 'xyz789')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
