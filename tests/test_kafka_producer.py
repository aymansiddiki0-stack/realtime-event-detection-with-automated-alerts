"""
Tests for event identity in the producer
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kafka_producer import normalize_url


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
