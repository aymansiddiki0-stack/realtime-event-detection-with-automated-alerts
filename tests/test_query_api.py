"""
Tests for the query API and its client
"""

import pytest
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from query_api import create_app, read_limit, DEFAULT_LIMIT, MAX_LIMIT, MAX_QUESTION_CHARS
from query_client import ask, health, base_url, format_source, QueryError, DEFAULT_BASE_URL


@dataclass
class FakeAnswer:
    question: str = 'q'
    text: str = 'an answer'
    sources: List[Dict] = field(default_factory=list)

    def as_dict(self):
        return {'question': self.question, 'answer': self.text,
                'sources': self.sources}


class FakeBuilder:
    def __init__(self, answer=None, explode=False):
        self._answer = answer or FakeAnswer()
        self.explode = explode
        self.calls = []

    def answer(self, question, limit=5):
        self.calls.append({'question': question, 'limit': limit})
        if self.explode:
            raise RuntimeError('database unreachable')
        return self._answer


def client_for(builder=None):
    builder = builder or FakeBuilder()
    app = create_app(builder=builder)
    app.config.update(TESTING=True)
    return app.test_client(), builder


# --- limits -----------------------------------------------------------------

def test_a_missing_or_junk_limit_falls_back_to_the_default():
    assert read_limit(None) == DEFAULT_LIMIT
    assert read_limit('') == DEFAULT_LIMIT
    assert read_limit('not a number') == DEFAULT_LIMIT


def test_a_limit_is_clamped_to_a_range_worth_generating_over():
    assert read_limit('1') == 1
    assert read_limit('0') == 1
    assert read_limit('-5') == 1
    assert read_limit(str(MAX_LIMIT + 100)) == MAX_LIMIT


# --- answering --------------------------------------------------------------

def test_a_question_is_answered_with_its_sources():
    answer = FakeAnswer(text='Indiana saw record flooding',
                        sources=[{'event_id': 'a', 'title': 'Flooding'}])
    client, builder = client_for(FakeBuilder(answer))

    response = client.get('/answer?q=flooding+in+Indiana')

    assert response.status_code == 200
    body = response.get_json()
    assert body['answer'] == 'Indiana saw record flooding'
    assert body['sources'][0]['event_id'] == 'a'
    assert builder.calls[0]['question'] == 'flooding in Indiana'


def test_the_limit_reaches_the_answer_builder():
    client, builder = client_for()

    client.get('/answer?q=news&limit=3')

    assert builder.calls[0]['limit'] == 3


def test_a_missing_question_is_rejected():
    client, builder = client_for()

    response = client.get('/answer')

    assert response.status_code == 400
    assert 'error' in response.get_json()
    assert builder.calls == []


def test_whitespace_is_not_a_question():
    client, builder = client_for()

    assert client.get('/answer?q=%20%20').status_code == 400
    assert builder.calls == []


def test_an_overlong_question_is_rejected_before_the_model_sees_it():
    client, builder = client_for()

    response = client.get('/answer?q=' + 'a' * (MAX_QUESTION_CHARS + 1))

    assert response.status_code == 400
    assert builder.calls == []


def test_a_question_at_the_length_limit_is_accepted():
    client, builder = client_for()

    assert client.get('/answer?q=' + 'a' * MAX_QUESTION_CHARS).status_code == 200
    assert len(builder.calls) == 1


# --- failures ---------------------------------------------------------------

def test_a_backend_failure_is_reported_not_disguised_as_no_results():
    """An empty answer would read as 'nothing found', a different fact"""
    client, _ = client_for(FakeBuilder(explode=True))

    response = client.get('/answer?q=news')

    assert response.status_code == 503
    assert 'error' in response.get_json()


def test_a_backend_failure_does_not_leak_its_internals():
    client, _ = client_for(FakeBuilder(explode=True))

    assert 'database unreachable' not in client.get('/answer?q=news').get_json()['error']


# --- operational endpoints --------------------------------------------------

def test_health_reports_without_touching_the_models():
    client, builder = client_for()

    response = client.get('/health')

    assert response.status_code == 200
    assert response.get_json()['status'] == 'ok'
    assert builder.calls == []


def test_metrics_are_exposed_for_scraping():
    client, _ = client_for()

    client.get('/answer?q=news')

    assert 'query_api_requests_total' in client.get('/metrics').data.decode()


# --- client -----------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code=200, payload=None, unreadable=False):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self._unreadable = unreadable

    def json(self):
        if self._unreadable:
            raise ValueError('not json')
        return self._payload


class FakeSession:
    def __init__(self, response=None, raises=None):
        self.response = response or FakeResponse()
        self.raises = raises
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({'url': url, 'params': params, 'timeout': timeout})
        if self.raises:
            raise self.raises
        return self.response


def test_the_service_address_defaults_to_the_compose_service(monkeypatch):
    monkeypatch.delenv('QUERY_API_URL', raising=False)

    assert base_url() == DEFAULT_BASE_URL


def test_the_service_address_is_configurable(monkeypatch):
    monkeypatch.setenv('QUERY_API_URL', 'http://localhost:8100/')

    assert base_url() == 'http://localhost:8100'


def test_a_question_reaches_the_answer_endpoint():
    session = FakeSession(FakeResponse(payload={'answer': 'text'}))

    payload = ask('flooding', limit=3, url='http://api:8100', session=session)

    assert payload == {'answer': 'text'}
    assert session.calls[0]['url'] == 'http://api:8100/answer'
    assert session.calls[0]['params'] == {'q': 'flooding', 'limit': 3}


def test_an_empty_question_never_leaves_the_client():
    session = FakeSession()

    with pytest.raises(QueryError):
        ask('   ', url='http://api:8100', session=session)

    assert session.calls == []


def test_an_unreachable_service_is_reported_as_such():
    """Not as an empty answer, which would claim no reporting matched"""
    session = FakeSession(raises=requests.ConnectionError('refused'))

    with pytest.raises(QueryError) as caught:
        ask('news', url='http://api:8100', session=session)

    assert 'reach' in str(caught.value).lower()


def test_a_timeout_says_how_long_was_waited():
    session = FakeSession(raises=requests.Timeout())

    with pytest.raises(QueryError) as caught:
        ask('news', url='http://api:8100', timeout=7, session=session)

    assert '7 seconds' in str(caught.value)


def test_the_services_own_explanation_is_preferred():
    session = FakeSession(FakeResponse(400, {'error': 'a question is required, as ?q='}))

    with pytest.raises(QueryError) as caught:
        ask('news', url='http://api:8100', session=session)

    assert 'a question is required' in str(caught.value)


def test_health_is_false_when_the_service_is_down():
    session = FakeSession(raises=requests.ConnectionError('refused'))

    assert health(url='http://api:8100', session=session) is False


def test_health_is_true_when_the_service_answers():
    assert health(url='http://api:8100', session=FakeSession(FakeResponse(200))) is True


# --- source formatting ------------------------------------------------------

def test_a_source_links_to_the_report_it_cites():
    heading, _ = format_source(1, {'title': 'Flooding', 'url': 'https://x/a'})

    assert heading == '1. [Flooding](https://x/a)'


def test_a_source_without_a_link_is_still_readable():
    heading, _ = format_source(2, {'title': 'Flooding'})

    assert heading == '2. Flooding'


def test_an_untitled_source_is_not_rendered_blank():
    assert '(untitled)' in format_source(1, {})[0]


def test_the_detail_line_carries_outlet_date_and_similarity():
    _, detail = format_source(1, {
        'title': 't', 'source': 'newsapi',
        'published_at': '2026-08-17T12:47:00', 'similarity': 0.6123,
    })

    assert 'newsapi' in detail
    assert '2026-08-17' in detail
    assert 'similarity 0.61' in detail


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
