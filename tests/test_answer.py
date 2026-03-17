"""
Tests for grounded question answering
"""

import pytest
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from answer import (
    AnswerBuilder, Answer, build_prompt, format_sources, DEFAULT_SOURCES,
)


class FakeStorage:
    def __init__(self, rows=None):
        self.rows = rows if rows is not None else []
        self.calls = []

    def search_events_by_vector(self, vector, limit=5):
        self.calls.append({'vector': vector, 'limit': limit})
        return self.rows


class FakeEmbedder:
    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return [0.1] * 8


class FakeGenerator:
    """Stands in for the transformers pipeline"""

    def __init__(self, text='A grounded answer.', explode=False):
        self.text = text
        self.explode = explode
        self.calls = []

    def __call__(self, messages, **kwargs):
        if self.explode:
            raise RuntimeError('model unavailable')
        self.calls.append({'messages': messages, 'kwargs': kwargs})
        return [{'generated_text': self.text}]


def row(event_id='e1', title='Flooding in Indiana', **kw):
    base = {
        'event_id': event_id, 'title': title,
        'description': 'Heavy rain caused record flooding.',
        'url': 'https://example.com/a', 'source': 'newsapi',
        'published_at': datetime(2026, 8, 17, 12, 0),
        'category': 'climate', 'crisis_level': 'high',
        'severity_score': 0.8, 'similarity': 0.6123456,
    }
    base.update(kw)
    return base


def builder_for(rows, generator=None):
    return AnswerBuilder(
        storage=FakeStorage(rows),
        embedder=FakeEmbedder(),
        generator=generator or FakeGenerator(),
    )


# --- prompt construction ----------------------------------------------------

def test_sources_are_numbered_so_the_model_can_refer_to_them():
    text = format_sources([row(title='First'), row(title='Second')])

    assert '[1] First' in text
    assert '[2] Second' in text


def test_a_source_carries_its_outlet_and_date():
    text = format_sources([row()])

    assert 'newsapi' in text
    assert '2026-08-17' in text


def test_missing_metadata_does_not_produce_empty_furniture():
    text = format_sources([{'title': 'Bare', 'description': '', 'source': ''}])

    assert '[1] Bare' in text
    assert '·' not in text


def test_the_prompt_separates_instructions_from_article_text():
    """News content is external input; it must not read as instructions"""
    messages = build_prompt('what happened', [row()])

    system, user = messages[0], messages[1]
    assert system['role'] == 'system'
    assert 'ignored' in system['content']
    assert '<reports>' in user['content']
    assert '</reports>' in user['content']


def test_the_question_appears_outside_the_reports_block():
    messages = build_prompt('what happened in Indiana', [row()])

    user = messages[1]['content']
    assert user.index('</reports>') < user.index('what happened in Indiana')


# --- answering --------------------------------------------------------------

def test_an_answer_carries_the_reporting_it_used():
    answer = builder_for([row(event_id='a'), row(event_id='b')]).answer('q')

    assert answer.text == 'A grounded answer.'
    assert [s['event_id'] for s in answer.sources] == ['a', 'b']


def test_the_question_is_what_gets_embedded():
    b = builder_for([row()])

    b.answer('flooding in Indiana')

    assert b._embedder.queries == ['flooding in Indiana']


def test_the_source_limit_reaches_the_search():
    b = builder_for([row()])

    b.answer('q', limit=3)

    assert b._storage.calls[0]['limit'] == 3


def test_the_default_limit_is_used_when_none_is_given():
    b = builder_for([row()])

    b.answer('q')

    assert b._storage.calls[0]['limit'] == DEFAULT_SOURCES


def test_an_empty_index_is_reported_rather_than_answered():
    """Nothing retrieved means nothing to ground on, so the model is skipped"""
    generator = FakeGenerator()
    answer = builder_for([], generator=generator).answer('q')

    assert answer.sources == []
    assert 'no indexed reporting' in answer.text
    assert generator.calls == []


def test_generation_is_deterministic():
    """Two identical questions should not produce different answers"""
    b = builder_for([row()])

    b.answer('q')

    assert b._generator.calls[0]['kwargs']['do_sample'] is False


def test_only_the_new_text_is_returned_not_the_prompt():
    b = builder_for([row()])

    b.answer('q')

    assert b._generator.calls[0]['kwargs']['return_full_text'] is False


def test_surrounding_whitespace_is_stripped_from_the_question():
    b = builder_for([row()])

    answer = b.answer('   flooding   ')

    assert answer.question == 'flooding'
    assert b._embedder.queries == ['flooding']


# --- serialization ----------------------------------------------------------

def test_an_answer_serializes_for_an_api_response():
    payload = builder_for([row()]).answer('q').as_dict()

    assert payload['question'] == 'q'
    assert payload['answer'] == 'A grounded answer.'
    assert len(payload['sources']) == 1


def test_a_serialized_source_is_traceable_back_to_its_report():
    source = builder_for([row()]).answer('q').as_dict()['sources'][0]

    assert source['event_id'] == 'e1'
    assert source['url'] == 'https://example.com/a'
    assert source['source'] == 'newsapi'
    assert source['published_at'] == '2026-08-17T12:00:00'


def test_similarity_is_rounded_rather_than_shown_to_full_precision():
    source = builder_for([row()]).answer('q').as_dict()['sources'][0]

    assert source['similarity'] == 0.6123


def test_a_missing_timestamp_serializes_as_null_not_a_crash():
    source = builder_for([row(published_at=None)]).answer('q').as_dict()['sources'][0]

    assert source['published_at'] is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
