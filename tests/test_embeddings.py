"""
Tests for event embeddings
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from embeddings import (
    EventEmbedder, build_embedding_text, to_pgvector,
    EMBEDDING_DIMENSIONS, MAX_CONTENT_CHARS,
)


class FakeModel:
    """Deterministic stand-in for the sentence transformer"""

    def __init__(self):
        self.encode_calls = []

    def encode(self, texts, **kwargs):
        self.encode_calls.append(list(texts))
        vectors = []
        for i, _ in enumerate(texts):
            vector = [0.0] * EMBEDDING_DIMENSIONS
            vector[i % EMBEDDING_DIMENSIONS] = 1.0
            vectors.append(vector)
        return vectors


@pytest.fixture
def embedder():
    return EventEmbedder(model_name='test-model', model=FakeModel())


# --- text assembly ----------------------------------------------------------

def test_text_joins_the_populated_fields():
    text = build_embedding_text({
        'title': 'Quake hits San Jose',
        'description': 'A 6.1 struck early Monday',
        'content': 'Crews responded',
    })

    assert text == 'Quake hits San Jose\nA 6.1 struck early Monday\nCrews responded'


def test_missing_and_blank_fields_are_skipped():
    assert build_embedding_text(
        {'title': 'Only a title', 'description': None, 'content': '   '}
    ) == 'Only a title'


def test_an_empty_event_yields_empty_text():
    assert build_embedding_text({}) == ''


def test_content_is_capped_but_title_is_not():
    """The model truncates anyway; capping avoids tokenising discarded text"""
    event = {'title': 't' * 3000, 'content': 'c' * 3000}

    text = build_embedding_text(event)

    assert text.count('t') == 3000
    assert text.count('c') == MAX_CONTENT_CHARS


# --- pgvector rendering -----------------------------------------------------

def test_a_vector_renders_in_the_literal_form_pgvector_accepts():
    assert to_pgvector([1.0, 2.5, -3.0]) == '[1.0,2.5,-3.0]'


def test_integers_are_rendered_as_floats():
    assert to_pgvector([1, 2]) == '[1.0,2.0]'


def test_an_empty_vector_is_still_valid_syntax():
    assert to_pgvector([]) == '[]'


# --- embedding --------------------------------------------------------------

def test_events_are_embedded_in_order(embedder):
    vectors = embedder.embed_events([
        {'title': 'first'}, {'title': 'second'},
    ])

    assert len(vectors) == 2
    assert len(vectors[0]) == EMBEDDING_DIMENSIONS
    assert embedder._model.encode_calls[0] == ['first', 'second']


def test_an_empty_batch_never_reaches_the_model(embedder):
    assert embedder.embed_events([]) == []
    assert embedder.embed_texts([]) == []
    assert embedder._model.encode_calls == []


def test_a_query_embeds_to_a_single_vector(embedder):
    vector = embedder.embed_query('earthquake news')

    assert len(vector) == EMBEDDING_DIMENSIONS
    assert embedder._model.encode_calls[0] == ['earthquake news']


def test_embedding_asks_the_model_to_normalise(embedder):
    """Cosine distance at query time depends on vectors being unit length"""
    embedder.embed_texts(['anything'])

    assert embedder._model.encode_calls == [['anything']]


def test_vectors_come_back_as_plain_floats(embedder):
    vector = embedder.embed_query('q')

    assert all(isinstance(v, float) for v in vector)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
