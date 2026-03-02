"""
Event embeddings - turns event text into vectors for semantic retrieval.

The model is small enough (~80MB, 384 dimensions) to run on CPU for every
ingested event, so embedding happens inline in the consumer that already
holds the NLP stack rather than in a separate service or scheduled job.

Vectors are L2-normalised at generation time, which lets the query path use
pgvector's <=> cosine-distance operator without renormalising per query.
"""

import os
import logging
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DIMENSIONS = 384

# Longer inputs are truncated by the model anyway; cutting here avoids paying
# to tokenise text that will be discarded.
MAX_CONTENT_CHARS = 1000


def build_embedding_text(event: Dict) -> str:
    """Assemble the text an event is represented by.

    Title and description carry most of the signal in a news item; content is
    included but capped, since the model's context window truncates it well
    before the full article body is consumed.
    """
    parts = []

    for field, limit in (('title', None), ('description', None),
                         ('content', MAX_CONTENT_CHARS)):
        value = (event.get(field) or '').strip()
        if not value:
            continue
        parts.append(value[:limit] if limit else value)

    return '\n'.join(parts)


def to_pgvector(vector: Sequence[float]) -> str:
    """Render a vector in the literal text form pgvector accepts: '[1,2,3]'"""
    return '[' + ','.join(repr(float(v)) for v in vector) + ']'


class EventEmbedder:
    """Loads the sentence-transformer model and embeds event text"""

    def __init__(self, model_name: Optional[str] = None, model=None):
        self.model_name = model_name or os.getenv('EMBEDDING_MODEL', DEFAULT_MODEL)
        self._model = model

    @property
    def model(self):
        # Loaded on first use so importing this module stays cheap; the weights
        # take a few seconds to initialise.
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of texts, normalised for cosine similarity"""
        if not texts:
            return []

        vectors = self.model.encode(
            list(texts),
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        return [list(map(float, vector)) for vector in vectors]

    def embed_events(self, events: Sequence[Dict]) -> List[List[float]]:
        """Embed events in order, one vector per event"""
        if not events:
            return []

        return self.embed_texts([build_embedding_text(e) for e in events])

    def embed_query(self, query: str) -> List[float]:
        """Embed a single search query"""
        return self.embed_texts([query])[0]


_embedder = None


def get_embedder() -> EventEmbedder:
    """Get the shared embedder instance"""
    global _embedder
    if _embedder is None:
        _embedder = EventEmbedder()
    return _embedder
