"""
Query API client - the dashboard's side of the HTTP boundary.

Kept out of the dashboard module so it can be tested without importing
Streamlit, and out of the answer layer because the whole point of the
boundary is that the caller carries no model stack.

Every failure becomes one exception carrying a sentence fit to show a reader.
A dashboard that renders a traceback has told them nothing actionable, and one
that renders an empty result has told them something false: that no reporting
matched, when in fact nothing was ever asked.
"""

import os
import logging
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = 'http://query-api:8100'
# Generation on CPU is slow; this is sized for the model, not the network.
DEFAULT_TIMEOUT = 120


class QueryError(Exception):
    """The question could not be put to the service, or it could not answer"""


def base_url() -> str:
    return os.getenv('QUERY_API_URL', DEFAULT_BASE_URL).rstrip('/')


def ask(question: str, limit: int = 5, url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT, session=None) -> Dict:
    """Put a question to the query service and return its answer payload"""
    question = (question or '').strip()
    if not question:
        raise QueryError("Ask a question first.")

    endpoint = f"{(url or base_url()).rstrip('/')}/answer"
    http = session or requests

    try:
        response = http.get(endpoint, params={'q': question, 'limit': limit},
                            timeout=timeout)
    except requests.Timeout:
        raise QueryError(
            f"The query service did not respond within {timeout} seconds."
        )
    except requests.RequestException as e:
        logger.warning(f"Could not reach the query service at {endpoint}: {e}")
        raise QueryError("Could not reach the query service.")

    if response.status_code >= 400:
        raise QueryError(_message_from(response))

    try:
        return response.json()
    except ValueError:
        raise QueryError("The query service returned something unreadable.")


def _message_from(response) -> str:
    """Prefer the service's own explanation over a bare status code"""
    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if isinstance(payload, dict) and payload.get('error'):
        return str(payload['error'])

    return f"The query service returned {response.status_code}."


def health(url: Optional[str] = None, timeout: int = 3, session=None) -> bool:
    """Whether the service is up, for telling a reader why nothing works"""
    http = session or requests
    try:
        response = http.get(f"{(url or base_url()).rstrip('/')}/health",
                            timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


def format_source(index: int, source: Dict):
    """One cited report, rendered as (heading, detail) for display.

    Lives here rather than in the dashboard because it reads the shape of the
    API payload, which is this module's business, and because anything in the
    dashboard module needs a running Streamlit to import.
    """
    title = source.get('title') or '(untitled)'
    url = source.get('url')
    heading = f"{index}. [{title}]({url})" if url else f"{index}. {title}"

    detail = [source.get('source') or 'unknown']

    published = (source.get('published_at') or '')[:10]
    if published:
        detail.append(published)

    similarity = source.get('similarity')
    if isinstance(similarity, (int, float)):
        detail.append(f"similarity {similarity:.2f}")

    return heading, ' · '.join(detail)
