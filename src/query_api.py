"""
Query API - serves grounded answers over HTTP.

This exists because of where the models live. Answering means embedding the
question and running a language model, which needs torch and transformers.
Those ship in the NLP image and deliberately not in the dashboard image,
which stays lean precisely because it carries no model stack. So the models
stay where they already are and the dashboard asks over HTTP.

It runs as its own service off the same image rather than inside the
consumer: that poll loop is single-threaded by design, and serving requests
from it would block ingest behind a query.
"""

import os
import logging

from flask import Flask, jsonify, request
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 5
MAX_LIMIT = 20
MAX_QUESTION_CHARS = 500

queries_total = Counter('query_api_requests_total', 'Questions asked')
queries_failed = Counter('query_api_errors_total', 'Questions that could not be answered')
queries_rejected = Counter('query_api_rejected_total', 'Questions rejected as malformed')
query_duration = Histogram('query_api_seconds', 'Time spent answering a question')


def read_limit(raw) -> int:
    """Clamp the caller's source count into a range worth generating over"""
    try:
        limit = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_LIMIT

    return max(1, min(limit, MAX_LIMIT))


def create_app(builder=None) -> Flask:
    """Build the app, with the answer builder injectable for tests"""
    app = Flask(__name__)
    app.config['_builder'] = builder

    def answer_builder():
        if app.config.get('_builder') is None:
            from answer import get_answer_builder
            app.config['_builder'] = get_answer_builder()
        return app.config['_builder']

    @app.get('/health')
    def health():
        return jsonify({'status': 'ok'})

    @app.get('/metrics')
    def metrics():
        return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

    @app.get('/answer')
    def answer():
        question = (request.args.get('q') or '').strip()

        if not question:
            queries_rejected.inc()
            return jsonify({'error': 'a question is required, as ?q='}), 400

        if len(question) > MAX_QUESTION_CHARS:
            # The embedding model truncates far below this, so a longer string
            # buys nothing and only widens what one request can push through.
            queries_rejected.inc()
            return jsonify({
                'error': f'question must be {MAX_QUESTION_CHARS} characters or fewer'
            }), 400

        queries_total.inc()

        try:
            with query_duration.time():
                result = answer_builder().answer(
                    question, limit=read_limit(request.args.get('limit'))
                )
        except Exception:
            # The question is fine; something behind it is not. Saying so beats
            # an empty answer, which would read as "nothing was found".
            queries_failed.inc()
            logger.exception(f"Failed to answer {question!r}")
            return jsonify({'error': 'the question could not be answered right now'}), 503

        return jsonify(result.as_dict())

    return app


def main():
    from waitress import serve

    port = int(os.getenv('QUERY_API_PORT', '8100'))
    app = create_app()

    # Load both models before the first request rather than during it, so a
    # cold start shows up in the logs instead of as one very slow question.
    try:
        from answer import get_answer_builder
        builder = get_answer_builder()
        builder.embedder.embed_query('warm up')
        _ = builder.generator
        logger.info("Models ready")
    except Exception as e:
        logger.error(f"Could not warm the models, the first query will pay for it: {e}")

    logger.info(f"Query API listening on {port}")
    serve(app, host='0.0.0.0', port=port, threads=4)


if __name__ == '__main__':
    main()
