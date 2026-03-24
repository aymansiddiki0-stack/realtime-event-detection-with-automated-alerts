# Realtime Event Pipeline & Alert System

An end-to-end pipeline that ingests real-time news from NewsAPI, Reddit, and GDELT, enriches it with NLP, detects emerging events, and serves both a monitoring dashboard and a grounded question-answering layer over everything it has indexed.

## Architecture

```
NewsAPI / Reddit / GDELT
        │
        ▼
   Kafka producer  ──►  Kafka  ──►  Stream processor (NLP enrichment + embedding)
                                            │
                                            ▼
                              PostgreSQL + pgvector (events, detections,
                              alerts, keyword baselines, embeddings)
                                            │
                        ┌───────────────────┼───────────────────┐
                        ▼                   ▼                   ▼
                Airflow (detection,   Query API (semantic     Streamlit dashboard
                summaries, cleanup)   search + local LLM)      + Ask panel
                        │                   │
                        ▼                   ▼
              Slack / email alerts   Grounded answers with
                                      cited sources
```

Prometheus and Grafana observe every service throughout.

## Tech Stack
- **Python 3.10+**
- **Ingestion:** NewsAPI, Reddit (PRAW), GDELT, via Apache Kafka
- **NLP:** spaCy (entity extraction), Hugging Face Transformers (zero-shot classification, severity scoring)
- **Semantic search:** sentence-transformers (`all-MiniLM-L6-v2`) for embeddings, PostgreSQL with `pgvector` for storage and cosine-similarity retrieval
- **Grounded Q&A:** a local instruct model (`Qwen/Qwen2.5-1.5B-Instruct`) running in-process — no external API, no per-query cost, no credential
- **Event detection:** keyword-spike detection against persisted historical baselines, TF-IDF + DBSCAN geographic/topic clustering
- **Storage:** PostgreSQL (`pgvector/pgvector:pg15`)
- **Orchestration:** Airflow, Docker Compose
- **Interface:** Streamlit, Plotly, Flask (query API, served by waitress)
- **Monitoring:** Prometheus, Grafana
- **Testing:** pytest (139 tests across 11 modules)

## Key Features
- **Ingestion** — a Kafka producer polls each source on its own schedule with rate-limit backoff, deduplicating articles by a normalized-URL hash so re-fetching the same headline never produces a duplicate row.
- **NLP enrichment** — the stream consumer classifies category and severity, extracts named entities, and embeds each event for semantic search, all in the same batch that persists it.
- **Event detection** — keyword mentions are compared against a persisted, windowed median rather than a same-run baseline, so a spike means "unusual for this keyword," not "the first thing we happened to see."
- **Alerting** — Slack and email notifications for critical events and daily summaries, gated by configurable severity thresholds.
- **Semantic Q&A** — ask a question in the dashboard and get an answer generated only from the reporting actually retrieved for it, with every source cited, linked, and scored by similarity. If nothing relevant is indexed, it says so rather than answering from the model's own training data.
- **Dashboard** — KPIs, trend visualizations, event filtering, and the ask panel, all in one Streamlit app.
- **Monitoring** — every service exports Prometheus metrics; Grafana ships with a provisioned dashboard, not an empty one.
- **Fully containerized**, with each service built from only the dependencies it actually needs (the dashboard image carries no ML stack; the NLP/query-API image does).

## Setup
1. Clone the repository.
2. Set your credentials:
   - `cp .env.example .env`
   - Edit `.env` with your database password, source API keys, and alert channel credentials.
   - Sources and alert channels left blank fall back to mock data / are simply disabled — you can start with none of them configured.
3. Run the setup script:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```
4. Start everything:
   ```
   docker compose up -d
   ```

### This starts
- Zookeeper + Kafka for event streaming
- PostgreSQL (with `pgvector`) for storage and semantic search
- Kafka producer and stream processor (NLP enrichment + embedding)
- Query API — semantic retrieval and grounded generation
- Streamlit dashboard
- Airflow for scheduled detection, summaries, and cleanup
- Prometheus and Grafana for monitoring

First startup downloads the embedding and generation models into a named volume (`huggingface-cache`), so the query API takes longer to become healthy the first time only.

## Running It
Open `http://localhost:8501` once the stack is up. The dashboard provides:
- A real-time event feed with filtering and search
- Top trending topics and keyword spikes
- Crisis-level indicators and alert history
- An **Ask** panel — type a question about indexed events and get a grounded answer with cited, linked sources

Generation runs on local CPU, so an answer typically takes on the order of two minutes; the panel says so rather than looking frozen.

### Access Points
| Service | URL |
| --- | --- |
| Dashboard | http://localhost:8501 |
| Query API | http://localhost:8100 (`/answer?q=...`, `/health`, `/metrics`) |
| Airflow | http://localhost:8081 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

## Configuration
All runtime settings live in:
- `.env` — credentials and runtime settings, read by both Compose and the application services
- `docker-compose.yml` — service topology and resource allocation

The generation and embedding models can be overridden without a code change via the `ANSWER_MODEL` and `EMBEDDING_MODEL` environment variables.

### Default storage
PostgreSQL holds events, detections, alerts, keyword baselines, and each event's embedding vector.

## Testing
Install the development dependencies, then run the suite:
```
pip install -r requirements-dev.txt
pytest -v
```

Test the alert system manually:
```
python test_alerts.py
```

## Known Limitations
- Generation runs on CPU and takes roughly two minutes per question — fine for a demonstration, not for interactive back-and-forth. A GPU would bring this down to seconds.
- The system answers from whatever it retrieves, including weak matches; it does not abstain on low-confidence retrieval, and there is no automated evaluation harness scoring answer quality.
- Named-entity recognition occasionally mislabels a person as a location, since it's driven by a small, general-purpose spaCy model.

## License
MIT License
