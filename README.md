# Realtime Event Pipeline & Alert System

An end-to-end project that builds a real-time event detection pipeline and interactive dashboard for analyzing news from multiple sources.
The pipeline connects to NewsAPI, Reddit, and GDELT, processes streams with NLP and machine learning, detects trending events, and sends alerts when critical patterns emerge.

### Tech Stack
- Python 3.10+
- APIs: NewsAPI, Reddit, GDELT
- Processing: Apache Kafka
- NLP/ML: spaCy, Hugging Face Transformers, scikit-learn (DBSCAN, TF-IDF)
- Storage: PostgreSQL
- Orchestration: Airflow, Docker Compose
- Visualization: Streamlit, Plotly
- Monitoring: Prometheus, Grafana
- Utilities: pytest, logging, dotenv

### Key Features
- Ingests real-time news data from NewsAPI, Reddit, and GDELT using Kafka.
- ETL Pipeline: A Kafka consumer enriches events with NLP (entity extraction, classification, severity scoring).
- Event Detection: Keyword spike detection, TF-IDF clustering, and DBSCAN grouping.
- Alerts: Slack and email notifications for critical events and daily summaries.
- Dashboard: Interactive Streamlit interface with KPIs, trend visualizations, and event filtering.
- Fully containerized with Docker Compose for easy deployment.

#### Setup
1. Clone the repository
2. Set your API keys:
    - Copy the example file: cp configs/credentials.env.example configs/credentials.env
    - Edit configs/credentials.env with your NewsAPI key, Reddit credentials, Slack webhook, etc.
3. Run the setup script:
    - chmod +x setup.sh
    - ./setup.sh
4. Start all services with Docker:
    - docker-compose up -d

### This starts:
- Kafka broker for event streaming
- Stream processor for NLP enrichment
- PostgreSQL for storage
- Airflow for orchestration
- Streamlit dashboard
- Prometheus and Grafana for monitoring

#### Running the Dashboard
- Once the pipeline is running, open your browser to http://localhost:8501
- The dashboard provides:
    - Real-time event feed with filtering and search
    - Top trending topics and keyword spikes
    - Crisis-level indicators and alert history
    - Interactive maps and time-series charts built with Plotly

#### Access Points
- Dashboard: http://localhost:8501
- Airflow: http://localhost:8081 
- Grafana: http://localhost:3000 

#### Configuration
All runtime settings are defined in:
- configs/credentials.env – API keys and secrets
- docker-compose.yml – Container resource allocation

#### Default storage:
- PostgreSQL: Events, detections, alerts, and keyword baselines

#### Testing
Run automated tests to validate NLP, detection, and storage logic:
- pytest -v

Test the alert system manually:
- python test_alerts.py

### License
MIT License
