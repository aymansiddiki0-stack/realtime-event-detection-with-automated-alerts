FROM python:3.10-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy 
RUN python -m spacy download en_core_web_sm

COPY src/ ./src/
COPY sql/ ./sql/
COPY dashboard/ ./dashboard/

RUN mkdir -p logs

ENV PYTHONPATH=/app:$PYTHONPATH

CMD ["python", "src/kafka_producer.py"]