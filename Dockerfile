FROM python:3.10-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY sql/ ./sql/

RUN mkdir -p logs

ENV PYTHONPATH=/app:$PYTHONPATH

CMD ["python", "src/kafka_producer.py"]
