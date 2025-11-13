FROM python:3.10-bookworm

WORKDIR /app

# Install system deps + Java for Spark
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy 
RUN python -m spacy download en_core_web_sm

COPY src/ ./src/
COPY configs/ ./configs/
COPY dashboard/ ./dashboard/

RUN mkdir -p logs

ENV PYTHONPATH=/app:$PYTHONPATH

CMD ["python", "src/kafka_producer.py"]