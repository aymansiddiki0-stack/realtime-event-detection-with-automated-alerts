#!/bin/bash

set -e

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker not found. Install from https://docs.docker.com/get-docker/"
        exit 1
    fi
    echo "Docker found"
}

check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        echo "Error: Docker Compose not found. Install from https://docs.docker.com/compose/install/"
        exit 1
    fi
    echo "Docker Compose found"
}

setup_credentials() {
    if [ -f ".env" ]; then
        echo "Environment file exists"
        return
    fi

    cp .env.example .env

    # Generated rather than shipped, so no deployment starts with a known key.
    local fernet secret
    fernet=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null || openssl rand -base64 32)
    secret=$(python3 -c "import secrets; print(secrets.token_hex(32))" 2>/dev/null || openssl rand -hex 32)

    sed -i.bak "s|^AIRFLOW_FERNET_KEY=.*|AIRFLOW_FERNET_KEY=${fernet}|" .env
    sed -i.bak "s|^AIRFLOW_SECRET_KEY=.*|AIRFLOW_SECRET_KEY=${secret}|" .env
    rm -f .env.bak

    echo "Created .env - set the change_me passwords and your API keys before starting"
}

create_directories() {
    mkdir -p logs
}

pull_images() {
    docker-compose pull
}

build_images() {
    docker-compose build
}

start_services() {
    # Compose waits on healthchecks, so services come up in dependency order.
    docker-compose up -d
}

check_health() {
    docker-compose ps
}

show_access_info() {
    echo ""
    echo "Setup complete"
    echo ""
    echo "Dashboard:  http://localhost:8501"
    echo "Airflow:    http://localhost:8081 (admin / AIRFLOW_ADMIN_PASSWORD from .env)"
    echo "Grafana:    http://localhost:3000 (credentials from .env)"
    echo "Prometheus: http://localhost:9090"
    echo ""
    echo "Edit .env with your API keys, then:"
    echo "docker-compose restart kafka-producer"
}

main() {
    check_docker
    check_docker_compose
    setup_credentials
    create_directories
    pull_images
    build_images
    start_services
    check_health
    show_access_info
}

main
