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
    if [ ! -f "configs/credentials.env" ]; then
        cp configs/credentials.env.example configs/credentials.env
        echo "Created configs/credentials.env - edit with your API keys"
    else
        echo "Credentials file exists"
    fi
}

create_directories() {
    mkdir -p logs data checkpoint
}

pull_images() {
    docker-compose pull
}

build_images() {
    docker-compose build
}

start_services() {
    docker-compose up -d zookeeper kafka postgres redis
    sleep 30
    docker-compose up -d
}

check_health() {
    sleep 10
    docker-compose ps
}

show_access_info() {
    echo ""
    echo "Setup complete"
    echo ""
    echo "Dashboard:  http://localhost:8501"
    echo "Airflow:    http://localhost:8081 (admin/admin)"
    echo "Grafana:    http://localhost:3000 (admin/admin)"
    echo "Prometheus: http://localhost:9090"
    echo ""
    echo "Edit configs/credentials.env with your API keys, then:"
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
