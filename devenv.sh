#!/bin/bash

# Create a _bash_history file if it doesn't exist
touch ".devcontainer/_bash_history"

CONTAINER_NAME="python-container"

is_container_running() {
    docker ps | grep "$CONTAINER_NAME" >/dev/null
}

stop_container() {
    if is_container_running; then
        echo "Stopping the container..."
        docker stop "$CONTAINER_NAME"
    else
        echo "Container is not running."
    fi
}

start_container() {
    docker-compose up -d --build
}

run_container() {
    if is_container_running; then
        echo "Environment already running - connecting..."
        if command -v winpty >/dev/null 2>&1; then
            winpty docker exec -it "$CONTAINER_NAME" bash
        else
            docker exec -it "$CONTAINER_NAME" bash
        fi
    else
        echo "Container is not running. Please start the container first."
    fi
}

# Check the first argument
case "$1" in
    start)
        start_container
        ;;
    run)
        run_container
        ;;
    stop)
        stop_container
        ;;
    *)
        echo "Usage: $0 {start|run|stop}"
        exit 1
esac
