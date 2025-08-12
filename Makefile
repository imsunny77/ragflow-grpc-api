.PHONY: help install dev-install proto test lint format clean docker-build docker-up docker-down run-server run-client

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

dev-install: ## Install with dev dependencies
	uv sync --group dev

proto: ## Generate protobuf files
	uv run python -m grpc_tools.protoc --python_out=src --grpc_python_out=src --proto_path=src/proto src/proto/ragflow.proto

test: ## Run tests
	uv run pytest tests/ -v

lint: ## Run linting
	uv run ruff check src/ tests/

format: ## Format code
	uv run ruff format src/ tests/

clean: ## Clean generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -f src/ragflow_pb2.py src/ragflow_pb2_grpc.py

docker-build: ## Build docker image
	docker-compose build

docker-up: ## Start services
	docker-compose up -d

docker-down: ## Stop services
	docker-compose down

docker-test: ## Run tests in docker
	docker-compose --profile test up --build --abort-on-container-exit

run-server: proto ## Run gRPC server
	uv run python -m src.server

run-client: ## Run example client
	uv run python examples/example_usage.py
