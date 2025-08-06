.PHONY: help install dev-install proto test lint format clean docker-build docker-up docker-down run-server run-client

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

dev-install: ## Install with dev dependencies
	uv sync --dev

proto: ## Generate protobuf files
	uv run python -m grpc_tools.protoc --python_out=src --grpc_python_out=src --proto_path=src/proto src/proto/ragflow.proto

test: ## Run tests
	uv run pytest tests/ -v

lint: ## Run linting
	uv run ruff check src/ tests/