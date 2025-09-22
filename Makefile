# MLOps Demo Makefile
# Convenient commands for managing the MLOps demo environment

.PHONY: help setup start stop restart status demo test clean docs

# Default target
help:
	@echo "MLOps Demo - Available Commands:"
	@echo "================================"
	@echo "  setup     - Complete environment setup"
	@echo "  start     - Start all services"
	@echo "  stop      - Stop all services"
	@echo "  restart   - Restart all services"
	@echo "  status    - Check service status"
	@echo "  demo      - Run interactive demo"
	@echo "  test      - Run test suite"
	@echo "  lint      - Run code linting"
	@echo "  train     - Train all models"
	@echo "  monitor   - Run monitoring checks"
	@echo "  clean     - Clean up containers and volumes"
	@echo "  logs      - Show service logs"
	@echo "  shell     - Open shell in API container"
	@echo "  docs      - Generate documentation"
	@echo "  summary   - Show project summary"

# Environment setup
setup:
	@echo "🏗️  Setting up MLOps environment..."
	python3 scripts/quick_start.py

# Service management
start:
	@echo "🚀 Starting services..."
	docker-compose up -d

stop:
	@echo "⏹️  Stopping services..."
	docker-compose down

restart:
	@echo "🔄 Restarting services..."
	docker-compose restart

status:
	@echo "📊 Service status:"
	docker-compose ps

# Demo and testing
demo:
	@echo "🎭 Running interactive demo..."
	python3 scripts/demo_script.py

test:
	@echo "🧪 Running tests..."
	docker-compose exec api pytest tests/ -v

lint:
	@echo "🔍 Running linters..."
	docker-compose exec api black --check .
	docker-compose exec api flake8 .

# Data and model operations
datasets:
	@echo "📊 Creating datasets..."
	docker-compose exec api python data/create_datasets.py

train:
	@echo "🤖 Training models..."
	docker-compose exec api python scripts/train_all_models.py

# Monitoring and maintenance
monitor:
	@echo "📈 Running monitoring checks..."
	docker-compose exec api python monitoring/check_model_performance.py
	docker-compose exec api python monitoring/check_data_drift.py

drift:
	@echo "🌊 Checking data drift..."
	docker-compose exec api python monitoring/drift_detection.py

# Utilities
logs:
	@echo "📜 Showing service logs..."
	docker-compose logs -f

logs-api:
	@echo "📜 Showing API logs..."
	docker-compose logs -f api

logs-mlflow:
	@echo "📜 Showing MLflow logs..."
	docker-compose logs -f mlflow

shell:
	@echo "🐚 Opening shell in API container..."
	docker-compose exec api bash

shell-root:
	@echo "🐚 Opening root shell in API container..."
	docker-compose exec --user root api bash

# Development
dev:
	@echo "🔧 Starting development environment..."
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

dev-logs:
	@echo "📜 Showing development logs..."
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f

# Documentation
docs:
	@echo "📚 Opening documentation..."
	@echo "Available documentation:"
	@echo "  • Setup Guide: docs/guides/setup_guide.md"
	@echo "  • Presentations: docs/presentations/"
	@echo "  • API Docs: http://localhost:8000/docs"

summary:
	@echo "📋 Generating project summary..."
	python3 scripts/project_summary.py

# Cleanup
clean:
	@echo "🧹 Cleaning up..."
	docker-compose down -v
	docker system prune -f

clean-all:
	@echo "🧹 Deep cleaning..."
	docker-compose down -v --remove-orphans
	docker system prune -af
	docker volume prune -f

# Health checks
health:
	@echo "🔍 Checking service health..."
	@curl -s http://localhost:8000/health | python3 -m json.tool || echo "API not responding"
	@curl -s http://localhost:5000 > /dev/null && echo "✅ MLflow: UP" || echo "❌ MLflow: DOWN"
	@curl -s http://localhost:3000 > /dev/null && echo "✅ Grafana: UP" || echo "❌ Grafana: DOWN"
	@curl -s http://localhost:9090 > /dev/null && echo "✅ Prometheus: UP" || echo "❌ Prometheus: DOWN"

# Build commands
build:
	@echo "🔨 Building containers..."
	docker-compose build

build-no-cache:
	@echo "🔨 Building containers (no cache)..."
	docker-compose build --no-cache

# Data operations
dvc-repro:
	@echo "🔄 Reproducing DVC pipeline..."
	docker-compose exec api dvc repro

dvc-dag:
	@echo "📊 Showing DVC pipeline DAG..."
	docker-compose exec api dvc dag

# MLflow operations
mlflow-ui:
	@echo "🔗 MLflow UI: http://localhost:5000"

grafana-ui:
	@echo "🔗 Grafana UI: http://localhost:3000 (admin/admin)"

prometheus-ui:
	@echo "🔗 Prometheus UI: http://localhost:9090"

api-docs:
	@echo "🔗 API Documentation: http://localhost:8000/docs"

# Backup and restore
backup:
	@echo "💾 Creating backup..."
	docker-compose exec api tar -czf /tmp/mlops_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz data/ models/ mlflow.db

# Quick validation
validate:
	@echo "✅ Running quick validation..."
	@make health
	@echo "🧪 Running smoke tests..."
	@docker-compose exec api python -c "import requests; print('✅ API OK' if requests.get('http://localhost:8000/health').status_code == 200 else '❌ API Failed')"

# CI/CD simulation
ci:
	@echo "🔄 Simulating CI/CD pipeline..."
	@make lint
	@make test
	@make train
	@make validate

# Show useful URLs
urls:
	@echo "🔗 Service URLs:"
	@echo "  • API Documentation: http://localhost:8000/docs"
	@echo "  • MLflow UI:         http://localhost:5000"
	@echo "  • Grafana Dashboard: http://localhost:3000"
	@echo "  • Prometheus:        http://localhost:9090"
	@echo "  • Jupyter Notebook:  http://localhost:8888"