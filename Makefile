# Makefile for energy forecasting pipeline

.PHONY: help install train predict api dashboard test lint format clean docker-build docker-up docker-down

help:
	@echo "Energy Forecasting Pipeline - Available Commands"
	@echo "=================================================="
	@echo "make install        - Install dependencies"
	@echo "make train          - Train the model"
	@echo "make predict        - Make predictions"
	@echo "make api            - Run FastAPI server"
	@echo "make dashboard      - Run Streamlit dashboard"
	@echo "make test           - Run tests"
	@echo "make lint           - Lint code"
	@echo "make format         - Format code"
	@echo "make clean          - Clean cache files"
	@echo "make docker-build   - Build Docker image"
	@echo "make docker-up      - Start Docker services"
	@echo "make docker-down    - Stop Docker services"

install:
	pip install -r requirements.txt

train:
	python main.py train --data-path /home/moukouba/equilibrium/model_ready.parquet

predict:
	python main.py predict \
		--data-path /home/moukouba/equilibrium/model_ready.parquet \
		--target-col da_energy_aeci_lmpexpost_ac

api:
	python main.py api --host 0.0.0.0 --port 8000

dashboard:
	python main.py dashboard

test:
	pytest tests/ -v

lint:
	pylint src/
	flake8 src/

format:
	black src/
	isort src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

docker-build:
	docker build -t energy-forecast:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down -v
	docker rmi energy-forecast:latest
