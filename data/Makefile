.PHONY: test up down doctor backup mcp

test:
	uv run pytest

doctor:
	uv run python -m sda_data.doctor

up:
	docker compose -f docker-compose.airflow.yml up airflow-init
	docker compose -f docker-compose.airflow.yml up -d

down:
	docker compose -f docker-compose.airflow.yml down

backup:
	./scripts/backup_offsite.sh

mcp:
	uv run python -m sda_data.mcp_server
