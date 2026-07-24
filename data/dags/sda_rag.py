"""Data-aware consumer DAG: rebuild the RAG index when articles refresh."""
from __future__ import annotations

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Asset

ARTICLES = Asset("sda://spaceflightnews/articles")
RAG_INDEX = Asset("sda://rag/index")

with DAG(
    dag_id="sda_rag",
    schedule=[ARTICLES],
    catchup=False,
    max_active_runs=1,
    tags=["sda", "rag"],
) as dag:
    build = BashOperator(
        task_id="build_rag_index",
        bash_command="cd /opt/project/data && "
                     "uv run python -m sda_data.tasks.build_rag_index",
        outlets=[RAG_INDEX],
    )
