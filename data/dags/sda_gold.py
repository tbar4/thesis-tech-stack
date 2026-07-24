"""Data-aware consumer DAG: rebuild gold when any snapshot asset updates."""
from __future__ import annotations

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Asset

TLE = Asset("sda://celestrak/element_sets")
NEOWS = Asset("sda://nasa/neows")
ARTICLES = Asset("sda://spaceflightnews/articles")
LAUNCHES = Asset("sda://spacedevs/launches")
GOLD = Asset("sda://gold/tables")

with DAG(
    dag_id="sda_gold",
    schedule=(TLE | NEOWS | ARTICLES | LAUNCHES),   # any-of, no clock
    catchup=False,
    max_active_runs=1,
    tags=["sda", "gold"],
) as dag:
    build = BashOperator(
        task_id="dbt_build",
        bash_command="cd /opt/project/data && uv run python -m sda_data.tasks.dbt_build",
        outlets=[GOLD],
    )
