"""Data-aware consumer DAG: screen for conjunctions when elements refresh."""
from __future__ import annotations

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Asset

TLE = Asset("sda://celestrak/element_sets")
CLOSE_APPROACHES = Asset("sda://gold/close_approaches")

with DAG(
    dag_id="sda_conjunctions",
    schedule=[TLE],                 # fires on the asset, not a timer
    catchup=False,
    max_active_runs=1,              # one writer for close_approaches
    tags=["sda", "conjunctions"],
) as dag:
    build = BashOperator(
        task_id="build_conjunctions",
        bash_command="cd /opt/project/data && "
                     "uv run python -m sda_data.tasks.build_conjunctions",
        outlets=[CLOSE_APPROACHES],
    )
