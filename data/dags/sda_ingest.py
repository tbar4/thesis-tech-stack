"""Thin producer DAG: schedule the standalone task modules, declare assets.
No fetching, parsing, or validating happens HERE; it happens in sda_data.tasks."""
from __future__ import annotations

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Asset

TLE_SNAPSHOT = Asset("sda://celestrak/element_sets")

RUN = "cd /opt/project/data && uv run python -m sda_data.tasks.{mod}"

with DAG(
    dag_id="sda_ingest",
    schedule="0 */6 * * *",            # clock drives the producer
    start_date=pendulum.datetime(2026, 7, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,                 # single node + one-writer-per-table
    default_args={"retries": 3, "retry_delay": pendulum.duration(minutes=5)},
    tags=["sda", "ingest"],
) as dag:
    fetch_tle = BashOperator(
        task_id="celestrak_tle",
        bash_command=RUN.format(mod="celestrak_tle --group active"),
        outlets=[TLE_SNAPSHOT],
    )
