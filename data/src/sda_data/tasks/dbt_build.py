"""Run the dbt gold build. Thin: dbt owns the SQL and its tests; this module
only locates the project, guarantees the gold dir exists, and shells out.

Standalone:
    uv run python -m sda_data.tasks.dbt_build
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from sda_data import config

PROJECT_DIR = Path(__file__).resolve().parents[3] / "dbt"


def run() -> None:
    gold_dir = config.data_home() / "data" / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("SDA_DATA_HOME", str(config.data_home()))
    subprocess.run(
        ["dbt", "build", "--project-dir", str(PROJECT_DIR),
         "--profiles-dir", str(PROJECT_DIR)],
        check=True,
        env=env,
    )


if __name__ == "__main__":
    run()
