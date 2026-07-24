"""Screen the latest element sets for close approaches; append to Delta.

Standalone:
    uv run python -m sda_data.tasks.build_conjunctions --window-hours 24
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from sda_data import silver
from sda_data.conjunctions import screen
from sda_data.query import latest_element_sets


def run(window_hours: float = 24.0, miss_threshold_km: float = 25.0) -> int:
    df = latest_element_sets()
    start = datetime.now(timezone.utc)
    hits = screen(df, start=start,
                  window_hours=window_hours, miss_threshold_km=miss_threshold_km)
    hits["screened_at"] = start.isoformat()
    if hits.empty:
        print("conjunctions: no close approaches under threshold")
        return 0
    version = silver.append("close_approaches", hits)
    print(f"conjunctions: {len(hits)} close approaches  delta=v{version}")
    return len(hits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-hours", type=float, default=24.0)
    ap.add_argument("--miss-km", type=float, default=25.0)
    args = ap.parse_args()
    run(window_hours=args.window_hours, miss_threshold_km=args.miss_km)


if __name__ == "__main__":
    main()
