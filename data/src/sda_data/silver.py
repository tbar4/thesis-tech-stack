"""Serving layer: Delta appends via delta-rs. One writer per table (enforced
by DAG design: each table is written by exactly one task, max_active_runs=1).
This layer is rebuildable from the frozen snapshots; it is NEVER the
provenance authority (book ch. 3.9 serving-layer invariants)."""
from __future__ import annotations

import pandas as pd
from deltalake import DeltaTable, write_deltalake

from sda_data import config


def element_sets_uri() -> str:
    """Local path by default; set SDA_DELTA_URI-style env later for s3://."""
    return str(config.delta_root() / "element_sets")


def append_element_sets(df: pd.DataFrame) -> int:
    """Append one gated frame; returns the new table version."""
    uri = element_sets_uri()
    write_deltalake(
        uri,
        df,
        mode="append",
        storage_options=config.s3_storage_options() or None,
    )
    return DeltaTable(uri, storage_options=config.s3_storage_options() or None).version()
