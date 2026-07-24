"""Serving layer: Delta appends via delta-rs. One writer per table (enforced
by DAG design: each table is written by exactly one task, max_active_runs=1).
This layer is rebuildable from the frozen snapshots; it is NEVER the
provenance authority (book ch. 3.9 serving-layer invariants)."""
from __future__ import annotations

import pandas as pd
from deltalake import DeltaTable, write_deltalake

from sda_data import config


def table_uri(name: str) -> str:
    """Local path by default; the delta tier root moves wholesale to s3:// later."""
    return str(config.delta_root() / name)


def append(name: str, df: pd.DataFrame) -> int:
    """Append one gated frame to a named table; returns the new table version."""
    uri = table_uri(name)
    opts = config.s3_storage_options() or None
    write_deltalake(uri, df, mode="append", storage_options=opts)
    return DeltaTable(uri, storage_options=opts).version()


def element_sets_uri() -> str:
    return table_uri("element_sets")


def append_element_sets(df: pd.DataFrame) -> int:
    return append("element_sets", df)
