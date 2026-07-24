"""Query the Delta serving layer with DuckDB. The table arrives as Arrow via
delta-rs (no network extension install), so this works identically against a
local path or MinIO."""
from __future__ import annotations

import duckdb
import pandas as pd
from deltalake import DeltaTable

from sda_data import config
from sda_data.silver import element_sets_uri


def latest_element_sets() -> pd.DataFrame:
    """One row per object: the element set from the newest fetch."""
    arrow = DeltaTable(
        element_sets_uri(), storage_options=config.s3_storage_options() or None
    ).to_pyarrow_table()
    con = duckdb.connect()
    con.register("element_sets", arrow)
    return con.sql(
        """
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, row_number() OVER (
                PARTITION BY norad_cat_id ORDER BY fetch_time DESC
            ) AS rn
            FROM element_sets
        ) WHERE rn = 1
        """
    ).df()
