"""Storage tiers and S3 settings. Everything is env-driven and read at call
time (functions, not import-time constants) so tests and containers can
redirect tiers without import-order games.

Tier contract (book ch. 0.5): local paths under SDA_DATA_HOME are the working
tier; MinIO on the NAS (S3_ENDPOINT_URL) is the durable tier. The live tier is
the redistribution boundary, in code.
"""
from __future__ import annotations

import os
from pathlib import Path

# Sources allowed onto the shippable raw tier. spacetrack/nasa deliberately absent.
REDISTRIBUTABLE = {"celestrak", "spacedevs"}


def data_home() -> Path:
    default = Path(__file__).resolve().parents[2]
    return Path(os.environ.get("SDA_DATA_HOME", default))


def raw_root() -> Path:
    """Immutable raw bytes, DVC-tracked, shippable."""
    return data_home() / "data" / "raw"


def snap_root() -> Path:
    """Normalized Parquet, derived from raw_root()."""
    return data_home() / "data" / "snapshots"


def live_root() -> Path:
    """EMBARGOED: space-track etc. git-ignored, never DVC-tracked."""
    return data_home() / "data" / "live"


def delta_root() -> Path:
    """Serving layer: Delta tables. Rebuildable, never the provenance authority."""
    return data_home() / "data" / "delta"


def s3_storage_options() -> dict[str, str]:
    """delta-rs style storage options for the MinIO endpoint, from env.
    Empty dict when unconfigured, so local-path workflows need no setup."""
    mapping = {
        "AWS_ENDPOINT_URL": os.environ.get("S3_ENDPOINT_URL", ""),
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    }
    return {k: v for k, v in mapping.items() if v}
