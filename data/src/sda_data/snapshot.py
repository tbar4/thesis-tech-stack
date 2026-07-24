"""Immutable, content-addressed raw snapshots. The filename IS the hash
(book eq. 9.1). Restricted sources land on the embargoed live tier and never
touch the shippable, DVC-tracked raw tier."""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from sda_data import config


def content_address(raw: bytes) -> str:
    """addr(s) = sha256(bytes(s))."""
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def write_raw_snapshot(
    source: str, kind: str, raw: bytes, fetch_time: datetime
) -> tuple[Path, str]:
    """Write bytes ONCE, named by their own hash."""
    addr = content_address(raw)
    digest = addr.split(":", 1)[1]
    root = config.raw_root() if source in config.REDISTRIBUTABLE else config.live_root()
    out = root / source / kind / fetch_time.strftime("%Y-%m-%d") / f"{digest}.raw"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():  # identical bytes hash identically: write-once is free
        out.write_bytes(raw)
    return out, addr
