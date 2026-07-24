"""Per-frame quality gate. Catches what individually-valid records hide."""
from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series


class ElementSetFrame(pa.DataFrameModel):
    norad_cat_id: Series[int] = pa.Field(gt=0, unique=True)   # no dup objects per snapshot
    object_name: Series[str] = pa.Field(nullable=False)
    mean_motion: Series[float] = pa.Field(gt=0, le=20)        # revs/day; >20 is garbage
    eccentricity: Series[float] = pa.Field(ge=0, lt=1)        # bound orbits only
    inclination: Series[float] = pa.Field(ge=0, le=180)
    source: Series[str] = pa.Field(isin=["celestrak", "spacedevs", "spacetrack"])
    snapshot_hash: Series[str] = pa.Field(str_startswith="sha256:")

    class Config:
        strict = False   # extra provenance columns are allowed
        coerce = True


class NeoWsFrame(pa.DataFrameModel):
    """NASA NeoWs close approaches. dlt lands the shape; pandera asserts the
    physics dlt's schema contract cannot."""

    neo_reference_id: Series[str] = pa.Field(nullable=False)
    close_approach_date: Series[str] = pa.Field(nullable=False)
    miss_distance_km: Series[float] = pa.Field(gt=0)              # positive by definition
    rel_velocity_kms: Series[float] = pa.Field(ge=0, le=100)      # sane approach speed
    absolute_magnitude_h: Series[float] = pa.Field(ge=-5, le=40)  # asteroid H range
    source: Series[str] = pa.Field(isin=["nasa"])
    snapshot_hash: Series[str] = pa.Field(str_startswith="sha256:")

    class Config:
        strict = False
        coerce = True


class ArticleFrame(pa.DataFrameModel):
    """Spaceflight News articles: the text branch feeding the 8.1 RAG corpus."""

    id: Series[int] = pa.Field(gt=0, unique=True)
    title: Series[str] = pa.Field(nullable=False)
    url: Series[str] = pa.Field(nullable=False)
    news_site: Series[str] = pa.Field(nullable=False)
    published_at: Series[str] = pa.Field(nullable=False)
    source: Series[str] = pa.Field(isin=["spaceflightnews"])
    snapshot_hash: Series[str] = pa.Field(str_startswith="sha256:")

    class Config:
        strict = False
        coerce = True


class LaunchFrame(pa.DataFrameModel):
    """thespacedevs Launch Library 2 launches (events table feed)."""

    id: Series[str] = pa.Field(nullable=False, unique=True)   # LL2 uuid
    name: Series[str] = pa.Field(nullable=False)
    net: Series[str] = pa.Field(nullable=False)               # no-earlier-than timestamp
    status_name: Series[str] = pa.Field(nullable=False)
    provider: Series[str] = pa.Field(nullable=False)
    source: Series[str] = pa.Field(isin=["spacedevs"])
    snapshot_hash: Series[str] = pa.Field(str_startswith="sha256:")

    class Config:
        strict = False
        coerce = True
