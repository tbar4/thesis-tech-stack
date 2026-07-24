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
    source: Series[str] = pa.Field(isin=["celestrak", "spacedevs"])
    snapshot_hash: Series[str] = pa.Field(str_startswith="sha256:")

    class Config:
        strict = False   # extra provenance columns are allowed
        coerce = True
