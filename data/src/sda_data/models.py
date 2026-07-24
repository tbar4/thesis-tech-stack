"""Per-record ingest contract. Provenance is stamped on EVERY record."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Provenance(BaseModel):
    source: str            # "celestrak"
    fetch_time: datetime   # when we pulled the raw snapshot
    snapshot_hash: str     # content address of the raw bytes


class ElementSetAtEpoch(BaseModel):
    """One GP/TLE orbital state at its epoch, normalized across feeds."""

    norad_cat_id: int = Field(gt=0)
    object_name: str
    epoch: datetime                            # the element-set epoch (NOT fetch_time)
    mean_motion: float = Field(gt=0)           # revs/day
    eccentricity: float = Field(ge=0, lt=1)
    inclination: float = Field(ge=0, le=180)   # degrees
    ra_of_asc_node: float = Field(ge=0, le=360)
    arg_of_pericenter: float = Field(ge=0, le=360)
    mean_anomaly: float = Field(ge=0, le=360)
    bstar: float | None = None
    prov: Provenance


def from_celestrak_gp(rec: dict, prov: Provenance) -> ElementSetAtEpoch:
    """Map one CelesTrak GP JSON record onto the common schema."""
    return ElementSetAtEpoch(
        norad_cat_id=int(rec["NORAD_CAT_ID"]),
        object_name=rec["OBJECT_NAME"],
        epoch=datetime.fromisoformat(rec["EPOCH"]),
        mean_motion=float(rec["MEAN_MOTION"]),
        eccentricity=float(rec["ECCENTRICITY"]),
        inclination=float(rec["INCLINATION"]),
        ra_of_asc_node=float(rec["RA_OF_ASC_NODE"]),
        arg_of_pericenter=float(rec["ARG_OF_PERICENTER"]),
        mean_anomaly=float(rec["MEAN_ANOMALY"]),
        bstar=float(rec["BSTAR"]) if rec.get("BSTAR") is not None else None,
        prov=prov,
    )
