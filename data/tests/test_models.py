from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from sda_data.models import ElementSetAtEpoch, Provenance, from_celestrak_gp

PROV = Provenance(
    source="celestrak",
    fetch_time=datetime(2026, 7, 24, tzinfo=timezone.utc),
    snapshot_hash="sha256:" + "0" * 64,
)

GP_RECORD = {
    "NORAD_CAT_ID": "25544",
    "OBJECT_NAME": "ISS (ZARYA)",
    "EPOCH": "2026-07-24T06:00:00",
    "MEAN_MOTION": "15.50",
    "ECCENTRICITY": "0.0003",
    "INCLINATION": "51.64",
    "RA_OF_ASC_NODE": "120.0",
    "ARG_OF_PERICENTER": "30.0",
    "MEAN_ANOMALY": "300.0",
    "BSTAR": "0.00012",
}


def test_maps_celestrak_gp_record():
    rec = from_celestrak_gp(GP_RECORD, PROV)
    assert rec.norad_cat_id == 25544
    assert rec.object_name == "ISS (ZARYA)"
    assert rec.mean_motion == pytest.approx(15.50)
    assert rec.prov.snapshot_hash.startswith("sha256:")


def test_bstar_is_optional():
    rec = from_celestrak_gp({**GP_RECORD, "BSTAR": None}, PROV)
    assert rec.bstar is None


def test_unbound_eccentricity_rejected():
    with pytest.raises(ValidationError):
        ElementSetAtEpoch(
            norad_cat_id=1,
            object_name="X",
            epoch=datetime(2026, 1, 1),
            mean_motion=15.0,
            eccentricity=1.4,  # not a bound orbit
            inclination=50.0,
            ra_of_asc_node=0.0,
            arg_of_pericenter=0.0,
            mean_anomaly=0.0,
            prov=PROV,
        )
