import pandas as pd
import pandera.errors
import pytest

from sda_data.gates import ElementSetFrame


def good_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "norad_cat_id": [25544, 48274],
            "object_name": ["ISS (ZARYA)", "CSS (TIANHE)"],
            "mean_motion": [15.50, 15.60],
            "eccentricity": [0.0003, 0.0004],
            "inclination": [51.64, 41.47],
            "source": ["celestrak", "celestrak"],
            "snapshot_hash": ["sha256:" + "0" * 64] * 2,
        }
    )


def test_valid_frame_passes():
    out = ElementSetFrame.validate(good_frame())
    assert len(out) == 2


def test_duplicate_norad_id_rejected():
    df = good_frame()
    df.loc[1, "norad_cat_id"] = 25544
    with pytest.raises(pandera.errors.SchemaError):
        ElementSetFrame.validate(df)


def test_impossible_mean_motion_rejected():
    df = good_frame()
    df.loc[0, "mean_motion"] = 25.0  # > 20 revs/day is garbage
    with pytest.raises(pandera.errors.SchemaError):
        ElementSetFrame.validate(df)


def test_spacetrack_is_a_valid_element_set_source():
    df = good_frame()
    df["source"] = "spacetrack"
    assert len(ElementSetFrame.validate(df)) == 2


def test_neows_frame_rejects_nonpositive_miss_distance():
    from sda_data.gates import NeoWsFrame

    df = pd.DataFrame(
        {
            "neo_reference_id": ["3542519"],
            "close_approach_date": ["2026-07-25"],
            "miss_distance_km": [-1.0],
            "rel_velocity_kms": [10.0],
            "absolute_magnitude_h": [22.0],
            "source": ["nasa"],
            "snapshot_hash": ["sha256:" + "0" * 64],
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        NeoWsFrame.validate(df)
