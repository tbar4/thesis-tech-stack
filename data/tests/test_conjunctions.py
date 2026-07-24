from datetime import datetime, timezone

import pandas as pd

from sda_data.conjunctions import orbital_radii, screen, sieve_pairs

EPOCH = datetime(2026, 7, 24, 6, 0, tzinfo=timezone.utc)


def element_row(norad, mean_motion=15.5, ecc=0.0003, mean_anomaly=300.0, incl=51.64):
    return {
        "norad_cat_id": norad, "object_name": f"OBJ-{norad}", "epoch": EPOCH,
        "mean_motion": mean_motion, "eccentricity": ecc, "inclination": incl,
        "ra_of_asc_node": 120.0, "arg_of_pericenter": 30.0,
        "mean_anomaly": mean_anomaly, "bstar": 0.0001,
        "snapshot_hash": "sha256:" + "0" * 64,
    }


def test_orbital_radii_iss_like():
    rp, ra = orbital_radii(15.5, 0.0)
    assert rp == ra                                  # circular
    assert 6700 < rp < 6900                          # ~420 km altitude


def test_sieve_prunes_leo_vs_geo():
    df = pd.DataFrame([element_row(1, mean_motion=15.5),
                       element_row(2, mean_motion=1.0027)])   # GEO
    assert sieve_pairs(df, pad_km=25.0) == []


def test_sieve_keeps_coplanar_leo_pair():
    df = pd.DataFrame([element_row(1), element_row(2, mean_anomaly=300.5)])
    assert sieve_pairs(df, pad_km=25.0) == [(0, 1)]


def test_screen_detects_close_phase_shifted_pair():
    # Identical orbit, mean anomaly offset by 0.01 deg: constant along-track
    # separation of roughly (0.01/360) * 2*pi*a ~ 1.2 km. Must be reported.
    df = pd.DataFrame([element_row(1, mean_anomaly=300.00),
                       element_row(2, mean_anomaly=300.01)])
    hits = screen(df, start=EPOCH, window_hours=1.0, miss_threshold_km=25.0)
    assert len(hits) == 1
    hit = hits.iloc[0]
    assert {hit.norad_i, hit.norad_j} == {1, 2}
    assert hit.miss_km < 5.0


def test_screen_ignores_far_pair():
    # Same orbit, opposite phase: separation ~ 2a, never close.
    df = pd.DataFrame([element_row(1, mean_anomaly=0.0),
                       element_row(2, mean_anomaly=180.0)])
    hits = screen(df, start=EPOCH, window_hours=1.0, miss_threshold_km=25.0)
    assert len(hits) == 0


def test_build_task_appends_delta(tmp_data_home):
    from deltalake import DeltaTable

    from sda_data.silver import append_element_sets, table_uri
    from sda_data.tasks.build_conjunctions import run as build_run

    rows = [element_row(1, mean_anomaly=300.00), element_row(2, mean_anomaly=300.01)]
    df = pd.DataFrame(rows)
    df["source"] = "celestrak"
    df["fetch_time"] = pd.Timestamp("2026-07-24", tz="UTC")
    append_element_sets(df)

    count = build_run(window_hours=1.0)
    assert count == 1
    table = DeltaTable(table_uri("close_approaches")).to_pandas()
    assert len(table) == 1 and table.iloc[0].miss_km < 5.0
