import duckdb
import pandas as pd

from sda_data import config
from sda_data.tasks.dbt_build import run

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def seed(subpath: str, df: pd.DataFrame) -> None:
    out = config.snap_root() / subpath
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)


def seed_all(tmp_data_home) -> None:
    t1 = pd.Timestamp("2026-07-24", tz="UTC")
    t2 = pd.Timestamp("2026-07-25", tz="UTC")
    seed("celestrak/element_sets/2026-07-24/a.parquet", pd.DataFrame({
        "norad_cat_id": [25544, 48274], "object_name": ["ISS", "CSS"],
        "mean_motion": [15.5, 15.6], "eccentricity": [0.0003, 0.0004],
        "inclination": [51.6, 41.5], "source": ["celestrak"] * 2,
        "snapshot_hash": [HASH_A] * 2, "fetch_time": [t1] * 2,
    }))
    seed("celestrak/element_sets/2026-07-25/b.parquet", pd.DataFrame({
        "norad_cat_id": [25544], "object_name": ["ISS"],
        "mean_motion": [15.51], "eccentricity": [0.0003],
        "inclination": [51.6], "source": ["celestrak"],
        "snapshot_hash": [HASH_B], "fetch_time": [t2],
    }))
    seed("spacedevs/launches/2026-07-24/a.parquet", pd.DataFrame({
        "id": ["u-1", "u-2"], "name": ["A", "B"],
        "net": ["2126-01-01T00:00:00Z", "2020-01-01T00:00:00Z"],  # one future, one past
        "status_name": ["Go", "Success"], "provider": ["X", "Y"],
        "source": ["spacedevs"] * 2, "snapshot_hash": [HASH_A] * 2,
        "fetch_time": [t1] * 2,
    }))
    seed("spaceflightnews/articles/2026-07-24/a.parquet", pd.DataFrame({
        "id": [101], "title": ["T"], "url": ["u"], "news_site": ["N"],
        "summary": ["s"], "published_at": ["2026-07-24T10:00:00Z"],
        "source": ["spaceflightnews"], "snapshot_hash": [HASH_A],
        "fetch_time": [t1],
    }))
    seed("nasa/neows/2026-07-24/a.parquet", pd.DataFrame({
        "neo_reference_id": ["354"], "name": ["PK9"],
        "close_approach_date": ["2026-07-25"], "miss_distance_km": [1.2e6],
        "rel_velocity_kms": [14.2], "absolute_magnitude_h": [21.8],
        "source": ["nasa"], "snapshot_hash": [HASH_A], "fetch_time": [t1],
    }))


def test_dbt_build_produces_gold(tmp_data_home):
    seed_all(tmp_data_home)
    run()   # dbt build: models + schema tests; raises on any failure

    con = duckdb.connect(str(config.data_home() / "data" / "gold" / "sda.duckdb"))
    latest = con.sql("select * from latest_element_sets order by norad_cat_id").df()
    assert len(latest) == 2                                   # one row per object
    assert latest[latest.norad_cat_id == 25544].iloc[0].snapshot_hash == HASH_B

    launches = con.sql("select * from upcoming_launches").df()
    assert list(launches.id) == ["u-1"]                       # past launch filtered out

    assert con.sql("select count(*) c from latest_articles").fetchone()[0] == 1
    assert con.sql("select count(*) c from neo_approaches").fetchone()[0] == 1
