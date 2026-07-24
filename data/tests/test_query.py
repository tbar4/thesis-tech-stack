import pandas as pd

from sda_data.query import latest_element_sets
from sda_data.silver import append_element_sets


def frame(norads, snapshot_hash):
    return pd.DataFrame(
        {
            "norad_cat_id": norads,
            "object_name": [f"OBJ-{n}" for n in norads],
            "mean_motion": [15.5] * len(norads),
            "eccentricity": [0.001] * len(norads),
            "inclination": [51.6] * len(norads),
            "source": ["celestrak"] * len(norads),
            "snapshot_hash": [snapshot_hash] * len(norads),
            "fetch_time": pd.to_datetime(["2026-07-24"] * len(norads), utc=True),
        }
    )


def test_latest_wins_per_object(tmp_data_home):
    append_element_sets(frame([1, 2], "sha256:" + "a" * 64))
    newer = frame([2], "sha256:" + "b" * 64)
    newer["fetch_time"] = pd.to_datetime(["2026-07-25"], utc=True)
    append_element_sets(newer)

    df = latest_element_sets()
    assert len(df) == 2                                    # one row per object
    row2 = df[df.norad_cat_id == 2].iloc[0]
    assert row2.snapshot_hash.startswith("sha256:b")       # newest fetch wins
