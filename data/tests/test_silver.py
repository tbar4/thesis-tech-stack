import pandas as pd
from deltalake import DeltaTable

from sda_data import config
from sda_data.silver import append_element_sets, element_sets_uri


def frame(norads):
    return pd.DataFrame(
        {
            "norad_cat_id": norads,
            "object_name": [f"OBJ-{n}" for n in norads],
            "mean_motion": [15.5] * len(norads),
            "eccentricity": [0.001] * len(norads),
            "inclination": [51.6] * len(norads),
            "source": ["celestrak"] * len(norads),
            "snapshot_hash": ["sha256:" + "0" * 64] * len(norads),
        }
    )


def test_uri_defaults_to_local_delta_tier(tmp_data_home):
    assert element_sets_uri() == str(config.delta_root() / "element_sets")


def test_append_creates_then_appends(tmp_data_home):
    append_element_sets(frame([1, 2]))
    append_element_sets(frame([3]))
    table = DeltaTable(element_sets_uri())
    assert table.version() == 1                     # two commits: v0 create, v1 append
    assert len(table.to_pandas()) == 3              # appends never overwrite


def test_append_returns_new_version(tmp_data_home):
    v0 = append_element_sets(frame([1]))
    v1 = append_element_sets(frame([2]))
    assert (v0, v1) == (0, 1)
