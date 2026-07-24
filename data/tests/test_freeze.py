import pandas as pd
from deltalake import DeltaTable

from sda_data import config
from sda_data.freeze import freeze_landing
from sda_data.gates import LaunchFrame
from sda_data.silver import table_uri


def make_landing(tmp_data_home):
    landing = config.snap_root() / "_landing" / "x" / "run1"
    landing.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["u-1", "u-2"],
            "name": ["Falcon 9 | Starlink", "Vulcan | USSF-106"],
            "net": ["2026-08-01T00:00:00Z", "2026-08-02T00:00:00Z"],
            "status_name": ["Go", "TBD"],
            "provider": ["SpaceX", "ULA"],
        }
    ).to_parquet(landing / "part.parquet", index=False)
    return landing


def test_freeze_landing_hashes_gates_and_appends(tmp_data_home):
    landing = make_landing(tmp_data_home)
    snapshot_hash = freeze_landing(
        landing, source="spacedevs", kind="launches",
        gate=LaunchFrame, delta_table="launches",
    )

    digest = snapshot_hash.split(":", 1)[1]
    assert list(config.raw_root().rglob(f"{digest}.raw"))          # frozen, shippable
    snaps = list((config.snap_root() / "spacedevs" / "launches").rglob("*.parquet"))
    assert len(snaps) == 1                                          # derived parquet
    table = DeltaTable(table_uri("launches")).to_pandas()
    assert len(table) == 2
    assert (table.snapshot_hash == snapshot_hash).all()
    assert (table.source == "spacedevs").all()
