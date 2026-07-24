from pathlib import Path


def test_tiers_hang_off_data_home(tmp_data_home):
    from sda_data import config

    assert config.data_home() == Path(tmp_data_home)
    assert config.raw_root() == tmp_data_home / "data" / "raw"
    assert config.snap_root() == tmp_data_home / "data" / "snapshots"
    assert config.live_root() == tmp_data_home / "data" / "live"
    assert config.delta_root() == tmp_data_home / "data" / "delta"


def test_redistributable_excludes_restricted_sources():
    from sda_data import config

    assert "celestrak" in config.REDISTRIBUTABLE
    assert "spacedevs" in config.REDISTRIBUTABLE
    assert "spacetrack" not in config.REDISTRIBUTABLE
    assert "nasa" not in config.REDISTRIBUTABLE


def test_s3_storage_options_from_env(monkeypatch):
    from sda_data import config

    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.example.com")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    opts = config.s3_storage_options()
    assert opts["AWS_ENDPOINT_URL"] == "https://s3.example.com"
    assert opts["AWS_ACCESS_KEY_ID"] == "k"
    assert opts["AWS_SECRET_ACCESS_KEY"] == "s"


def test_s3_storage_options_empty_when_unconfigured(monkeypatch):
    from sda_data import config

    for var in ("S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert config.s3_storage_options() == {}
