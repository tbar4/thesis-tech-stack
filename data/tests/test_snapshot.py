import hashlib
from datetime import datetime, timezone

from sda_data import config
from sda_data.snapshot import content_address, write_raw_snapshot

FETCH = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)


def test_content_address_is_sha256_of_bytes():
    raw = b"orbital bytes"
    expected = "sha256:" + hashlib.sha256(raw).hexdigest()
    assert content_address(raw) == expected


def test_redistributable_source_lands_on_raw_tier(tmp_data_home):
    path, addr = write_raw_snapshot("celestrak", "gp-active", b"abc", FETCH)
    assert path.is_file()
    assert config.raw_root() in path.parents
    assert path.name == addr.split(":", 1)[1] + ".raw"
    assert "2026-07-24" in str(path)


def test_restricted_source_lands_on_live_tier(tmp_data_home):
    path, _ = write_raw_snapshot("spacetrack", "cdm", b"secret", FETCH)
    assert config.live_root() in path.parents
    assert config.raw_root() not in path.parents


def test_write_is_idempotent_for_identical_bytes(tmp_data_home):
    p1, a1 = write_raw_snapshot("celestrak", "gp-active", b"same", FETCH)
    p2, a2 = write_raw_snapshot("celestrak", "gp-active", b"same", FETCH)
    assert p1 == p2 and a1 == a2


def test_snapshot_file_is_never_mutated(tmp_data_home):
    p1, _ = write_raw_snapshot("celestrak", "gp-active", b"v1", FETCH)
    before = p1.read_bytes()
    write_raw_snapshot("celestrak", "gp-active", b"v2", FETCH)  # new addr, new file
    assert p1.read_bytes() == before
