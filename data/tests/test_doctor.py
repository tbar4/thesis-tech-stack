from sda_data.doctor import check_environment


def test_reports_missing_s3_config(tmp_data_home, monkeypatch):
    for var in ("S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(var, raising=False)
    issues = check_environment()
    assert any("S3_ENDPOINT_URL" in issue for issue in issues)


def test_clean_when_configured(tmp_data_home, monkeypatch):
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.example.com")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    assert check_environment() == []


def test_reports_unwritable_data_home(monkeypatch):
    monkeypatch.setenv("SDA_DATA_HOME", "/proc/definitely-not-writable")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.example.com")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    issues = check_environment()
    assert any("SDA_DATA_HOME" in issue for issue in issues)
