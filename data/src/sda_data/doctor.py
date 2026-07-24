"""`make doctor`: is this machine set up to run the pipeline?
Checks are pure and fast; printing happens only in main()."""
from __future__ import annotations

import os

from sda_data import config


def check_environment() -> list[str]:
    """Return human-readable issues; empty list means healthy."""
    issues: list[str] = []

    for var in ("S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if not os.environ.get(var):
            issues.append(f"{var} is not set (durable tier unreachable; see .env.example)")

    home = config.data_home()
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / ".doctor-probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        issues.append(f"SDA_DATA_HOME {home} is not writable: {exc}")

    return issues


def main() -> None:
    issues = check_environment()
    if not issues:
        print("doctor: all checks passed")
        raise SystemExit(0)
    for issue in issues:
        print(f"doctor: FAIL  {issue}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
