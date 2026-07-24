"""Close-approach screening: sieve, then propagate.

All-vs-all conjunction screening is O(n^2) propagations; the standard fix is a
cheap apogee/perigee band sieve first (two objects whose radial bands never
overlap cannot approach), then SGP4 only on survivors. Positions are
propagated once per OBJECT over the shared grid and cached, so each surviving
pair costs numpy distance math, not another propagation."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sgp4.api import WGS72, Satrec, jday

MU_KM3_S2 = 398600.4418          # Earth GM
TWO_PI = 2.0 * math.pi


def orbital_radii(mean_motion_rev_day: float, eccentricity: float) -> tuple[float, float]:
    """(perigee_radius_km, apogee_radius_km) from mean motion + eccentricity."""
    n_rad_s = mean_motion_rev_day * TWO_PI / 86400.0
    a = (MU_KM3_S2 / (n_rad_s * n_rad_s)) ** (1.0 / 3.0)
    return a * (1.0 - eccentricity), a * (1.0 + eccentricity)


def sieve_pairs(df: pd.DataFrame, pad_km: float = 25.0) -> list[tuple[int, int]]:
    """Index pairs whose padded radial bands overlap. Everyone else is pruned
    without a single propagation."""
    bands = [orbital_radii(r.mean_motion, r.eccentricity) for r in df.itertuples()]
    pairs: list[tuple[int, int]] = []
    for i in range(len(bands)):
        rp_i, ra_i = bands[i]
        for j in range(i + 1, len(bands)):
            rp_j, ra_j = bands[j]
            if rp_i - ra_j > pad_km or rp_j - ra_i > pad_km:
                continue                                  # bands can never meet
            pairs.append((i, j))
    return pairs


def satrec_from_elements(row) -> Satrec:
    """Build an SGP4 satrec straight from the normalized element-set schema."""
    epoch: datetime = row.epoch
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    jd, fr = jday(epoch.year, epoch.month, epoch.day,
                  epoch.hour, epoch.minute, epoch.second + epoch.microsecond / 1e6)
    sat = Satrec()
    sat.sgp4init(
        WGS72, "i", int(row.norad_cat_id), jd + fr - 2433281.5,
        float(getattr(row, "bstar", 0.0) or 0.0), 0.0, 0.0,
        float(row.eccentricity),
        math.radians(row.arg_of_pericenter),
        math.radians(row.inclination),
        math.radians(row.mean_anomaly),
        row.mean_motion * TWO_PI / 1440.0,        # rev/day -> rad/min
        math.radians(row.ra_of_asc_node),
    )
    return sat


def _positions(sat: Satrec, jds: np.ndarray, frs: np.ndarray) -> np.ndarray | None:
    """(T, 3) TEME positions in km, or None if SGP4 errors anywhere."""
    errs, pos, _vel = sat.sgp4_array(jds, frs)
    if np.any(errs != 0):
        return None
    return pos


def screen(
    df: pd.DataFrame,
    start: datetime,
    window_hours: float = 24.0,
    step_s: float = 60.0,
    miss_threshold_km: float = 25.0,
    pad_km: float = 25.0,
) -> pd.DataFrame:
    """Screen all pairs; return close approaches under the threshold.

    Coarse pass on the shared grid finds each pair's minimum-distance sample;
    a fine pass (1 s steps, +-step_s around it) sharpens tca and miss."""
    pairs = sieve_pairs(df, pad_km=pad_km)
    n_steps = int(window_hours * 3600.0 / step_s) + 1
    offsets = np.arange(n_steps) * step_s
    jd0, fr0 = jday(start.year, start.month, start.day,
                    start.hour, start.minute, start.second)
    jds = np.full(n_steps, jd0)
    frs = fr0 + offsets / 86400.0

    sats = [satrec_from_elements(row) for row in df.itertuples()]
    cache: dict[int, np.ndarray | None] = {}

    def positions(idx: int) -> np.ndarray | None:
        if idx not in cache:
            cache[idx] = _positions(sats[idx], jds, frs)
        return cache[idx]

    hits: list[dict] = []
    for i, j in pairs:
        pi, pj = positions(i), positions(j)
        if pi is None or pj is None:
            continue                                  # decayed / bad elements
        dist = np.linalg.norm(pi - pj, axis=1)
        k = int(np.argmin(dist))
        if dist[k] >= miss_threshold_km:
            continue

        # Fine pass: 1 s resolution around the coarse minimum.
        lo = max(0.0, offsets[k] - step_s)
        hi = min(offsets[-1], offsets[k] + step_s)
        fine = np.arange(lo, hi + 1.0)
        fj = np.full(fine.shape, jd0)
        ff = fr0 + fine / 86400.0
        fi_pos = _positions(sats[i], fj, ff)
        fj_pos = _positions(sats[j], fj, ff)
        if fi_pos is None or fj_pos is None:
            continue
        fdist = np.linalg.norm(fi_pos - fj_pos, axis=1)
        m = int(np.argmin(fdist))
        row_i, row_j = df.iloc[i], df.iloc[j]
        hits.append({
            "norad_i": int(row_i.norad_cat_id),
            "norad_j": int(row_j.norad_cat_id),
            "tca": (start + timedelta(seconds=float(fine[m]))).isoformat(),
            "miss_km": float(fdist[m]),
            "snapshot_hash_i": row_i.snapshot_hash,
            "snapshot_hash_j": row_j.snapshot_hash,
        })
    return pd.DataFrame(hits, columns=[
        "norad_i", "norad_j", "tca", "miss_km", "snapshot_hash_i", "snapshot_hash_j",
    ])
