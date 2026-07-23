# From orbital data to verifiable tasks

The chapter before this one gave me a store: normalized, provenance-stamped, content-hashed snapshots of real space data (chapter 3.9). A snapshot is not yet an eval. It is a pile of two-line element sets, catalog rows, and conjunction reports, and none of it is a *question with a checkable answer*. This chapter is the transform that turns that pile into task instances, and it is the chapter that makes the whole thesis honest for Space Domain Awareness. The claim the book is built on is that a verifiable reward is a fixed program with no opinion (chapter 3.4, chapter 9), and for SDA that program is *orbital mechanics itself*. If I ask "do these two objects pass within 5 km of each other in the next 24 hours," I do not need a human, a rubric, or a judge to grade the answer. I propagate both objects with SGP4, find the closest approach, and read the number off. The physics is the oracle, and it is free, exact, and deterministic. That single fact is why SDA belongs in a book about evals as rewards: nature already wrote the answer key.

So the job here is narrow and load-bearing. Take the 3.9 snapshot, define a small set of **task families** where the answer is a number or a boolean the physics can settle, and for each family build the four things a verifiable item needs: a **prompt**, a **gold answer**, a **verifier**, and a **difficulty** stratum, all stamped with the snapshot **provenance** it was built from. The verifier is the interesting part. It is not a separate checker that might disagree with the answer key; it is the *same* Skyfield/SGP4 code path that generated the gold, so the thing that writes the answer and the thing that grades the response are one program. I plug that program into the existing `thesis_suite` verify/score API (chapter 3.4 style), replacing the toy set-difference placeholders with real orbital verifiers. The output of this chapter is a versioned SDA task set that feeds chapter 3.11's freeze.

## Theory

### The task families

A snapshot supports several families, each with a machine-checkable ground truth. I lead with the one that carries the thread.

- **Conjunction screening (flagship).** Given two objects' TLEs and a screening window, "do they conjunct within $X$ km in the next 24 hours?" The oracle propagates both, computes the time of closest approach (TCA) and the miss distance, and the answer is a boolean (or the miss distance itself as a number). This is the task that makes SDA *SDA*: it is exactly what a conjunction-assessment desk does, and the ground truth is a deterministic consequence of two element sets.
- **Orbital-element derivation.** From a single TLE, compute the orbital period, apogee and perigee altitude, inclination, or revolutions per day. These are closed-form functions of the mean elements, so the gold is a one-line calculation and the model has to actually parse the TLE and do the orbital-mechanics arithmetic.
- **Decay / reentry ordering.** Given several objects, order them by predicted decay (lowest perigee and highest drag term $B^\*$ decay soonest). The gold is a permutation; the verifier checks the ordering.
- **Pass prediction / visibility.** From a ground site's latitude, longitude, and elevation, "when is the next pass above 10 degrees elevation, and what is the maximum elevation?" This needs topocentric geometry (an observer on a rotating Earth looking at the satellite), which Skyfield computes directly.
- **Catalog correlation / identification.** Given an object's approximate elements or a partial designation, identify the catalog entry it matches. The gold is a NORAD ID; the verifier is exact match after normalization.

Every one of these is verifiable in the chapter 3.11 sense: the answer is a number, a boolean, a permutation, or a normalized string that a small deterministic checker settles without a human. What is new relative to the toy suite is that the checker is a *physics engine*, and the gold answers were produced by that same engine.

### What a TLE encodes and what SGP4 does

To trust the oracle I have to know, at reading depth, what it computes. A **two-line element set** (TLE) is the standard packed encoding of an object's orbit at a reference instant called the **epoch**. It carries the classical mean elements plus a drag term:

- **mean motion** $n$: revolutions per day (line 2, the field the period comes from),
- **eccentricity** $e$: shape of the ellipse (0 circular, toward 1 elongated),
- **inclination** $i$: tilt of the orbit plane to the equator,
- **right ascension of the ascending node** $\Omega$ (RAAN): where the orbit crosses the equator going north,
- **argument of perigee** $\omega$: orientation of the ellipse within the plane,
- **mean anomaly** $M$: where the object is along the orbit at epoch,
- **epoch**: the timestamp all of the above are referenced to,
- $B^\*$ (B-star): a drag-like coefficient the propagator uses to model atmospheric decay.

The word **mean** is the whole point and the whole danger. These are not instantaneous osculating elements; they are elements fitted so that when you feed them to one specific propagator, **SGP4** (Simplified General Perturbations 4), you recover the object's position. SGP4 is an analytic propagator: it takes the TLE and a time and returns position and velocity, applying secular and periodic perturbations (Earth oblateness $J_2$, atmospheric drag through $B^\*$) in closed form rather than by numerical integration. The elements and the propagator are a matched pair. You must never hand-roll orbital mechanics from a TLE's numbers with textbook Kepler formulas and expect meters of accuracy, because the numbers only mean what they mean *inside SGP4*. This is the first reason the tech stack is locked to `sgp4` + Skyfield: the library *is* the specification.

```admonish derivation title="TLE mean elements to period, apogee, perigee"
The element-derivation family needs the closed-form map from mean elements to the quantities I ask about. Mean motion in the TLE is $n$ revolutions per day; convert to radians per second, $n_{\text{rad}} = n \cdot 2\pi / 86400$. The orbital period is simply the reciprocal of mean motion,

$$T = \frac{86400}{n} \ \text{seconds} = \frac{1440}{n} \ \text{minutes}. \tag{10.1}$$

The semi-major axis follows from Kepler's third law with Earth's gravitational parameter $\mu = 398600.4418\ \text{km}^3/\text{s}^2$,

$$a = \left(\frac{\mu}{n_{\text{rad}}^2}\right)^{1/3}. \tag{10.2}$$

From $a$ and eccentricity $e$, the apogee and perigee *radii* are $r_a = a(1+e)$ and $r_p = a(1-e)$, and the altitudes subtract Earth's equatorial radius $R_\oplus = 6378.137\ \text{km}$,

$$h_a = a(1+e) - R_\oplus, \qquad h_p = a(1-e) - R_\oplus. \tag{10.3}$$

These are the gold answers for the period, apogee, and perigee items: a few lines of arithmetic on the mean elements. I still compute them by reading the elements straight off the parsed `Satrec` object the library gives me, not by re-scraping the TLE text, so the gold and the propagator agree on the same parsed numbers. Revolutions per day is $n$ itself, and inclination is read directly. The difficulty of the item comes not from the formula but from making the model parse a raw TLE and apply the right one.
```

### Time of closest approach and miss distance

The flagship family needs the miss distance, and this is where the oracle earns the name. Let SGP4 give me the TEME position of object 1 as $\mathbf{r}_1(t)$ and object 2 as $\mathbf{r}_2(t)$ over the screening window. The relative position is

$$\mathbf{r}(t) = \mathbf{r}_1(t) - \mathbf{r}_2(t), \tag{10.4}$$

and the closest approach over a window $[t_0, t_0 + W]$ is the minimization

$$d^\* = \min_{t \in [t_0,\, t_0+W]} \lVert \mathbf{r}(t) \rVert, \qquad t_{\text{TCA}} = \arg\min_{t} \lVert \mathbf{r}(t) \rVert. \tag{10.5}$$

The miss distance $d^\*$ is the gold answer, $t_{\text{TCA}}$ is when it happens, and the conjunction boolean is $d^\* \le X$ for the screening threshold $X$.

```admonish derivation title="Finding the minimum: sample, then refine on the derivative"
Minimizing $\lVert \mathbf{r}(t) \rVert$ directly is awkward because of the square root, so I minimize its square, $g(t) = \lVert \mathbf{r}(t) \rVert^2 = \mathbf{r}(t)\cdot\mathbf{r}(t)$, which has its minimum at the same $t$. Differentiating,

$$g'(t) = 2\,\mathbf{r}(t)\cdot\dot{\mathbf{r}}(t) = 2\,\mathbf{r}(t)\cdot\mathbf{v}_{\text{rel}}(t), \tag{10.6}$$

where $\mathbf{v}_{\text{rel}} = \mathbf{v}_1 - \mathbf{v}_2$ is the relative velocity SGP4 returns alongside position. So the closest approach is exactly where the relative position and relative velocity are perpendicular, $\mathbf{r}\cdot\mathbf{v}_{\text{rel}} = 0$, transitioning from closing (negative) to opening (positive).

In practice I do a two-stage search, which is what conjunction-screening tools do. **Coarse sample**: evaluate $\lVert\mathbf{r}(t)\rVert$ on a grid over the window (say every 30 seconds across 24 hours) and find the grid minimum. **Refine**: near that grid point, root-find $\mathbf{r}\cdot\mathbf{v}_{\text{rel}} = 0$ (a few bisection or Brent steps on equation 10.6, or a local fine re-sample) to get $t_{\text{TCA}}$ to sub-second precision, then evaluate the miss distance there. The coarse step must be fine enough not to step over a fast LEO encounter (relative speeds reach ~15 km/s, so a 30-second step moves the geometry by hundreds of km; I use a coarse grid fine enough that the true minimum's basin is never skipped, then let the refine stage do the precision). The output is $(t_{\text{TCA}}, d^\*)$, computed once, used both to write the gold answer and, later, to grade the model.
```

### The tolerance band

The gold miss distance is a float like `4.812 km`. A model that reasons correctly might answer `4.8 km`, `4.81`, `4812 m`, or `~4.8`. Exact float equality would mark all of those wrong, which is a construct-validity disaster: I would be measuring formatting, not orbital reasoning. So a numeric answer counts as correct when it lands inside a **tolerance band** around the gold.

```admonish derivation title="Absolute plus relative tolerance"
A candidate value $x$ is graded correct against gold $x^\*$ iff

$$\lvert x - x^\* \rvert \ \le\ \texttt{atol} + \texttt{rtol}\cdot\lvert x^\* \rvert. \tag{10.7}$$

The **absolute** term `atol` handles answers near zero (a 2 percent relative band around a 0.1 km miss distance is meaninglessly tight); the **relative** term `rtol` handles large answers (a fixed 50 m band around a 7000 km apogee is absurdly strict). Together they are the standard `numpy.isclose` policy, and they are correct for the same reason floating-point equality is wrong: the gold itself is the output of an iterative refine, so it carries its own numerical noise, and the model's arithmetic carries rounding. I set per-family bands: miss distance `atol = 0.05 km` (50 m), `rtol = 0.02`; orbital period `atol = 0.1 min`, `rtol = 0.001`; TCA time a flat `60 s`. These are recorded in the item and the datasheet, because the tolerance *is* part of what "correct" means and a future reader must be able to audit it.

There is one subtlety the boolean conjunction family forces. The verdict $d^\* \le X$ is discontinuous at the threshold, so an item whose true miss distance sits within the numeric tolerance of $X$ has a genuinely ambiguous label: a model that computes $d^\*$ correctly to within tolerance could land on either side. I exclude those near-threshold items from the boolean family (a **guard band**: drop any item with $\lvert d^\* - X \rvert$ smaller than the miss-distance tolerance) so the boolean gold is never noise. Those excluded borderline cases are exactly the interesting *numeric* items, so they move to the miss-distance-as-a-number family instead of being wasted.
```

### Difficulty, sizing, dedup, and where this goes

The rest of the discipline is inherited from Part III and I only note where SDA plugs in. **Difficulty** is measured, not assigned (chapter 3.11): I pilot the draft items against the reference model and bin by solve-rate into easy / medium / hard, and those bands feed the curriculum in chapter 7.5. For conjunctions the difficulty knobs are physical: an encounter with $d^\*$ far from the threshold $X$ is easy (the verdict is obvious even from a sloppy propagation), while $d^\*$ near $X$ is hard (it demands an accurate TCA search), and the screening horizon and number of candidate pairs to screen tune it further. **Item counts** come from the 3.7 power analysis: the suite is sized to resolve an eight-point paired gain, so I generate enough per family to hit the 100/100/100 stratified test target plus dev headroom. **Dedup** runs the 3.8 contamination scan; here it is easy to satisfy by construction, because each item is freshly synthesized from a specific snapshot's element sets and the gold is computed, never copied from any public conjunction report. And the **output feeds chapter 3.11's freeze**: this chapter is the generator, 3.11 is the freeze, manifest, datasheet, and git tag that turn the generated set into the immutable instrument the thesis measures against.

## Tooling

The stack is locked. The oracle is `sgp4` (the reference propagator, vectorized) plus **Skyfield** (which wraps SGP4 and adds the time scales and topocentric geometry the pass-prediction family needs). I never hand-roll the propagation; I call the library, because the library is the specification of what a TLE means. Task instances are **Pydantic** models so the schema is typed and validated at generation time. The verifier plugs into the `thesis_suite` verify/score API from chapter 3.4: I add real orbital verifiers next to the toy `set` and `exact` checkers, then remove the toy ones, since the placeholders were only ever standing in for this. Every task instance carries the 3.9 snapshot **content-hash** as provenance, so an item is traceable to the exact element sets it was built from and is reproducible forever.

```bash title="setup.sh"
uv init sda-tasks && cd sda-tasks
uv add "sgp4>=2.23" "skyfield>=1.49" "numpy>=1.26" "pydantic>=2.7"
uv add --editable ../thesis-suite      # the verify/score API from chapters 3.4 / 3.11
```

The oracle module is the ground truth. Notice it is pure physics: given TLEs it returns numbers, and it is called both to *generate* gold and, through the verifier, to grade.

```python title="sda_tasks/oracle.py"
"""The physics oracle: SGP4/Skyfield ground truth for SDA tasks.

Never hand-roll orbital mechanics; call the library. Positions are TEME
kilometers, velocities TEME km/s, exactly as sgp4 returns them.
"""
from __future__ import annotations
import numpy as np
from sgp4.api import Satrec, SatrecArray, jday

MU = 398600.4418          # km^3 / s^2, Earth gravitational parameter
R_EARTH = 6378.137        # km, equatorial radius

def parse(line1: str, line2: str) -> Satrec:
    """Parse a TLE into a Satrec. The parsed mean elements are the source of
    truth for the element-derivation family, so gold and propagator agree."""
    return Satrec.twoline2rv(line1, line2)

# ---- element-derivation family: closed form on the parsed mean elements ----
def elements(sat: Satrec) -> dict:
    n_rad_per_min = sat.no_kozai              # mean motion, rad/min
    revs_per_day = n_rad_per_min * 1440.0 / (2.0 * np.pi)
    n_rad_per_s = n_rad_per_min / 60.0
    a = (MU / n_rad_per_s**2) ** (1.0 / 3.0)  # eq 10.2
    e = sat.ecco
    return {
        "period_min": 1440.0 / revs_per_day,          # eq 10.1
        "revs_per_day": revs_per_day,
        "semi_major_axis_km": a,
        "apogee_alt_km": a * (1 + e) - R_EARTH,       # eq 10.3
        "perigee_alt_km": a * (1 - e) - R_EARTH,
        "inclination_deg": np.degrees(sat.inclo),
        "eccentricity": e,
    }

# ---- conjunction family: relative position, TCA, miss distance -------------
def _states(sats: SatrecArray, jd: np.ndarray, fr: np.ndarray):
    err, r, v = sats.sgp4(jd, fr)              # r,v shape (nsats, ntimes, 3), TEME km
    if err.any():
        raise ValueError(f"SGP4 propagation error codes: {err[err != 0]}")
    return r, v

def screen_conjunction(tle_a, tle_b, start, window_hours=24.0,
                       coarse_s=30.0):
    """Return (tca_minutes_from_start, miss_distance_km) via eq 10.4-10.6.

    `start` is (year, month, day, hour, minute, second) UTC. Coarse-sample the
    range on a grid, then refine on r . v_rel = 0 around the grid minimum.
    """
    sats = SatrecArray([parse(*tle_a), parse(*tle_b)])
    n = int(window_hours * 3600.0 / coarse_s) + 1
    offsets_s = np.linspace(0.0, window_hours * 3600.0, n)
    jd0, fr0 = jday(*start)
    jd = np.full(n, jd0)
    fr = fr0 + offsets_s / 86400.0
    r, v = _states(sats, jd, fr)
    rel = r[0] - r[1]                          # eq 10.4
    dist = np.linalg.norm(rel, axis=1)
    i = int(np.argmin(dist))                   # coarse minimum

    # refine: local fine grid around the coarse minimum bracket
    lo = offsets_s[max(i - 1, 0)]
    hi = offsets_s[min(i + 1, n - 1)]
    fine = np.linspace(lo, hi, 400)
    jf = np.full(fine.size, jd0)
    ff = fr0 + fine / 86400.0
    rf, vf = _states(sats, jf, ff)
    relf = rf[0] - rf[1]
    distf = np.linalg.norm(relf, axis=1)
    j = int(np.argmin(distf))
    return fine[j] / 60.0, float(distf[j])     # minutes, km
```

The pass-prediction family uses Skyfield's observer geometry rather than raw TEME, because "elevation above a ground site" is topocentric and needs a rotating Earth.

```python title="sda_tasks/passes.py"
"""Pass prediction via Skyfield topocentric geometry."""
from skyfield.api import load, wgs84, EarthSatellite

_ts = load.timescale()

def next_pass(line1, line2, name, lat_deg, lon_deg, elev_m, day_utc,
              min_elevation_deg=10.0):
    """Return (rise, culminate, set) times and peak elevation for the first
    pass above `min_elevation_deg` on the given UTC day."""
    sat = EarthSatellite(line1, line2, name, _ts)
    site = wgs84.latlon(lat_deg, lon_deg, elevation_m=elev_m)
    t0 = _ts.utc(*day_utc)
    t1 = _ts.utc(day_utc[0], day_utc[1], day_utc[2] + 1)
    times, events = sat.find_events(site, t0, t1,
                                    altitude_degrees=min_elevation_deg)
    if len(times) == 0:
        return None
    # events: 0 = rise, 1 = culminate, 2 = set
    culm = next((t for t, e in zip(times, events) if e == 1), None)
    alt, az, _ = (sat - site).at(culm).altaz()
    return {"culminate_utc": culm.utc_iso(), "peak_elevation_deg": alt.degrees}
```

The task-instance model is Pydantic, and it carries the snapshot hash as provenance.

```python title="sda_tasks/models.py"
"""Typed task instances. `verifier` names the checker; `provenance` pins the
3.9 snapshot content-hash the item was built from."""
from pydantic import BaseModel, Field

class TaskInstance(BaseModel):
    id: str
    domain: str                       # conjunction | elements | decay | pass | catalog
    verifier: str                     # conjunction | miss_distance_km | period_min | ...
    input: str                        # the TLE(s) / catalog rows the model sees
    question: str
    target: str                       # gold answer, computed by the oracle
    tolerance: dict = Field(default_factory=dict)   # atol/rtol for numeric families
    difficulty: str = "medium"        # measured in the pilot (feeds 7.5)
    provenance: str                   # "snapshot:<content-sha256>[:12]"
    license: str = "CC0-1.0"          # derived task, gold computed by the oracle
```

The verifiers are the payload of the chapter. Each is a pure `(response, target) -> bool` function with the `thesis_suite` signature, and the numeric ones apply the tolerance band from equation (10.7). The design point the spec insists on: the verifier and the gold generator share the oracle, so the answer key and the grader are the same physics.

```python title="sda_tasks/verifiers.py"
"""Orbital verifiers plugged into the thesis_suite verify/score API.

These REPLACE the toy set-difference / exact placeholders from the draft suite.
Each verifier is a pure (response, target) -> bool. Numeric families apply the
tolerance band of eq 10.7. The gold `target` was produced by sda_tasks.oracle,
so grader and answer key are one code path.
"""
from __future__ import annotations
import re

_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

def _last_number(text: str, unit_scale: dict | None = None) -> float | None:
    """Extract the model's final numeric answer, normalizing units (m->km)."""
    t = text.replace(",", "")
    if unit_scale:
        # if the answer is stated in meters, fold to km before comparing
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*(m|meters|km|kilometers)\b", t, re.I)
        if m and m.group(2).lower() in ("m", "meters"):
            return float(m.group(1)) / 1000.0
    nums = _NUM.findall(t)
    return float(nums[-1]) if nums else None

def _isclose(x: float, gold: float, atol: float, rtol: float) -> bool:
    return abs(x - gold) <= atol + rtol * abs(gold)   # eq 10.7

def numeric_km(response: str, target: str, atol=0.05, rtol=0.02) -> bool:
    x = _last_number(response, unit_scale={"km": 1.0})
    return x is not None and _isclose(x, float(target), atol, rtol)

def numeric_min(response: str, target: str, atol=0.1, rtol=0.001) -> bool:
    x = _last_number(response)
    return x is not None and _isclose(x, float(target), atol, rtol)

def conjunction(response: str, target: str) -> bool:
    """Boolean verdict: does the response's yes/no match the gold verdict?"""
    truthy, falsy = {"yes", "true", "conjunct", "conjunction"}, {"no", "false", "clear"}
    gold = target.strip().lower() in truthy
    hit = next((w for w in re.findall(r"[a-z]+", response.lower())
                if w in truthy | falsy), None)
    return hit is not None and (hit in truthy) == gold

# register next to (and replacing) the toy checkers
ORBITAL_VERIFIERS = {
    "conjunction": conjunction,
    "miss_distance_km": numeric_km,
    "period_min": numeric_min,
    "apogee_alt_km": numeric_km,
    "perigee_alt_km": numeric_km,
}

def check(item, response: str) -> bool:
    """thesis_suite entry point: dispatch an item to its orbital verifier."""
    return ORBITAL_VERIFIERS[item.verifier](response, item.target)
```

```admonish gotcha title="Epoch, frame, and units are where SGP4 oracles quietly go wrong"
The oracle is only trustworthy if I respect three things the library will not enforce for me. **Frame:** `sgp4` returns position and velocity in **TEME** (True Equator, Mean Equinox), an inertial-ish frame, *not* ECEF/ECI-of-date and *not* lat/lon. Differencing two TEME vectors (equation 10.4) is fine because both objects are in the same frame, but the moment I want "position over a ground site" I must go through Skyfield's topocentric conversion, never subtract a TEME vector from a lat/lon. Mixing TEME and ECEF is the classic silent error that puts a conjunction off by thousands of km. **Epoch and time:** I feed absolute Julian dates (`jday(...)` giving `jd, fr`), which sidesteps the "minutes since epoch" convention, but the deeper trap is that SGP4 accuracy degrades away from the TLE's epoch. A screening window days from epoch is propagating a stale fit; I keep the window near the snapshot's element epochs and record the epoch age, because a task built on a week-old TLE has a gold answer with real error bars. **Units:** everything the library hands back is kilometers and km/s. If a model answers in meters and my extractor does not fold the unit, a correct answer reads as 1000x wrong; the verifier's unit normalization (the `m -> km` fold above) exists precisely for that. Km-versus-m, TEME-versus-ECEF, and epoch age are the three checks I run before trusting any generated gold.
```

## Lab

I generate a small SDA task set from a 3.9 snapshot, then show the oracle grading three worked model responses: one correct within tolerance, one wrong, and one that is numerically right but in the wrong format. The two TLEs below are illustrative records drawn from a pinned celestrak snapshot (a real run reads them from the DVC-pinned Parquet of chapter 3.9 and stamps the item with that snapshot's content-hash); the numbers the oracle returns on them are worked synthetic values, and you regenerate them against your own snapshot.

```python title="sda_tasks/generate.py"
"""Build a versioned SDA task set from a 3.9 snapshot. Gold answers are
computed by the oracle; each item is stamped with the snapshot content-hash."""
import json
from pathlib import Path
from sda_tasks.oracle import parse, elements, screen_conjunction
from sda_tasks.models import TaskInstance

SNAPSHOT_SHA = "b7f3a9c1e2d4"   # 3.9 snapshot content-hash (first 12), the provenance

# two illustrative TLEs from the pinned snapshot (celestrak, redistributable)
SAT_A = ("1 25544U 98067A   26203.51782528  .00016717  00000-0  30074-3 0  9991",
         "2 25544  51.6382  88.4247 0004796  76.6963 283.4442 15.50075566454121")
SAT_B = ("1 48274U 21035A    26203.47630000  .00002182  00000-0  62911-4 0  9992",
         "2 48274  51.6431  84.9110 0007899  92.1004 268.0912 15.48920201123456")

SCREEN_THRESHOLD_KM = 5.0
START = (2026, 7, 22, 0, 0, 0)

def build():
    items = []

    # element-derivation item (medium): period from SAT_A
    el = elements(parse(*SAT_A))
    items.append(TaskInstance(
        id="sda-elem-0001", domain="elements", verifier="period_min",
        input="\n".join(SAT_A),
        question="Compute the orbital period of this object, in minutes.",
        target=f"{el['period_min']:.3f}",
        tolerance={"atol": 0.1, "rtol": 0.001},
        difficulty="medium", provenance=f"snapshot:{SNAPSHOT_SHA}"))

    # conjunction screening (flagship): boolean verdict
    tca_min, miss_km = screen_conjunction(SAT_A, SAT_B, START, window_hours=24.0)
    verdict = "yes" if miss_km <= SCREEN_THRESHOLD_KM else "no"
    items.append(TaskInstance(
        id="sda-conj-0001", domain="conjunction", verifier="conjunction",
        input="OBJECT A:\n" + "\n".join(SAT_A) + "\n\nOBJECT B:\n" + "\n".join(SAT_B),
        question=(f"Do these two objects conjunct within {SCREEN_THRESHOLD_KM:g} km "
                  f"in the next 24 hours? Answer yes or no."),
        target=verdict,
        difficulty="hard", provenance=f"snapshot:{SNAPSHOT_SHA}"))

    # same encounter as a numeric item (the interesting near-threshold case)
    items.append(TaskInstance(
        id="sda-conj-0002", domain="conjunction", verifier="miss_distance_km",
        input="OBJECT A:\n" + "\n".join(SAT_A) + "\n\nOBJECT B:\n" + "\n".join(SAT_B),
        question="What is the miss distance at closest approach, in km, over the next 24 hours?",
        target=f"{miss_km:.3f}",
        tolerance={"atol": 0.05, "rtol": 0.02},
        difficulty="hard", provenance=f"snapshot:{SNAPSHOT_SHA}"))

    out = Path("taskset/v0.1"); out.mkdir(parents=True, exist_ok=True)
    (out / "items.jsonl").write_text(
        "\n".join(json.dumps(i.model_dump()) for i in items))
    print(f"gold: TCA={tca_min:.2f} min, miss={miss_km:.3f} km, verdict={verdict}")
    print(f"wrote {len(items)} items -> {out/'items.jsonl'} (provenance {SNAPSHOT_SHA})")
    return items

if __name__ == "__main__":
    build()
```

Now grade three worked responses to the numeric miss-distance item, whose gold the oracle computed as a worked synthetic `miss = 4.812 km` (regenerate on your snapshot to confirm). The point is that the *same* verifier settles all three without a human.

```python title="sda_tasks/grade_demo.py"
"""Grade three worked responses against the oracle-computed gold."""
from sda_tasks.verifiers import numeric_km, conjunction

GOLD_MISS_KM = "4.812"   # worked synthetic value from the oracle; regenerate to confirm

responses = {
    "correct_within_tol": "Propagating both, closest approach is about 4.79 km.",
    "wrong":              "They are well separated; the miss distance is roughly 41 km.",
    "right_answer_bad_format": "The minimum separation is 4812 meters.",
}

for label, text in responses.items():
    ok = numeric_km(text, GOLD_MISS_KM)   # atol=0.05 km, rtol=0.02  (eq 10.7)
    print(f"{label:24s} -> {'CORRECT' if ok else 'INCORRECT'}")

# boolean verdict demo (gold = 'yes' since 4.812 <= 5.0 threshold)
print("boolean 'no' vs gold yes ->",
      "CORRECT" if conjunction("No, they stay clear.", "yes") else "INCORRECT")
```

```admonish thesis-thread
Here the SDA thread stops being flavor. Since the preface, "Space Domain Awareness" has been a label on a pile of toy set-difference items that had nothing to do with space; the answer key was a placeholder and the reader was asked to imagine the real thing. This chapter builds the real thing. A conjunction-screening item is a genuine question a space-operations desk answers every day, its gold answer is a deterministic consequence of two real element sets propagated by the standard propagator, and its verifier is that same propagator. There is no judge, no rubric, no opinion anywhere in the loop: orbital mechanics wrote the answer key, and the verifiable reward that chapter 9 argued for abstractly is, for SDA, literally the output of `sgp4`. The toy checkers are deleted; the frozen suite of chapter 3.11 will be built from these families, and every downstream claim ("INT4 held reasoning," "GRPO moved the SDA delta") becomes a claim about whether the model can do orbital reasoning that the physics can check.
```

**What you should see.** Running `generate.py` writes `taskset/v0.1/items.jsonl` with element and conjunction items, each stamped `snapshot:b7f3a9c1e2d4`, and prints the oracle's computed TCA, miss distance, and verdict for the flagship pair. Running `grade_demo.py` prints `correct_within_tol -> CORRECT` (4.79 is inside the 50 m + 2 percent band around 4.812), `wrong -> INCORRECT` (41 km is nowhere near), and `right_answer_bad_format -> CORRECT` (4812 meters folds to 4.812 km and matches), which is the whole demonstration: the verifier rewards the reasoning, not the formatting, and rejects the genuinely wrong answer, using one physics oracle for both the answer key and the grade. The artifact is the generated task set `taskset/v0.1/items.jsonl` plus the orbital verifier module `sda_tasks/verifiers.py`, and both feed chapter 3.11, where the calibrated, deduped, power-sized set is frozen, content-hashed, datasheeted, and tagged into the instrument the rest of the thesis reads from. The gold miss distance, TCA, and per-family solve rates are (measured on the baseline machine against the pinned snapshot -- record value, date, driver).

```admonish read-along
[BRM] chapter 3 is the companion for why a reasoning model wants a programmatic verifier: the answer is checkable, so the reward can be cheap and honest, and this chapter is the SDA-specific realization of exactly that, with orbital mechanics playing the role of the checker. [AIE] chapter 8 (dataset engineering) is the companion for the generation side: how you turn a raw data source into task instances with clean provenance and a defensible ground truth, which is the discipline that keeps this task set from being a folder of plausible-looking questions with fragile answers. Read them together as the two halves of "a verifiable task is a data-engineering artifact whose grader is a physics engine."
```

```admonish substack-seed
"For space, the answer key is a law of physics." Most eval tasks need a human to write the ground truth, and every human answer key is a place where bias, error, and disagreement leak into your measurement. Orbital mechanics doesn't need one. If I ask whether two satellites pass within five kilometers tomorrow, I don't consult an expert, I propagate both objects with the standard model and read the miss distance off the geometry, and the same code that writes the answer grades the response. The post can carry a reader from "verifiable reward is a nice idea" to "here is a domain where the reward is free, exact, and impossible to sweet-talk," land on the design detail that makes it real (the grader and the answer key are one program, so they cannot disagree), and close on the tolerance band, the one place judgment still lives: deciding how close is close enough is the only opinion left in a task the universe otherwise settles for you.
```
