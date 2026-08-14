"""Cleaning and statistics for hourly wind observations.

Input is a Polars DataFrame from :func:`arctic_sim.ingest.env_canada.load_rea_point`
(or any other loader with the same schema).  Everything downstream uses SI units:
wind speed in m/s, direction in degrees (0=N, clockwise, 90=E).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl
from scipy import signal, stats

from arctic_sim.ingest.env_canada import to_si


def subset_year(df: pl.DataFrame, year: int) -> pl.DataFrame:
    return df.filter(pl.col("LOCAL_DATE").dt.year() == year).sort("LOCAL_DATE")


def clean(df: pl.DataFrame, *, drop_calm_direction: bool = True) -> pl.DataFrame:
    """Convert to SI, drop rows without a valid wind speed, expose clean columns.

    Returns a DataFrame with ``time`` (LOCAL_DATE), ``wind_speed_ms``,
    ``wind_direction_deg`` (may be null if the original DIRECTION was calm/0).
    """
    out = to_si(df).rename({"LOCAL_DATE": "time"}).select(
        "time", "wind_speed_ms", "wind_direction_deg"
    )
    out = out.filter(pl.col("wind_speed_ms").is_not_null())
    if drop_calm_direction:
        # ECCC encodes calm periods as direction 0; keep the speed row but null the direction.
        out = out.with_columns(
            pl.when(pl.col("wind_direction_deg") == 0.0)
            .then(None)
            .otherwise(pl.col("wind_direction_deg"))
            .alias("wind_direction_deg")
        )
    return out


@dataclass
class GapReport:
    total_hours_expected: int
    total_hours_present: int
    coverage_pct: float
    longest_gap_hours: int
    n_gaps_ge_6h: int


def gap_report(df: pl.DataFrame, year: int) -> GapReport:
    """Compute a simple gap summary for a cleaned single-year DataFrame."""
    n = df.height
    hours_in_year = 8784 if _is_leap(year) else 8760
    times = df["time"].to_numpy()
    if times.size < 2:
        return GapReport(hours_in_year, n, 100.0 * n / hours_in_year, hours_in_year - n, 0)
    diffs_h = np.diff(times).astype("timedelta64[h]").astype(int)
    missing_stretches = diffs_h[diffs_h > 1] - 1
    longest = int(missing_stretches.max()) if missing_stretches.size else 0
    n_long = int((missing_stretches >= 6).sum())
    return GapReport(
        total_hours_expected=hours_in_year,
        total_hours_present=n,
        coverage_pct=100.0 * n / hours_in_year,
        longest_gap_hours=longest,
        n_gaps_ge_6h=n_long,
    )


def _is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0


@dataclass
class WindSummary:
    n: int
    mean_ms: float
    median_ms: float
    std_ms: float
    max_ms: float
    ti: float  # hour-to-hour variability std/mean; NOT meteorological turbulence intensity
    p95_ms: float


def summary(df: pl.DataFrame) -> WindSummary:
    s = df["wind_speed_ms"].to_numpy()
    mean = float(s.mean())
    return WindSummary(
        n=s.size,
        mean_ms=mean,
        median_ms=float(np.median(s)),
        std_ms=float(s.std(ddof=1)),
        max_ms=float(s.max()),
        ti=float(s.std(ddof=1) / mean) if mean > 0 else float("nan"),
        p95_ms=float(np.percentile(s, 95)),
    )


def weibull_fit(speeds_ms: np.ndarray) -> tuple[float, float]:
    """Fit a 2-parameter Weibull (loc fixed at 0) and return (shape k, scale c)."""
    positive = speeds_ms[speeds_ms > 0]
    k, _loc, c = stats.weibull_min.fit(positive, floc=0.0)
    return float(k), float(c)


def psd_welch(speeds_ms: np.ndarray, fs_hz: float = 1.0 / 3600.0) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD of hourly wind speeds. Returns (frequency Hz, power (m/s)^2/Hz)."""
    nperseg = min(len(speeds_ms), 24 * 30)
    f, pxx = signal.welch(speeds_ms - speeds_ms.mean(), fs=fs_hz, nperseg=nperseg)
    return f, pxx


def direction_bins(directions_deg: np.ndarray, n_bins: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Bin wind directions on a 0-360 circle. Bin 0 is centered on North.

    Returns (bin_centers_deg, counts).
    """
    bin_width = 360.0 / n_bins
    shifted = (directions_deg + bin_width / 2.0) % 360.0
    counts, _edges = np.histogram(shifted, bins=n_bins, range=(0.0, 360.0))
    centers = np.arange(n_bins) * bin_width
    return centers, counts


def as_utc(t: datetime | np.datetime64) -> datetime:
    if isinstance(t, np.datetime64):
        return t.astype("datetime64[s]").astype(datetime)
    return t
