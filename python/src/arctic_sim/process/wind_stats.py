"""Cleaning and statistics for hourly wind observations.

Input is a Polars DataFrame from :func:`arctic_sim.ingest.env_canada.load_rea_point`
(or any other loader with the same schema).  Everything downstream uses SI units:
wind speed in m/s, direction in degrees (0=N, clockwise, 90=E).
"""

from __future__ import annotations

import math
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


IEC_I_REF = {"A+": 0.18, "A": 0.16, "B": 0.14, "C": 0.12}

# Isotropic von Karman: sigma_u^2 = (2/3) * integral(E(k) dk) = (2/3) * I * A * L^(2/3)
# where I = (1/2) * B(5/2, 1/3) is the beta-function integral of u^4/(1+u^2)^(17/6).
# (Anisotropic RDT enhances sigma_u by ~1.25x; recalibrate empirically per grid.)
_ISOTROPIC_VK_INTEGRAL = 0.5 * float(math.gamma(5.0 / 2.0) * math.gamma(1.0 / 3.0) / math.gamma(17.0 / 6.0))
_ISOTROPIC_SIGMA_FACTOR = (2.0 / 3.0) * _ISOTROPIC_VK_INTEGRAL


@dataclass
class MannParams:
    """IEC-informed Mann model parameters. Units: SI."""

    alpha_epsilon_23: float  # m^(4/3) / s^2
    length_scale_m: float
    gamma: float
    v_hub_ms: float
    z_hub_m: float
    z0_m: float
    ti_target: float
    sigma_u_target_ms: float
    iec_class: str
    notes: str


def derive_mann_params(
    v_measured_ms: float,
    *,
    z_measured_m: float = 10.0,
    z_hub_m: float = 30.0,
    z0_m: float = 0.001,
    iec_class: str = "C",
) -> MannParams:
    """Derive IEC-informed Mann model parameters at a target hub altitude.

    Steps (all per IEC 61400-1 Ed 4):
      1. Extrapolate mean wind from the measurement height to z_hub via the
         neutral log-law: U(z_hub) = U(z_meas) * ln(z_hub/z0) / ln(z_meas/z0).
      2. sigma_u_target from the Normal Turbulence Model:
             sigma_u = I_ref * (0.75 * V_hub + 5.6)
         where I_ref depends on the IEC turbulence class (C ~= offshore/smooth).
      3. Length scale L = 0.8 * Lambda_1  where  Lambda_1 = 0.7 * min(z_hub, 60).
      4. Gamma = 3.9 (IEC default for neutral atmospheric turbulence).
      5. alpha * epsilon^(2/3) from the isotropic von Karman relation
             sigma_u^2 = (2/3) * integral(E(k) dk) = 0.688 * A * L^(2/3)
         so  A = sigma_u^2 / (0.688 * L^(2/3)).

    Notes:
      - The 0.688 constant is exact for the infinite-grid isotropic case.
      - A finite grid loses the high-k spectral tail above Nyquist, so the
        empirical A on the target grid is typically 1.5-2x larger.
      - Anisotropic RDT (Mann's Gamma) enhances sigma_u further by ~1.25x
        relative to isotropic, so anisotropic A is smaller by ~1.5x.
      - Always refine on the actual grid: A_new = A_old * (sigma_target / sigma_measured)^2.
    """
    if iec_class not in IEC_I_REF:
        raise ValueError(f"unknown IEC class {iec_class!r}; expected one of {list(IEC_I_REF)}")
    i_ref = IEC_I_REF[iec_class]

    v_hub = v_measured_ms * math.log(z_hub_m / z0_m) / math.log(z_measured_m / z0_m)
    sigma_u = i_ref * (0.75 * v_hub + 5.6)
    ti = sigma_u / v_hub

    lambda_1 = 0.7 * min(z_hub_m, 60.0)
    length_scale = 0.8 * lambda_1
    gamma = 3.9
    ae23 = sigma_u ** 2 / (_ISOTROPIC_SIGMA_FACTOR * length_scale ** (2.0 / 3.0))

    return MannParams(
        alpha_epsilon_23=ae23,
        length_scale_m=length_scale,
        gamma=gamma,
        v_hub_ms=v_hub,
        z_hub_m=z_hub_m,
        z0_m=z0_m,
        ti_target=ti,
        sigma_u_target_ms=sigma_u,
        iec_class=iec_class,
        notes=(
            "Isotropic infinite-grid asymptote. Finite grids typically need 1.5-2x "
            "more alpha; anisotropic RDT needs ~1.5x less. Refine on target grid: "
            "A_new = A_old * (sigma_target / sigma_measured)^2."
        ),
    )
