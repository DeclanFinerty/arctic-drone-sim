"""REA Point 2024 wind climatology: cleaning, summary, and plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from arctic_sim.ingest.env_canada import load_rea_point
from arctic_sim.process.wind_stats import (
    clean,
    direction_bins,
    gap_report,
    psd_welch,
    subset_year,
    summary,
    weibull_fit,
)

YEAR = 2024
OUT = Path(__file__).resolve().parents[2] / "output" / "plots" / f"wind_{YEAR}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    raw = load_rea_point()
    year_df = subset_year(raw, YEAR)
    df = clean(year_df)

    gaps = gap_report(df, YEAR)
    stats_ = summary(df)
    speeds = df["wind_speed_ms"].to_numpy()
    dirs = df.filter(df["wind_direction_deg"].is_not_null())["wind_direction_deg"].to_numpy()

    k, c = weibull_fit(speeds)

    print(f"--- REA Point {YEAR} wind climatology ---")
    print(f"coverage: {gaps.total_hours_present}/{gaps.total_hours_expected}"
          f" ({gaps.coverage_pct:.1f}%), longest gap {gaps.longest_gap_hours}h,"
          f" {gaps.n_gaps_ge_6h} gaps >= 6h")
    print(f"mean {stats_.mean_ms:.2f} m/s, median {stats_.median_ms:.2f}, "
          f"std {stats_.std_ms:.2f}, max {stats_.max_ms:.2f}, p95 {stats_.p95_ms:.2f}")
    print(f"hour-to-hour variability (std/mean): {stats_.ti:.3f}")
    print(f"Weibull fit: k={k:.3f}, c={c:.3f}")

    plot_time_series(df, stats_)
    plot_wind_rose(dirs, speeds[df["wind_direction_deg"].is_not_null().to_numpy()])
    plot_weibull_hist(speeds, k, c, stats_)
    plot_psd(speeds)
    print(f"wrote plots to {OUT}")


def plot_time_series(df, stats_) -> None:
    times = df["time"].to_numpy()
    speed = df["wind_speed_ms"].to_numpy()
    direction = df["wind_direction_deg"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(times, speed, linewidth=0.4, color="#1f6feb")
    axes[0].axhline(stats_.mean_ms, color="#d62728", linestyle="--", linewidth=0.8,
                    label=f"mean {stats_.mean_ms:.2f} m/s")
    axes[0].set_ylabel("Wind speed [m/s]")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(times, direction, s=1.5, color="#2ca02c", alpha=0.4)
    axes[1].set_yticks([0, 90, 180, 270, 360])
    axes[1].set_yticklabels(["N", "E", "S", "W", "N"])
    axes[1].set_ylim(0, 360)
    axes[1].set_ylabel("Wind direction")
    axes[1].grid(alpha=0.3)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[-1].set_xlabel(str(YEAR))

    fig.suptitle(f"REA Point {YEAR} — hourly wind speed and direction", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "time_series.png", dpi=150)
    plt.close(fig)


def plot_wind_rose(directions_deg: np.ndarray, speeds_ms: np.ndarray) -> None:
    n_bins = 16
    bin_width = 360.0 / n_bins
    centers, counts = direction_bins(directions_deg, n_bins=n_bins)

    speed_bands = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 30)]
    colors = ["#c6dbef", "#6baed6", "#2171b5", "#08519c", "#08306b"]

    shifted = (directions_deg + bin_width / 2.0) % 360.0
    bin_idx = np.clip((shifted // bin_width).astype(int), 0, n_bins - 1)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    theta = np.deg2rad(centers)

    bottom = np.zeros(n_bins)
    for (lo, hi), color in zip(speed_bands, colors, strict=True):
        band_mask = (speeds_ms >= lo) & (speeds_ms < hi)
        band_counts = np.bincount(bin_idx[band_mask], minlength=n_bins)
        band_pct = 100.0 * band_counts / len(speeds_ms)
        ax.bar(theta, band_pct, width=np.deg2rad(bin_width * 0.9),
               bottom=bottom, color=color, edgecolor="white", linewidth=0.4,
               label=f"{lo}-{hi if hi < 30 else '∞'} m/s")
        bottom += band_pct

    ax.set_title(f"REA Point {YEAR} wind rose  (n={len(speeds_ms)})", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "wind_rose.png", dpi=150)
    plt.close(fig)


def plot_weibull_hist(speeds: np.ndarray, k: float, c: float, stats_) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, max(30.0, speeds.max()), 40)
    ax.hist(speeds, bins=bins, density=True, color="#1f6feb", alpha=0.7,
            edgecolor="white", linewidth=0.3, label="observed")
    x = np.linspace(0.01, bins[-1], 500)
    ax.plot(x, stats.weibull_min.pdf(x, k, loc=0.0, scale=c), color="#d62728",
            linewidth=1.6, label=f"Weibull fit  k={k:.2f}, c={c:.2f} m/s")
    ax.axvline(stats_.mean_ms, color="#333", linestyle="--", linewidth=0.8,
               label=f"mean {stats_.mean_ms:.2f} m/s")
    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Probability density")
    ax.set_title(f"REA Point {YEAR} wind speed distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "weibull_fit.png", dpi=150)
    plt.close(fig)


def plot_psd(speeds: np.ndarray) -> None:
    f, pxx = psd_welch(speeds)
    positive = f > 0
    period_h = 1.0 / (f[positive] * 3600.0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.loglog(period_h, pxx[positive], color="#1f6feb")
    for marker_h, label in [(24, "1 day"), (24 * 7, "1 week"), (24 * 30, "1 month")]:
        ax.axvline(marker_h, color="#888", linewidth=0.6, linestyle="--")
        ax.text(marker_h, ax.get_ylim()[1] * 0.5, label,
                rotation=90, va="top", ha="right", fontsize=8, color="#555")
    ax.set_xlabel("Period [hours]")
    ax.set_ylabel(r"PSD $[(m/s)^2 / Hz]$")
    ax.set_title(f"REA Point {YEAR} wind speed PSD (Welch, hourly)")
    ax.grid(which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "psd.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
