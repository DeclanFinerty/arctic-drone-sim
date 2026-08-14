"""Plot monthly REA Point wind observation coverage and highlight 2025."""
from __future__ import annotations

import calendar
from datetime import date
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import polars as pl

from arctic_sim.ingest.env_canada import load_rea_point

HIGHLIGHT_YEAR = 2024
OUTPUT = Path(__file__).resolve().parents[2] / "output" / "plots" / "coverage" / "rea_point_coverage.png"


def monthly_coverage(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Return one row per (year, month) with `pct` = valid hours / hours in month."""
    per_month = (
        df.with_columns(
            pl.col("LOCAL_DATE").dt.year().alias("year"),
            pl.col("LOCAL_DATE").dt.month().alias("month"),
        )
        .group_by(["year", "month"])
        .agg(pl.col(column).is_not_null().sum().alias("valid"))
        .sort(["year", "month"])
    )
    hours_in_month = [
        24 * calendar.monthrange(y, m)[1]
        for y, m in zip(per_month["year"], per_month["month"], strict=True)
    ]
    return per_month.with_columns(
        pl.Series("hours_in_month", hours_in_month),
    ).with_columns(
        (100.0 * pl.col("valid") / pl.col("hours_in_month")).alias("pct"),
        pl.date(pl.col("year"), pl.col("month"), 1).alias("month_start"),
    )


def main() -> None:
    df = load_rea_point()
    print(f"loaded {df.height} hourly rows  {df['LOCAL_DATE'].min()} -> {df['LOCAL_DATE'].max()}")

    speed = monthly_coverage(df, "WIND_SPEED")
    direction = monthly_coverage(df, "WIND_DIRECTION")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, series, label in [
        (axes[0], speed, "Wind speed"),
        (axes[1], direction, "Wind direction"),
    ]:
        dates = series["month_start"].to_list()
        pcts = series["pct"].to_list()
        ax.bar(dates, pcts, width=28, color="#1f6feb", edgecolor="none")
        ax.axhline(100, color="#888", linewidth=0.7, linestyle="--")
        ax.axvspan(
            date(HIGHLIGHT_YEAR, 1, 1),
            date(HIGHLIGHT_YEAR + 1, 1, 1),
            color="#ffd166",
            alpha=0.35,
            zorder=0,
        )
        ax.set_ylim(0, 110)
        ax.set_ylabel(f"{label}\n% of hours")
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Month")

    highlight_patch = mpatches.Patch(color="#ffd166", alpha=0.35, label=f"{HIGHLIGHT_YEAR}")
    bar_patch = mpatches.Patch(color="#1f6feb", label="Monthly coverage")
    fig.legend(handles=[bar_patch, highlight_patch], loc="upper right", ncol=2)
    fig.suptitle(
        f"REA Point (Climate ID 2403450) — hourly wind observation coverage\n"
        f"{df['LOCAL_DATE'].min().date()}  to  {df['LOCAL_DATE'].max().date()}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
