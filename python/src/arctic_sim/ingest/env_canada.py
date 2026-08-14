"""REA Point hourly climate data — Environment Canada / ECCC.

Data source
-----------
Station: REA POINT, Melville Island, NU  (Climate ID 2403450, ~75.38 N, -105.72 E)

The raw CSVs under ``data/raw/env_canada/rea_point/`` were pulled manually from
the ECCC Climate Data Extraction Tool:

    https://climate-change.canada.ca/climate-data/#/hourly-climate-data

Steps used:
  1. Search for station "REA POINT".
  2. Select the "Hourly" frequency.
  3. Select the desired date range.  The tool caps exports at 10,000 rows per
     download, so multi-year ranges must be pulled in overlapping chunks.
  4. Export as CSV; save into ``data/raw/env_canada/rea_point/``.

The loader below globs every CSV in that directory, concatenates, and drops
any duplicate LOCAL_DATE rows produced by overlapping chunks.

Alternate sources (not currently used)
--------------------------------------
- ECCC GeoMET OGC/WFS API — programmatic access to the same hourly records.
- MSC Datamart bulk files (``dd.weather.gc.ca``) — raw daily/monthly archives.
Switch to one of these if the manual export becomes a bottleneck (e.g. yearly
refreshes, additional stations).

Units in raw CSVs
-----------------
- WIND_SPEED:     km/h        -> convert to m/s (divide by 3.6)
- WIND_DIRECTION: tens of degrees (0-36, 0 = calm)  -> multiply by 10 for degrees
- TEMP:           degrees C
- STATION_PRESSURE: kPa
Missing/flagged observations use ``M`` or blank.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

STATION_NAME = "REA POINT"
CLIMATE_ID = "2403450"

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[4] / "data" / "raw" / "env_canada" / "rea_point"


def load_rea_point(raw_dir: Path | None = None) -> pl.DataFrame:
    """Load and concatenate all REA Point hourly CSVs into one DataFrame.

    Returns a Polars DataFrame with LOCAL_DATE parsed as datetime, sorted
    chronologically, with duplicate LOCAL_DATE rows dropped.  Units are kept
    as in the raw file — apply :func:`to_si` at the consumer boundary.
    """
    src = raw_dir if raw_dir is not None else DEFAULT_RAW_DIR
    csvs = sorted(src.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no CSVs found in {src}")

    frames = [
        pl.read_csv(
            csv,
            try_parse_dates=True,
            infer_schema_length=10_000,
            null_values=["", "M", "NA"],
        )
        for csv in csvs
    ]
    return (
        pl.concat(frames)
        .unique(subset=["LOCAL_DATE"])
        .sort("LOCAL_DATE")
    )


def to_si(df: pl.DataFrame) -> pl.DataFrame:
    """Add SI-unit wind columns: ``wind_speed_ms`` and ``wind_direction_deg``."""
    return df.with_columns(
        (pl.col("WIND_SPEED").cast(pl.Float64) / 3.6).alias("wind_speed_ms"),
        (pl.col("WIND_DIRECTION").cast(pl.Float64) * 10.0).alias("wind_direction_deg"),
    )
