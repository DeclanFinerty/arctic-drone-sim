"""Visualize the ETOPO 2022 ice-surface elevation tile covering REA Point.
Marks the station and a candidate simulation domain footprint, and derives a
surface-roughness (z0) map from elevation for wind-profile modeling.

Usage:
    uv run python scripts/visualize_bathy.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

REPO = Path(__file__).resolve().parents[2]
BATHY = REPO / "data" / "bathy" / "ETOPO_2022_IceSurface_15as.tiff"
OUT_DIR = REPO / "output" / "plots" / "bathy"

# REA Point station (from ECCC CSV metadata).
REA_LAT = 75.377
REA_LON = -105.715

# Sim domain footprint in meters (matches wind-field grid).
SIM_SIZE_M = 1280.0
# 1 degree latitude ~ 111 km; 1 degree longitude at 75N ~ 28.7 km.
DEG_LAT_PER_M = 1.0 / 111_000.0
DEG_LON_PER_M = 1.0 / (111_000.0 * np.cos(np.radians(REA_LAT)))


def elevation_to_z0(elev_m: np.ndarray) -> np.ndarray:
    """Map elevation to surface roughness length.

    Rough categorization for this Arctic tile:
      - Sea / sea ice (elev < 0):    z0 = 0.0005 m (snow-covered ice, calm)
      - Coastal margin (0-5 m):      z0 = 0.01 m  (transitional)
      - Land, low relief (5-100 m):  z0 = 0.05 m  (tundra, gravel)
      - Land, high relief (>100 m):  z0 = 0.15 m  (broken terrain)
    """
    z0 = np.full_like(elev_m, 0.0005, dtype=np.float64)
    z0[elev_m >= 0] = 0.01
    z0[elev_m >= 5] = 0.05
    z0[elev_m >= 100] = 0.15
    return z0


def load_bathy():
    with rasterio.open(BATHY) as ds:
        elev = ds.read(1).astype(np.float64)
        left, bottom, right, top = ds.bounds
    lons = np.linspace(left, right, elev.shape[1])
    lats = np.linspace(top, bottom, elev.shape[0])  # rasterio rows go top-down
    return elev, lons, lats


def plot_elevation(elev, lons, lats):
    fig, ax = plt.subplots(figsize=(11, 6))
    norm = TwoSlopeNorm(vmin=elev.min(), vcenter=0.0, vmax=max(elev.max(), 10.0))
    # Ocean cool blues, land warm browns/greens.
    cmap = LinearSegmentedColormap.from_list("landsea", [
        "#08306b", "#4292c6", "#c6dbef", "#e5e5e5",
        "#f7ecc1", "#c2a875", "#8c5a3c", "#3d2818",
    ])
    im = ax.imshow(elev, extent=(lons[0], lons[-1], lats[-1], lats[0]),
                   origin="upper", cmap=cmap, norm=norm, aspect="auto")
    fig.colorbar(im, ax=ax, label="Elevation [m] (negative = sea depth)")

    ax.contour(lons, lats, elev, levels=[0.0], colors="black", linewidths=0.7)

    ax.scatter([REA_LON], [REA_LAT], s=180, marker="*", color="#ffcc00",
               edgecolor="black", zorder=5, label="REA Point (75.377°N)")

    dlat = 0.5 * SIM_SIZE_M * DEG_LAT_PER_M
    dlon = 0.5 * SIM_SIZE_M * DEG_LON_PER_M
    ax.plot([REA_LON - dlon, REA_LON + dlon, REA_LON + dlon,
             REA_LON - dlon, REA_LON - dlon],
            [REA_LAT - dlat, REA_LAT - dlat, REA_LAT + dlat,
             REA_LAT + dlat, REA_LAT - dlat],
            color="#d62728", linewidth=1.5,
            label=f"Sim domain ({SIM_SIZE_M:.0f} m)")

    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    ax.set_title("ETOPO 2022 ice-surface elevation around REA Point, Melville Island")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "elevation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_z0(elev, lons, lats):
    z0 = elevation_to_z0(elev)
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(z0, extent=(lons[0], lons[-1], lats[-1], lats[0]),
                   origin="upper", cmap="YlOrBr", aspect="auto",
                   norm=plt.matplotlib.colors.LogNorm(vmin=z0.min(), vmax=z0.max()))
    fig.colorbar(im, ax=ax, label="Surface roughness z0 [m] (log scale)")

    ax.contour(lons, lats, elev, levels=[0.0], colors="black", linewidths=0.7)
    ax.scatter([REA_LON], [REA_LAT], s=180, marker="*", color="#ffcc00",
               edgecolor="black", zorder=5, label="REA Point")
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    ax.set_title("Derived surface roughness z0 from elevation classes")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "z0_map.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def extract_sim_domain(elev, lons, lats, domain_size_m: float = SIM_SIZE_M,
                        n: int = 128, tag: str = ""):
    """Sample bathy over the sim footprint, return (elev, z0) grids at the
    wind-field resolution. Also save a JSON sidecar for downstream use.
    """
    xs_m = np.linspace(-domain_size_m / 2, domain_size_m / 2, n)
    ys_m = np.linspace(-domain_size_m / 2, domain_size_m / 2, n)
    xx_m, yy_m = np.meshgrid(xs_m, ys_m, indexing="ij")
    sample_lon = REA_LON + xx_m * DEG_LON_PER_M
    sample_lat = REA_LAT + yy_m * DEG_LAT_PER_M

    i = np.clip(np.searchsorted(lats[::-1], sample_lat), 0, len(lats) - 1)
    i = len(lats) - 1 - i
    j = np.clip(np.searchsorted(lons, sample_lon), 0, len(lons) - 1)
    elev_domain = elev[i, j]
    z0_domain = elevation_to_z0(elev_domain)

    out_dir = REPO / "data" / "processed" / "bathy"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    np.save(out_dir / f"elevation_m{suffix}.npy", elev_domain)
    np.save(out_dir / f"z0_m{suffix}.npy", z0_domain)
    (out_dir / f"metadata{suffix}.json").write_text(json.dumps({
        "shape": [n, n],
        "origin_m": [0.0, 0.0],
        "spacing_m": [domain_size_m / (n - 1), domain_size_m / (n - 1)],
        "domain_size_m": domain_size_m,
        "center_lat": REA_LAT,
        "center_lon": REA_LON,
        "source": "ETOPO 2022 IceSurface 15as, sampled by nearest neighbor",
        "z0_scheme": "elevation<0: 0.0005; 0-5m: 0.01; 5-100m: 0.05; >100m: 0.15",
    }, indent=2))
    print(f"wrote sim-domain bathy ({tag or 'default'}) to {out_dir}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    extent_m = (xs_m[0], xs_m[-1], ys_m[0], ys_m[-1])
    im0 = axes[0].imshow(elev_domain.T, origin="lower", extent=extent_m,
                          cmap="terrain", aspect="equal")
    fig.colorbar(im0, ax=axes[0], label="Elevation [m]")
    axes[0].contour(xs_m, ys_m, elev_domain.T, levels=[0.0],
                    colors="black", linewidths=1)
    axes[0].scatter([0], [0], s=180, marker="*", color="#ffcc00",
                    edgecolor="black", zorder=5, label="REA Point (domain center)")
    axes[0].set_title(f"Sim-domain elevation ({domain_size_m:.0f} m, centered on REA Point)")
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("y [m]")
    axes[0].legend(loc="upper right", fontsize=8)

    im1 = axes[1].imshow(z0_domain.T, origin="lower", extent=extent_m,
                          cmap="YlOrBr", aspect="equal",
                          norm=plt.matplotlib.colors.LogNorm(
                              vmin=z0_domain.min(), vmax=z0_domain.max()))
    fig.colorbar(im1, ax=axes[1], label="z0 [m] (log)")
    axes[1].contour(xs_m, ys_m, elev_domain.T, levels=[0.0],
                    colors="black", linewidths=1)
    axes[1].set_title("Sim-domain surface roughness z0")
    axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("y [m]")

    fig.tight_layout()
    out = OUT_DIR / f"sim_domain{'_' + tag if tag else ''}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    frac_sea = float(np.mean(elev_domain < 0))
    print(f"sim domain: {frac_sea * 100:.1f}% sea, "
          f"{(1 - frac_sea) * 100:.1f}% land   "
          f"elev min={elev_domain.min():.1f}, max={elev_domain.max():.1f}   "
          f"z0 range=[{z0_domain.min():.4g}, {z0_domain.max():.4g}] m")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    elev, lons, lats = load_bathy()
    print(f"loaded bathy: {elev.shape} pixels, "
          f"lon [{lons[0]:.2f}, {lons[-1]:.2f}], "
          f"lat [{lats[-1]:.2f}, {lats[0]:.2f}]")
    plot_elevation(elev, lons, lats)
    plot_z0(elev, lons, lats)
    extract_sim_domain(elev, lons, lats, domain_size_m=SIM_SIZE_M, n=128, tag="1280m")
    extract_sim_domain(elev, lons, lats, domain_size_m=10_000.0, n=128, tag="10km")


if __name__ == "__main__":
    main()
