"""Compare station-keep performance across terrain types by log-law-scaling
the drone-altitude mean wind for each surface roughness.

For a given synoptic driver (encoded as observed 10 m wind), the wind at the
drone's 30 m altitude depends on local surface roughness z0. Rougher terrain
= more shear = lower wind at height (for same u*). This script fixes the
synoptic condition, computes what the drone actually sees at each of three
terrain sites, and reports how station-keep error and battery scale.

Usage:
    uv run python scripts/sweep_terrain.py
    uv run python scripts/sweep_terrain.py --observed 3,5,7,9,11,13
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASELINE_MEAN_U = 6.4  # embedded mean of the Mann field

# Site classification: three z0 values covering the range visible in the
# 10 km bathy sample around REA Point.
SITES = [
    ("sea_ice",  0.0005, "#1f6feb", "Sea ice (z0=0.0005 m)"),
    ("tundra",   0.05,   "#2ca02c", "Tundra   (z0=0.05 m)   -- REA Point local"),
    ("rocky",    0.15,   "#8b4513", "Rocky    (z0=0.15 m)   -- inland hills"),
]

# Log law: U(z) = u*/kappa * ln(z/z0). Anchor u* to REA Point's observed 10 m
# wind under the site's z0. If we assume the same synoptic driver (same u*)
# across sites within the 10 km bathy sample, drone-altitude wind differs
# purely through the log-law ratio.
KAPPA = 0.4
Z_OBS = 10.0
Z_DRONE = 30.0
REA_Z0 = 0.05


def drone_wind_at_site(observed_10m_ms: float, z0_site: float) -> float:
    u_star = KAPPA * observed_10m_ms / np.log(Z_OBS / REA_Z0)
    return u_star / KAPPA * np.log(Z_DRONE / z0_site)


CONFIG_TEMPLATE = """\
seed = 42

[wind_field]
load_from = "output/wind_fields/mann_anisotropic"
taylor_advection_ms = [{adv_u}, 0.0, 0.0]
mean_offset_ms = [{offset_u}, 0.0, 0.0]
turbulence_scale = 1.0

[drone]
mass_kg = 1.5
max_thrust_n = 30.0
max_wind_speed_ms = 15.0
max_speed_ms = 20.0
battery_capacity_wh = 100.0
drag_area_m2 = 0.05
drag_coefficient = 1.0
air_density_kgm3 = 1.29
power_per_thrust_w_per_n = 6.7

[controller]
kp = 10.0
kv = 6.0
ki = 0.5

[mission]
type = "station_keep"
target_m = [640.0, 640.0, 30.0]
tolerance_m = 1.0
initial_position_m = [640.0, 640.0, 30.0]

[simulation]
duration_s = 60.0
dt_s = 0.01
"""


def run_one(v_drone_ms: float, workdir: Path, tag: str) -> dict:
    offset_u = v_drone_ms - BASELINE_MEAN_U
    cfg_path = workdir / f"cfg_{tag}.toml"
    run_name = f"terrain_{tag}"
    cfg_path.write_text(CONFIG_TEMPLATE.format(adv_u=v_drone_ms, offset_u=offset_u))
    subprocess.run(
        ["cargo", "run", "--release", "--quiet", "-p", "sim-engine", "--",
         "--config", str(cfg_path), "--run-name", run_name],
        cwd=REPO, check=True, capture_output=True,
    )
    meta = json.loads((REPO / "output" / "runs" / run_name / "metadata.json").read_text())
    battery = np.load(REPO / "output" / "runs" / run_name / "battery_wh.npy")
    return {
        "v_drone_ms": v_drone_ms,
        "score_m": meta["score"],
        "battery_wh": float(battery[0] - battery[-1]),
        "mean_wind_ms": meta["mean_wind_ms"],
    }


def plot_curves(observed: np.ndarray, results: dict, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for name, z0, color, label in SITES:
        v_drone = np.array([r["v_drone_ms"] for r in results[name]])
        score = np.array([r["score_m"] for r in results[name]])
        battery = np.array([r["battery_wh"] for r in results[name]])

        axes[0].plot(observed, v_drone, "o-", color=color, label=label)
        axes[1].plot(observed, score, "o-", color=color, label=label)
        axes[2].plot(observed, battery, "o-", color=color, label=label)

    axes[0].set_xlabel("Observed 10 m wind (synoptic driver) [m/s]")
    axes[0].set_ylabel("Drone-altitude (30 m) wind [m/s]")
    axes[0].set_title("Log-law scaling of drone wind by terrain")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Observed 10 m wind (synoptic driver) [m/s]")
    axes[1].set_ylabel("Station-keep RMS error [m]")
    axes[1].set_title("Controller error vs terrain")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)

    axes[2].set_xlabel("Observed 10 m wind (synoptic driver) [m/s]")
    axes[2].set_ylabel("Battery drained in 60 s [Wh]")
    axes[2].set_title("Energy cost vs terrain")
    axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)

    fig.suptitle("Same synoptic wind → different drone experience by terrain",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed", type=str, default="3,5,7,9,11,13",
                        help="observed 10 m wind speeds (m/s) at REA Point")
    args = parser.parse_args()
    observed = np.array([float(x) for x in args.observed.split(",")])

    workdir = Path(tempfile.mkdtemp(prefix="arctic_terrain_"))
    print(f"3 sites x {len(observed)} synoptic points = "
          f"{3 * len(observed)} runs")
    results = {name: [] for name, _, _, _ in SITES}
    try:
        for u_obs in observed:
            for name, z0, _, _ in SITES:
                v_drone = drone_wind_at_site(float(u_obs), z0)
                tag = f"{name}_obs{u_obs:.1f}"
                r = run_one(v_drone, workdir, tag)
                r["z0"] = z0
                r["observed_10m_ms"] = float(u_obs)
                results[name].append(r)
                print(f"  {name:8s}  U_obs={u_obs:5.1f}  "
                      f"z0={z0:.4f}  U_drone={v_drone:5.2f}  "
                      f"RMS={r['score_m']:.3f} m  batt={r['battery_wh']:.3f} Wh")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    out_dir = REPO / "output" / "plots" / "sweeps" / "terrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "terrain_results.json").write_text(json.dumps({
        "observed_10m_ms": observed.tolist(),
        "sites": {name: [{k: v for k, v in r.items()} for r in results[name]]
                   for name, _, _, _ in SITES},
    }, indent=2))
    plot_curves(observed, results, out_dir / "terrain_comparison.png")
    print(f"wrote {out_dir / 'terrain_comparison.png'}")


if __name__ == "__main__":
    main()
