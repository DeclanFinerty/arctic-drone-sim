"""Sweep station-keep performance across mean wind speed and turbulence
intensity. Writes a temporary config per grid point, invokes the sim-engine
CLI, collects score + battery + termination, and plots heatmaps.

Usage:
    uv run python scripts/sweep_station_keep.py
    uv run python scripts/sweep_station_keep.py --v 2,5,8,11,14 --ti 0.5,1.0,1.5,2.0
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
BASELINE_MEAN_U = 6.4  # embedded mean of the loaded Mann field
BASELINE_SIGMA_U = 1.183  # measured from the anisotropic field at gen time

CONFIG_TEMPLATE = """\
seed = 42

[wind_field]
load_from = "output/wind_fields/mann_anisotropic"
taylor_advection_ms = [{adv_u}, 0.0, 0.0]
mean_offset_ms = [{offset_u}, 0.0, 0.0]
turbulence_scale = {ti_scale}

[drone]
mass_kg = 1.5
max_thrust_n = {max_thrust_n}
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
target_m = [640.0, 640.0, {alt_m}]
tolerance_m = 1.0
initial_position_m = [640.0, 640.0, {alt_m}]

[simulation]
duration_s = {duration_s}
dt_s = 0.01
"""


def parse_floats(spec: str) -> np.ndarray:
    return np.array([float(x) for x in spec.split(",")])


def run_one(v_mean: float, ti_scale: float, duration_s: float, workdir: Path,
            max_thrust_n: float, alt_m: float, tag: str) -> dict:
    offset_u = v_mean - BASELINE_MEAN_U
    cfg_path = workdir / f"cfg_{tag}_v{v_mean:.1f}_ti{ti_scale:.2f}.toml"
    run_name = f"sweep_{tag}_v{v_mean:.1f}_ti{ti_scale:.2f}"
    cfg_path.write_text(CONFIG_TEMPLATE.format(
        adv_u=v_mean,      # Taylor advection at the target mean
        offset_u=offset_u,
        ti_scale=ti_scale,
        max_thrust_n=max_thrust_n,
        alt_m=alt_m,
        duration_s=duration_s,
    ))

    cmd = [
        "cargo", "run", "--release", "--quiet", "-p", "sim-engine", "--",
        "--config", str(cfg_path),
        "--run-name", run_name,
    ]
    subprocess.run(cmd, cwd=REPO, check=True, capture_output=True)

    meta = json.loads((REPO / "output" / "runs" / run_name / "metadata.json").read_text())
    battery = np.load(REPO / "output" / "runs" / run_name / "battery_wh.npy")
    drained = float(battery[0] - battery[-1])
    return {
        "v_mean": v_mean,
        "ti_scale": ti_scale,
        "score_m": meta["score"],
        "battery_wh": drained,
        "mean_wind_ms": meta["mean_wind_ms"],
        "terminated": meta["terminated_reason"],
    }


def plot_heatmap(v: np.ndarray, ti: np.ndarray, values: np.ndarray, title: str,
                 label: str, cmap: str, out_path: Path, fmt: str = "{:.2f}") -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(values.T, origin="lower", aspect="auto", cmap=cmap,
                   extent=(v[0] - 0.5 * (v[1] - v[0]),
                           v[-1] + 0.5 * (v[1] - v[0]),
                           ti[0] - 0.5 * (ti[1] - ti[0]),
                           ti[-1] + 0.5 * (ti[1] - ti[0])))
    fig.colorbar(im, ax=ax, label=label)
    for i, vv in enumerate(v):
        for j, tt in enumerate(ti):
            ax.text(vv, tt, fmt.format(values[i, j]), ha="center", va="center",
                    color="white", fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.35, pad=1, edgecolor="none"))
    ax.set_xlabel("Mean wind speed [m/s]")
    ax.set_ylabel("Turbulence scale (× IEC baseline)")
    ax.set_xticks(v)
    ax.set_yticks(ti)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v", type=str, default="2,5,8,11,14",
                        help="mean wind speeds (m/s) to sweep")
    parser.add_argument("--ti", type=str, default="0.5,1.0,1.5,2.0",
                        help="turbulence-scale multipliers to sweep")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--max-thrust", type=float, default=30.0,
                        help="drone max thrust (N). Nominal 30; 15 = severely degraded")
    parser.add_argument("--alt", type=float, default=30.0,
                        help="hover altitude (m)")
    parser.add_argument("--tag", type=str, default="baseline",
                        help="sweep label — plots + json go to sweeps/<tag>/")
    args = parser.parse_args()

    v_mesh = parse_floats(args.v)
    ti_mesh = parse_floats(args.ti)
    print(f"sweep: {len(v_mesh)} x {len(ti_mesh)} = {len(v_mesh) * len(ti_mesh)} runs")

    workdir = Path(tempfile.mkdtemp(prefix="arctic_sweep_"))
    try:
        results = []
        for i, v in enumerate(v_mesh):
            for j, ti in enumerate(ti_mesh):
                r = run_one(float(v), float(ti), args.duration, workdir,
                            args.max_thrust, args.alt, args.tag)
                print(f"  v={v:5.1f} ti={ti:4.2f}  score={r['score_m']:6.3f} m  "
                      f"battery={r['battery_wh']:.3f} Wh  mean|wind|={r['mean_wind_ms']:.2f}"
                      f"  {r['terminated']}")
                results.append(r)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    score = np.zeros((len(v_mesh), len(ti_mesh)))
    battery = np.zeros_like(score)
    wind = np.zeros_like(score)
    for r in results:
        i = int(np.where(v_mesh == r["v_mean"])[0][0])
        j = int(np.where(ti_mesh == r["ti_scale"])[0][0])
        score[i, j] = r["score_m"]
        battery[i, j] = r["battery_wh"]
        wind[i, j] = r["mean_wind_ms"]

    out_dir = REPO / "output" / "plots" / "sweeps" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "sweep_results.json").write_text(json.dumps({
        "tag": args.tag,
        "max_thrust_n": args.max_thrust,
        "alt_m": args.alt,
        "v_mesh": v_mesh.tolist(),
        "ti_mesh": ti_mesh.tolist(),
        "score_rms_m": score.tolist(),
        "battery_wh": battery.tolist(),
        "mean_wind_ms": wind.tolist(),
    }, indent=2))

    subtitle = f"max_thrust={args.max_thrust:.0f} N, altitude={args.alt:.0f} m"
    plot_heatmap(v_mesh, ti_mesh, score,
                 f"Station-keep RMS error — {subtitle}",
                 "RMS position error [m]", "viridis",
                 out_dir / "sweep_score_rms.png")
    plot_heatmap(v_mesh, ti_mesh, battery,
                 f"Battery drained — {subtitle}",
                 "Battery [Wh]", "magma",
                 out_dir / "sweep_battery.png", fmt="{:.2f}")
    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
