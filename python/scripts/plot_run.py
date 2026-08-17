"""Visualize a simulation run: trajectory overlay on wind slice, position
error time series, control effort, battery.

Usage:
    uv run python scripts/plot_run.py                    # default: output/runs/station_keep
    uv run python scripts/plot_run.py <run_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "output" / "runs" / "station_keep"
WIND_DIR = REPO / "output" / "wind_fields" / "mann_anisotropic"
PLOTS = REPO / "output" / "plots" / f"run_{RUN.name}"


def load_run():
    meta = json.loads((RUN / "metadata.json").read_text())
    return {
        "times": np.load(RUN / "times.npy"),
        "positions": np.load(RUN / "positions.npy"),
        "velocities": np.load(RUN / "velocities.npy"),
        "battery": np.load(RUN / "battery_wh.npy"),
        "commanded_force": np.load(RUN / "commanded_force_n.npy"),
        "wind_at_drone": np.load(RUN / "wind_at_drone_ms.npy"),
        "meta": meta,
    }


def load_wind_slice(z_target_m: float):
    meta = json.loads((WIND_DIR / "metadata.json").read_text())
    u = np.load(WIND_DIR / "u.npy")
    v = np.load(WIND_DIR / "v.npy")
    origin = np.array(meta["origin"])
    spacing = np.array(meta["spacing"])
    shape = tuple(meta["shape"])
    k = int(round((z_target_m - origin[2]) / spacing[2]))
    k = max(0, min(shape[2] - 1, k))
    x = origin[0] + np.arange(shape[0]) * spacing[0]
    y = origin[1] + np.arange(shape[1]) * spacing[1]
    return {"x": x, "y": y, "u_slice": u[:, :, k], "v_slice": v[:, :, k],
            "z_used_m": float(origin[2] + k * spacing[2])}


def plot_trajectory(run) -> None:
    target = np.array(run["meta"]["target_m"])
    ws = load_wind_slice(target[2])
    speed = np.sqrt(ws["u_slice"] ** 2 + ws["v_slice"] ** 2)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(
        speed.T,
        origin="lower",
        extent=(ws["x"][0], ws["x"][-1], ws["y"][0], ws["y"][-1]),
        cmap="Blues",
        alpha=0.85,
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="Horizontal wind speed [m/s]")

    step = max(1, len(ws["x"]) // 16)
    xs, ys = np.meshgrid(ws["x"][::step], ws["y"][::step], indexing="ij")
    ax.quiver(xs, ys, ws["u_slice"][::step, ::step], ws["v_slice"][::step, ::step],
              color="#333", alpha=0.4, scale=200, width=0.002)

    px, py = run["positions"][:, 0], run["positions"][:, 1]
    ax.plot(px, py, color="#d62728", linewidth=1.2, label="Drone trajectory")
    ax.scatter([px[0]], [py[0]], color="#2ca02c", s=60, marker="o",
               edgecolor="black", zorder=5, label="Start")
    ax.scatter([target[0]], [target[1]], color="#ffcc00", s=120, marker="*",
               edgecolor="black", zorder=5, label="Target")

    # Frame around the whole trajectory + target with a generous margin so
    # the wind field context is visible even for near-hover runs.
    x_all = np.concatenate([px, [target[0]]])
    y_all = np.concatenate([py, [target[1]]])
    span = max(np.ptp(x_all), np.ptp(y_all), 20.0)
    pad = 0.6 * span
    cx = (x_all.min() + x_all.max()) / 2
    cy = (y_all.min() + y_all.max()) / 2
    ax.set_xlim(cx - pad, cx + pad)
    ax.set_ylim(cy - pad, cy + pad)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Drone trajectory over wind slice at z = {ws['z_used_m']:.0f} m")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "trajectory.png", dpi=150)
    plt.close(fig)


def plot_time_series(run) -> None:
    t = run["times"]
    pos = run["positions"]
    target = np.array(run["meta"]["target_m"])
    err = pos - target[None, :]
    err_mag = np.linalg.norm(err, axis=1)
    force_mag = np.linalg.norm(run["commanded_force"], axis=1)
    wind_mag = np.linalg.norm(run["wind_at_drone"], axis=1)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, err[:, 0], label="x error", color="#1f6feb")
    axes[0].plot(t, err[:, 1], label="y error", color="#2ca02c")
    axes[0].plot(t, err[:, 2], label="z error", color="#d62728")
    axes[0].plot(t, err_mag, label="|error|", color="black", linewidth=1.2, linestyle="--")
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].axhline(run["meta"]["tolerance_m"], color="gray", linewidth=0.4, linestyle=":")
    axes[0].axhline(-run["meta"]["tolerance_m"], color="gray", linewidth=0.4, linestyle=":")
    axes[0].set_ylabel("Position error [m]")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_title(f"RMS error = {run['meta']['score']:.3f} m,   tolerance = {run['meta']['tolerance_m']} m")

    # Split force into horizontal (fx, fy) and vertical (fz - hover) so
    # turbulence response is visible instead of drowned by the ~mg z-thrust.
    force = run["commanded_force"]
    t_ctrl = t[1:1 + len(force)]
    hover_n = 1.5 * 9.81
    axes[1].plot(t_ctrl, force[:, 0], label="fx", color="#1f6feb", linewidth=0.7)
    axes[1].plot(t_ctrl, force[:, 1], label="fy", color="#2ca02c", linewidth=0.7)
    axes[1].plot(t_ctrl, force[:, 2] - hover_n, label=f"fz − mg", color="#d62728", linewidth=0.7)
    axes[1].axhline(0, color="gray", linewidth=0.4)
    axes[1].set_ylabel("Force [N]")
    axes[1].legend(loc="upper right", fontsize=8, ncol=3)
    axes[1].grid(alpha=0.3)
    axes[1].set_title("Commanded force components (hover thrust subtracted from fz)")

    # Wind vector components at the drone — shows the turbulent variation
    # the controller has to react to.
    wind = run["wind_at_drone"]
    t_wind = t[1:1 + len(wind)]
    axes[2].plot(t_wind, wind[:, 0], label="u (streamwise)", color="#1f6feb", linewidth=0.7)
    axes[2].plot(t_wind, wind[:, 1], label="v (lateral)", color="#2ca02c", linewidth=0.7)
    axes[2].plot(t_wind, wind[:, 2], label="w (vertical)", color="#d62728", linewidth=0.7)
    axes[2].plot(t_wind, wind_mag, label="|wind|", color="black", linewidth=0.8, linestyle="--")
    axes[2].set_ylabel("Wind [m/s]")
    axes[2].legend(loc="upper right", fontsize=8, ncol=4)
    axes[2].grid(alpha=0.3)

    axes[3].plot(t, run["battery"], color="#2ca02c", linewidth=1.0)
    axes[3].set_ylabel("Battery [Wh]")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(alpha=0.3)
    drained = run["battery"][0] - run["battery"][-1]
    rate = drained / t[-1] * 3600 if t[-1] > 0 else 0.0
    axes[3].set_title(f"Battery drain: {drained:.3f} Wh over {t[-1]:.0f} s  ({rate:.1f} W avg)")

    fig.suptitle(f"Station-keep run — mean|wind| = {run['meta']['mean_wind_ms']:.2f} m/s",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PLOTS / "time_series.png", dpi=150)
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    run = load_run()
    print(f"loaded run {RUN.name}: steps={run['meta']['steps_run']}, "
          f"score={run['meta']['score']:.3f} m, "
          f"terminated={run['meta']['terminated_reason']}")
    plot_trajectory(run)
    plot_time_series(run)
    print(f"wrote plots to {PLOTS}")


if __name__ == "__main__":
    main()
