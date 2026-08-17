"""Plot a formation-flight run: leader + follower trajectories over a wind
slice, plus per-drone position error and inter-drone distance time series.

Usage:
    uv run python scripts/plot_formation.py                 # default: output/runs/formation
    uv run python scripts/plot_formation.py <run_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "output" / "runs" / "formation"
WIND_DIR = REPO / "output" / "wind_fields" / "mann_anisotropic"
PLOTS = REPO / "output" / "plots" / f"formation_{RUN.name}"

LEADER = "leader"
COLORS = {
    "leader": "#d62728",
    "N": "#1f6feb",
    "S": "#2ca02c",
    "E": "#9467bd",
    "W": "#ff7f0e",
}


def load_drone(name: str):
    d = RUN / name
    return {
        "name": name,
        "times": np.load(d / "times.npy"),
        "positions": np.load(d / "positions.npy"),
        "velocities": np.load(d / "velocities.npy"),
        "battery": np.load(d / "battery_wh.npy"),
        "commanded_force": np.load(d / "commanded_force_n.npy"),
        "wind_at_drone": np.load(d / "wind_at_drone_ms.npy"),
        "meta": json.loads((d / "metadata.json").read_text()),
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


def plot_trajectory(drones):
    leader = drones[LEADER]
    z = float(leader["positions"][0, 2])
    ws = load_wind_slice(z)
    speed = np.sqrt(ws["u_slice"] ** 2 + ws["v_slice"] ** 2)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(speed.T, origin="lower",
                   extent=(ws["x"][0], ws["x"][-1], ws["y"][0], ws["y"][-1]),
                   cmap="Blues", alpha=0.8, aspect="equal")
    fig.colorbar(im, ax=ax, label="Horizontal wind speed [m/s]")

    step = max(1, len(ws["x"]) // 20)
    xs, ys = np.meshgrid(ws["x"][::step], ws["y"][::step], indexing="ij")
    ax.quiver(xs, ys, ws["u_slice"][::step, ::step], ws["v_slice"][::step, ::step],
              color="#333", alpha=0.35, scale=250, width=0.0018)

    for name, d in drones.items():
        px, py = d["positions"][:, 0], d["positions"][:, 1]
        ax.plot(px, py, color=COLORS.get(name, "black"),
                linewidth=1.1 if name != LEADER else 1.6,
                label=name, alpha=0.85)
        ax.scatter([px[0]], [py[0]], color=COLORS.get(name, "black"),
                   s=45, edgecolor="black", zorder=5)

    # Frame around all trajectories with margin.
    all_x = np.concatenate([d["positions"][:, 0] for d in drones.values()])
    all_y = np.concatenate([d["positions"][:, 1] for d in drones.values()])
    cx = 0.5 * (all_x.min() + all_x.max())
    cy = 0.5 * (all_y.min() + all_y.max())
    span = max(np.ptp(all_x), np.ptp(all_y)) * 0.65
    ax.set_xlim(cx - span, cx + span); ax.set_ylim(cy - span, cy + span)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Formation over wind slice at z = {ws['z_used_m']:.0f} m")
    ax.legend(loc="upper right", ncol=1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "trajectory.png", dpi=150)
    plt.close(fig)


def plot_time_series(drones):
    leader = drones[LEADER]
    t = leader["times"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Leader course-tracking error (from FollowCourse target).
    # We need the target trajectory — reconstruct from mission metadata if
    # available; otherwise skip. Since the sweep saves just target_m at t=0,
    # we don't have the course; approximate leader tracking as speed vs time.
    speed = np.linalg.norm(leader["velocities"], axis=1)
    axes[0].plot(t, speed, color=COLORS[LEADER], label="leader |velocity|")
    axes[0].set_ylabel("Leader speed [m/s]")
    axes[0].grid(alpha=0.3)
    axes[0].set_title(f"Leader course-tracking (RMS error = {leader['meta']['score']:.2f} m)")
    axes[0].legend(loc="upper right", fontsize=8)

    # Inter-drone distance: each follower to leader (should be ~offset magnitude).
    for name, d in drones.items():
        if name == LEADER:
            continue
        sep = np.linalg.norm(d["positions"] - leader["positions"], axis=1)
        offset_target = np.linalg.norm(d["positions"][0] - leader["positions"][0])
        axes[1].plot(t, sep, color=COLORS.get(name, "black"),
                     label=f"{name} (target {offset_target:.0f} m)", linewidth=0.9)
    axes[1].set_ylabel("Distance to leader [m]")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right", ncol=4, fontsize=8)
    axes[1].set_title("Follower separation from leader")

    # Battery drain, all drones.
    for name, d in drones.items():
        axes[2].plot(t, d["battery"], color=COLORS.get(name, "black"),
                     label=f"{name}", linewidth=0.9)
    axes[2].set_ylabel("Battery [Wh]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper right", ncol=5, fontsize=8)
    total_drain = sum(d["battery"][0] - d["battery"][-1] for d in drones.values())
    axes[2].set_title(f"Battery — fleet total drain {total_drain:.2f} Wh over {t[-1]:.0f} s")

    fig.tight_layout()
    fig.savefig(PLOTS / "time_series.png", dpi=150)
    plt.close(fig)


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    names = [d.name for d in RUN.iterdir() if d.is_dir()]
    if LEADER not in names:
        raise SystemExit(f"expected '{LEADER}' subdir under {RUN}, got {names}")
    # Sort with leader first, followers in a consistent order.
    ordered = [LEADER] + sorted(n for n in names if n != LEADER)
    drones = {n: load_drone(n) for n in ordered}
    print(f"loaded {len(drones)} drones from {RUN.name}")
    plot_trajectory(drones)
    plot_time_series(drones)
    print(f"wrote plots to {PLOTS}")


if __name__ == "__main__":
    main()
