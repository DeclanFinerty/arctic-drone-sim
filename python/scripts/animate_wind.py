"""Animate a horizontal slice of the wind field advecting past under Taylor's
frozen-turbulence hypothesis. Optionally overlay a simulation run's drone
trajectory so you can watch the gusts push it around.

Usage:
    uv run python scripts/animate_wind.py                    # wind field only
    uv run python scripts/animate_wind.py --run station_keep # + drone overlay
    uv run python scripts/animate_wind.py --z 30 --seconds 60 --fps 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = Path(__file__).resolve().parents[2]
WIND_DIR = REPO / "output" / "wind_fields" / "mann_anisotropic"


def load_slice(z_target_m: float):
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
    return {
        "x": x, "y": y,
        "u0": u[:, :, k], "v0": v[:, :, k],
        "spacing": spacing, "origin": origin,
        "z_used_m": float(origin[2] + k * spacing[2]),
    }


FORMATION_COLORS = {
    "leader": "#d62728",
    "N": "#1f6feb", "S": "#2ca02c", "E": "#9467bd", "W": "#ff7f0e",
}


def load_run(run_name: str):
    """Return a list of drone traces. Single-drone runs return a 1-element list
    with name 'drone'; formation runs return one entry per subdirectory."""
    run_dir = REPO / "output" / "runs" / run_name
    subdirs = sorted(d for d in run_dir.iterdir() if d.is_dir())
    if subdirs:
        # Put 'leader' first if present.
        subdirs.sort(key=lambda p: (p.name != "leader", p.name))
        drones = []
        for d in subdirs:
            drones.append({
                "name": d.name,
                "times": np.load(d / "times.npy"),
                "positions": np.load(d / "positions.npy"),
                "meta": json.loads((d / "metadata.json").read_text()),
            })
        return drones
    return [{
        "name": "drone",
        "times": np.load(run_dir / "times.npy"),
        "positions": np.load(run_dir / "positions.npy"),
        "meta": json.loads((run_dir / "metadata.json").read_text()),
    }]


def load_advection_ms():
    """Read Taylor advection from the sim config; fall back to a default."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    cfg = tomllib.loads((REPO / "configs" / "default.toml").read_text())
    return np.array(cfg.get("wind_field", {}).get("taylor_advection_ms", [6.4, 0.0, 0.0]))


def frame_slice(ws, t: float, adv_ms: np.ndarray):
    """Roll the base slice to represent Taylor advection at time t.

    A stationary observer sees the field that WAS at (x - vx*t, y - vy*t) at t=0.
    Since the Mann field is periodic (FFT-generated), np.roll gives the exact
    displacement modulo the domain size.
    """
    shift_i = int(round(-adv_ms[0] * t / ws["spacing"][0]))
    shift_j = int(round(-adv_ms[1] * t / ws["spacing"][1]))
    u = np.roll(ws["u0"], (shift_i, shift_j), axis=(0, 1))
    v = np.roll(ws["v0"], (shift_i, shift_j), axis=(0, 1))
    return u, v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None,
                        help="run name under output/runs/ to overlay (optional)")
    parser.add_argument("--z", type=float, default=30.0, help="slice altitude [m]")
    parser.add_argument("--seconds", type=float, default=60.0, help="sim time to span")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--speedup", type=float, default=4.0,
                        help="animation plays this many sim-seconds per real second")
    parser.add_argument("--zoom", type=float, default=None,
                        help="half-width of view around drone/target [m]. Default: full domain")
    parser.add_argument("--out", type=str, default=None, help="output .gif path")
    args = parser.parse_args()

    ws = load_slice(args.z)
    adv_ms = load_advection_ms()
    drones = load_run(args.run) if args.run else None

    n_frames = int(args.fps * args.seconds / args.speedup)
    t_of_frame = np.linspace(0.0, args.seconds, n_frames)

    fig, ax = plt.subplots(figsize=(9, 8))
    speed0 = np.sqrt(ws["u0"] ** 2 + ws["v0"] ** 2)
    vmax = float(np.percentile(speed0, 99) * 1.05)
    vmin = float(np.percentile(speed0, 1) * 0.95)

    extent = (ws["x"][0], ws["x"][-1], ws["y"][0], ws["y"][-1])
    im = ax.imshow(speed0.T, origin="lower", extent=extent, cmap="Blues",
                   vmin=vmin, vmax=vmax, aspect="equal")
    fig.colorbar(im, ax=ax, label="Horizontal wind speed [m/s]")

    step = max(1, len(ws["x"]) // 24)
    xs, ys = np.meshgrid(ws["x"][::step], ws["y"][::step], indexing="ij")
    quiv = ax.quiver(xs, ys, ws["u0"][::step, ::step], ws["v0"][::step, ::step],
                     color="#222", alpha=0.55, scale=200, width=0.002)

    drone_artists = []  # per-drone (dot, trail)
    if drones is not None:
        for d in drones:
            color = FORMATION_COLORS.get(d["name"], "#d62728")
            lw = 1.6 if d["name"] == "leader" else 1.1
            ms = 10 if d["name"] == "leader" else 8
            trail, = ax.plot([], [], "-", color=color, linewidth=lw,
                             alpha=0.75, zorder=6)
            dot, = ax.plot([], [], "o", color=color, markersize=ms,
                           markeredgecolor="black", zorder=7, label=d["name"])
            drone_artists.append((dot, trail, d))
        ax.legend(loc="upper right", fontsize=8)

    if args.zoom is not None:
        if drones is not None:
            all_x = np.concatenate([d["positions"][:, 0] for d in drones])
            all_y = np.concatenate([d["positions"][:, 1] for d in drones])
            cx = 0.5 * (all_x.min() + all_x.max())
            cy = 0.5 * (all_y.min() + all_y.max())
        else:
            cx = 0.5 * (ws["x"][0] + ws["x"][-1])
            cy = 0.5 * (ws["y"][0] + ws["y"][-1])
        ax.set_xlim(cx - args.zoom, cx + args.zoom)
        ax.set_ylim(cy - args.zoom, cy + args.zoom)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    title = ax.set_title("")

    def update(k):
        t = t_of_frame[k]
        u, v = frame_slice(ws, t, adv_ms)
        speed = np.sqrt(u ** 2 + v ** 2)
        im.set_data(speed.T)
        quiv.set_UVC(u[::step, ::step], v[::step, ::step])
        title.set_text(f"z = {ws['z_used_m']:.0f} m,   t = {t:5.2f} s   "
                       f"(mean advection {adv_ms[0]:.1f}, {adv_ms[1]:.1f} m/s)")
        artists = [im, quiv, title]
        for dot, trail, d in drone_artists:
            idx = np.searchsorted(d["times"], t)
            idx = min(idx, len(d["times"]) - 1)
            px = d["positions"][:idx + 1, 0]
            py = d["positions"][:idx + 1, 1]
            dot.set_data([px[-1]], [py[-1]])
            trail.set_data(px, py)
            artists += [dot, trail]
        return artists

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps,
                        blit=False)

    out_dir = REPO / "output" / "plots" / "animations"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        tag = f"_with_{args.run}" if args.run else ""
        out_path = out_dir / f"wind_z{int(args.z)}_{int(args.seconds)}s{tag}.gif"
    else:
        out_path = Path(args.out)

    print(f"rendering {n_frames} frames -> {out_path}")
    ani.save(str(out_path), writer=PillowWriter(fps=args.fps))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
