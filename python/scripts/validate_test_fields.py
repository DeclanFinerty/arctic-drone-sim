"""Load the three analytical wind fields written by the Rust example
`generate_test_fields` and produce validation plots."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "output" / "wind_fields"
PLOTS = Path(__file__).resolve().parents[2] / "output" / "plots" / "test_fields"


@dataclass
class Grid:
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray
    shape: tuple[int, int, int]
    generator: str


def load(name: str) -> Grid:
    d = ROOT / name
    meta = json.loads((d / "metadata.json").read_text())
    return Grid(
        u=np.load(d / "u.npy"),
        v=np.load(d / "v.npy"),
        w=np.load(d / "w.npy"),
        origin=np.array(meta["origin"]),
        spacing=np.array(meta["spacing"]),
        shape=tuple(meta["shape"]),
        generator=meta["generator"],
    )


def axis(grid: Grid, dim: int) -> np.ndarray:
    return grid.origin[dim] + np.arange(grid.shape[dim]) * grid.spacing[dim]


def plot_uniform(grid: Grid) -> None:
    x = axis(grid, 0)
    y = axis(grid, 1)
    k = grid.shape[2] // 2
    step = 8
    xs, ys = np.meshgrid(x[::step], y[::step], indexing="ij")
    u_slice = grid.u[::step, ::step, k]
    v_slice = grid.v[::step, ::step, k]
    speeds = np.sqrt(grid.u[:, :, k] ** 2 + grid.v[:, :, k] ** 2)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(
        speeds.T,
        origin="lower",
        extent=(x[0], x[-1], y[0], y[-1]),
        cmap="viridis",
        aspect="equal",
    )
    ax.quiver(xs, ys, u_slice, v_slice, color="white", scale=100, width=0.003)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Uniform field — horizontal slice at z = {grid.origin[2] + k * grid.spacing[2]:.0f} m")
    fig.colorbar(im, ax=ax, label="wind speed [m/s]")
    fig.tight_layout()
    fig.savefig(PLOTS / "uniform.png", dpi=150)
    plt.close(fig)


def plot_shear(grid: Grid) -> None:
    z = axis(grid, 2)
    profile = grid.u[0, 0, :]
    z_ref = 10.0
    u_ref = 5.715
    analytical = u_ref * (np.maximum(z, 1e-6) / z_ref) ** (1.0 / 7.0)

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.plot(profile, z, marker="o", markersize=3, linewidth=1.0, label="grid sample")
    ax.plot(analytical, z, linestyle="--", linewidth=1.4, color="#d62728",
            label=r"analytical $U(z) = U_{ref}(z/z_{ref})^{1/7}$")
    ax.axhline(z_ref, color="#888", linewidth=0.6, linestyle=":",
               label=f"reference height {z_ref} m")
    ax.axvline(u_ref, color="#888", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Wind speed u [m/s]")
    ax.set_ylabel("Height z [m]")
    ax.set_title("Power-law shear — vertical profile")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "shear.png", dpi=150)
    plt.close(fig)


def plot_single_mode(grid: Grid) -> None:
    x = axis(grid, 0)
    y = axis(grid, 1)
    k = grid.shape[2] // 2
    slice_u = grid.u[:, :, k]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"width_ratios": [1.4, 1]})

    im = axes[0].imshow(
        slice_u.T,
        origin="lower",
        extent=(x[0], x[-1], y[0], y[-1]),
        cmap="RdBu_r",
        vmin=-2.2, vmax=2.2,
        aspect="equal",
    )
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title("Single Fourier mode — u at mid-height")
    fig.colorbar(im, ax=axes[0], label="u [m/s]")

    j = grid.shape[1] // 2
    axes[1].plot(x, grid.u[:, j, k], linewidth=1.0, label="grid sample")
    axes[1].plot(x, 2.0 * np.sin(2 * np.pi * x / 200.0),
                 linestyle="--", linewidth=1.4, color="#d62728",
                 label=r"$2\sin(2\pi x / 200)$")
    axes[1].axhline(0, color="#888", linewidth=0.5)
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("u [m/s]")
    axes[1].set_title(f"1D cut at y = {y[j]:.0f}, z = {grid.origin[2] + k * grid.spacing[2]:.0f} m")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS / "single_mode.png", dpi=150)
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    for name, plotter in [
        ("uniform", plot_uniform),
        ("shear", plot_shear),
        ("single_mode", plot_single_mode),
    ]:
        grid = load(name)
        print(f"{name:12s} shape={grid.shape} generator={grid.generator}")
        plotter(grid)
    print(f"wrote plots to {PLOTS}")


if __name__ == "__main__":
    main()
