"""Validate the isotropic Mann turbulence field written by the Rust example
`generate_mann_field`. Produces heatmaps, 1D cuts, spectra, and a divergence
sanity check."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[2] / "output" / "wind_fields" / "mann_isotropic"
PLOTS = Path(__file__).resolve().parents[2] / "output" / "plots" / "mann_isotropic"


def load():
    meta = json.loads((ROOT / "metadata.json").read_text())
    return {
        "u": np.load(ROOT / "u.npy"),
        "v": np.load(ROOT / "v.npy"),
        "w": np.load(ROOT / "w.npy"),
        "origin": np.array(meta["origin"]),
        "spacing": np.array(meta["spacing"]),
        "shape": tuple(meta["shape"]),
        "generator": meta["generator"],
    }


def summary_stats(grid) -> None:
    for name, arr in [("u", grid["u"]), ("v", grid["v"]), ("w", grid["w"])]:
        print(f"  {name}: mean={arr.mean():+.3f}  std={arr.std():.3f}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}")


def plot_slices(grid) -> None:
    """Horizontal slice (x-y plane) at mid-height for u, v, w."""
    k = grid["shape"][2] // 2
    x = grid["origin"][0] + np.arange(grid["shape"][0]) * grid["spacing"][0]
    y = grid["origin"][1] + np.arange(grid["shape"][1]) * grid["spacing"][1]
    extent = (x[0], x[-1], y[0], y[-1])
    u_prime = grid["u"] - grid["u"].mean()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, arr, title, cmap, sym in [
        (axes[0], u_prime[:, :, k], "u' (streamwise fluctuation)", "RdBu_r", True),
        (axes[1], grid["v"][:, :, k], "v (crosswise)", "RdBu_r", True),
        (axes[2], grid["w"][:, :, k], "w (vertical)", "RdBu_r", True),
    ]:
        v_max = max(abs(arr.min()), abs(arr.max())) if sym else None
        vmin, vmax = (-v_max, v_max) if sym else (None, None)
        im = ax.imshow(arr.T, origin="lower", extent=extent, cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="m/s", fraction=0.046, pad=0.04)

    fig.suptitle(f"Isotropic Mann field — horizontal slice at z = "
                 f"{grid['origin'][2] + k * grid['spacing'][2]:.0f} m",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOTS / "slices.png", dpi=150)
    plt.close(fig)


def plot_1d_cut(grid) -> None:
    """1D cut of u through the grid center along x."""
    j = grid["shape"][1] // 2
    k = grid["shape"][2] // 2
    x = grid["origin"][0] + np.arange(grid["shape"][0]) * grid["spacing"][0]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, grid["u"][:, j, k], linewidth=0.9, color="#1f6feb", label="u(x)")
    ax.axhline(grid["u"].mean(), color="#d62728", linestyle="--", linewidth=0.8,
               label=f"mean = {grid['u'].mean():.2f} m/s")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("u [m/s]")
    ax.set_title(f"1D cut through the field (y={j * grid['spacing'][1]:.0f} m, "
                 f"z={k * grid['spacing'][2]:.0f} m)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "cut_1d.png", dpi=150)
    plt.close(fig)


def plot_spectrum(grid) -> None:
    """Streamwise PSD averaged over y-z cross-section; overlay -5/3 reference."""
    dx = grid["spacing"][0]
    u_prime = grid["u"] - grid["u"].mean()
    nx = grid["shape"][0]

    # Average periodograms over all (j, k) lines along x.
    psd_sum = np.zeros(nx // 2 + 1)
    freqs = np.fft.rfftfreq(nx, d=dx)
    for j in range(grid["shape"][1]):
        for k in range(grid["shape"][2]):
            line = u_prime[:, j, k]
            spectrum = np.abs(np.fft.rfft(line)) ** 2 / nx / (1.0 / dx)
            psd_sum += spectrum
    psd = psd_sum / (grid["shape"][1] * grid["shape"][2])

    k_wave = 2 * np.pi * freqs[1:]  # rad/m, skip zero
    e_k = psd[1:]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.loglog(k_wave, e_k, color="#1f6feb", linewidth=1.2, label="Mann field (1D streamwise)")

    ref_k = k_wave[(k_wave > 0.05) & (k_wave < 0.2)]
    if ref_k.size:
        ref_amp = e_k[np.argmin(np.abs(k_wave - ref_k[0]))]
        ref_line = ref_amp * (ref_k / ref_k[0]) ** (-5.0 / 3.0)
        ax.loglog(ref_k, ref_line, color="#d62728", linestyle="--", linewidth=1.4,
                  label=r"$k^{-5/3}$ reference (Kolmogorov)")

    L = 16.8
    ax.axvline(1.0 / L, color="#888", linewidth=0.6, linestyle=":", label=f"1/L (L={L} m)")
    ax.axvline(np.pi / dx, color="#888", linewidth=0.6, linestyle="-.", label="Nyquist")

    ax.set_xlabel(r"Wavenumber $k$ [rad/m]")
    ax.set_ylabel(r"1D streamwise PSD  $F_{uu}(k)$  [(m/s)$^2$·m]")
    ax.set_title("Streamwise 1D energy spectrum")
    ax.legend()
    ax.grid(which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "psd.png", dpi=150)
    plt.close(fig)


def divergence_check(grid) -> float:
    """RMS of numerical divergence relative to RMS of individual gradients.
    For an isotropic vK field the projection guarantees div=0 in expectation."""
    dx, dy, dz = grid["spacing"]
    u = grid["u"]
    v = grid["v"]
    w = grid["w"]
    dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
    dvdy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dy)
    dwdz = (np.roll(w, -1, axis=2) - np.roll(w, 1, axis=2)) / (2 * dz)
    div = dudx + dvdy + dwdz
    scale = np.sqrt((dudx ** 2 + dvdy ** 2 + dwdz ** 2).mean())
    return float(np.sqrt((div ** 2).mean()) / scale)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    grid = load()
    print(f"generator: {grid['generator']}")
    print(f"shape={grid['shape']}, spacing={grid['spacing']}, origin={grid['origin']}")
    summary_stats(grid)
    plot_slices(grid)
    plot_1d_cut(grid)
    plot_spectrum(grid)
    ratio = divergence_check(grid)
    print(f"divergence RMS / gradient RMS = {ratio:.4f}   (small = incompressible)")
    print(f"wrote plots to {PLOTS}")


if __name__ == "__main__":
    main()
