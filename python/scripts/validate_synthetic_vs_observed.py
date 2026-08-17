"""Phase 1 validation: compare the synthetic Mann turbulence PSD against the
observed REA Point 2024 PSD, spanning ~7 decades of scale via Taylor's
frozen-turbulence hypothesis (temporal observed freq -> spatial wavenumber).

The two spectra do NOT overlap in wavenumber — observed hourly data reaches
down to periods of ~2 hours (large-scale synoptic weather), Mann covers the
turbulence scales (~m to km). Together they show whether the synthetic field
plugs into the low end of the observed spectrum with a consistent slope.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from arctic_sim.ingest.env_canada import load_rea_point
from arctic_sim.process.wind_stats import (
    clean,
    psd_welch,
    subset_year,
    summary,
    weibull_fit,
)

YEAR = 2024
MANN_DIR = Path(__file__).resolve().parents[2] / "output" / "wind_fields" / "mann_anisotropic"
PLOTS = Path(__file__).resolve().parents[2] / "output" / "plots" / "validation"


def load_synth_field():
    meta = json.loads((MANN_DIR / "metadata.json").read_text())
    return {
        "u": np.load(MANN_DIR / "u.npy"),
        "v": np.load(MANN_DIR / "v.npy"),
        "w": np.load(MANN_DIR / "w.npy"),
        "spacing": np.array(meta["spacing"]),
        "shape": tuple(meta["shape"]),
        "generator": meta["generator"],
    }


def synth_streamwise_psd(u: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Average the 1D streamwise PSD over all (y, z) lines through the field."""
    nx = u.shape[0]
    u_prime = u - u.mean()
    freqs = np.fft.rfftfreq(nx, d=dx)
    psd_sum = np.zeros(nx // 2 + 1)
    for j in range(u.shape[1]):
        for k in range(u.shape[2]):
            spec = np.abs(np.fft.rfft(u_prime[:, j, k])) ** 2 / nx / (1.0 / dx)
            psd_sum += spec
    psd = psd_sum / (u.shape[1] * u.shape[2])
    k = 2 * np.pi * freqs
    return k, psd


def obs_psd_to_wavenumber(speeds_ms: np.ndarray, U_mean: float):
    """Compute observed temporal PSD and convert to spatial wavenumber via
    Taylor's hypothesis: k = 2*pi*f / U, F(k) = F(f) * U / (2*pi)."""
    f, F_f = psd_welch(speeds_ms)
    k = 2 * np.pi * f / U_mean
    F_k = F_f * U_mean / (2 * np.pi)
    return k, F_k


def plot_combined_psd(k_obs, F_obs, k_synth, F_synth, U_mean, L, dx):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    mask_o = k_obs > 0
    mask_s = k_synth > 0
    ax.loglog(k_obs[mask_o], F_obs[mask_o], color="#d62728", linewidth=1.3,
              label=f"Observed REA Point 2024 (Taylor: U={U_mean:.2f} m/s)")
    ax.loglog(k_synth[mask_s], F_synth[mask_s], color="#1f6feb", linewidth=1.3,
              label="Synthetic Mann anisotropic (streamwise 1D)")

    # -5/3 reference: pick a decade in the middle where both spectra should
    # (if the physics were extrapolatable) fall on the same slope.
    k_ref = np.geomspace(1e-4, 1e-1, 20)
    # Anchor near k = 3e-2 rad/m (inertial range of the synthetic).
    idx_anchor = np.argmin(np.abs(k_synth - 3e-2))
    amp_anchor = F_synth[idx_anchor]
    k_anchor = k_synth[idx_anchor]
    ref = amp_anchor * (k_ref / k_anchor) ** (-5.0 / 3.0)
    ax.loglog(k_ref, ref, color="#666", linewidth=1.2, linestyle="--",
              label=r"$k^{-5/3}$ (Kolmogorov)")

    # Vertical annotations at meaningful physical scales.
    scale_lines = [
        (1e-7, "1 year"),
        (2 * np.pi / (U_mean * 30 * 24 * 3600), "1 month"),
        (2 * np.pi / (U_mean * 24 * 3600), "1 day"),
        (2 * np.pi / (U_mean * 3600), "1 hour"),
        (2 * np.pi / (U_mean * 60), "1 min"),
        (1.0 / L, f"1/L (L={L} m)"),
        (np.pi / dx, "grid Nyquist (20 m)"),
    ]
    ymin, ymax = 1e-2, 1e10
    ax.set_ylim(ymin, ymax)
    for k_val, label in scale_lines:
        ax.axvline(k_val, color="#aaa", linewidth=0.5, linestyle=":")
        ax.text(k_val, ymax * 0.3, label, rotation=90, va="top", ha="right",
                fontsize=8, color="#555")

    ax.set_xlabel(r"Wavenumber $k$ [rad/m]  (Taylor's hypothesis for observed)")
    ax.set_ylabel(r"1D PSD  $F_{uu}(k)$  [(m/s)$^2 \cdot$m]")
    ax.set_title("Wind PSD — observed synoptic scales + synthetic turbulence scales")
    ax.set_xlim(1e-8, 1)
    ax.legend(loc="lower left")
    ax.grid(which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "psd_combined.png", dpi=150)
    plt.close(fig)


def plot_distributions(obs_speeds: np.ndarray, synth_u: np.ndarray, k_wb: float, c_wb: float):
    from scipy import stats as scistats

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.linspace(0, 20, 50)

    ax.hist(obs_speeds, bins=bins, density=True, alpha=0.55, color="#d62728",
            edgecolor="white", linewidth=0.3, label="Observed 2024 (all hours)")
    ax.hist(synth_u.flatten(), bins=bins, density=True, alpha=0.55, color="#1f6feb",
            edgecolor="white", linewidth=0.3,
            label=f"Synthetic Mann (single snapshot, U_mean={synth_u.mean():.2f} m/s)")

    x = np.linspace(0.01, 20, 500)
    ax.plot(x, scistats.weibull_min.pdf(x, k_wb, loc=0.0, scale=c_wb),
            color="#8b0000", linewidth=1.5, label=f"Weibull fit  k={k_wb:.2f}, c={c_wb:.2f}")

    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Probability density")
    ax.set_title("Wind speed distribution — observed (all hours, all weather) vs synthetic (fixed IEC condition)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "distributions.png", dpi=150)
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)

    # --- Observed --------------------------------------------------------
    obs_df = clean(subset_year(load_rea_point(), YEAR))
    obs_speeds = obs_df["wind_speed_ms"].to_numpy()
    obs_summary = summary(obs_df)
    k_wb, c_wb = weibull_fit(obs_speeds)
    U_mean = obs_summary.mean_ms

    # --- Synthetic -------------------------------------------------------
    synth = load_synth_field()
    dx = float(synth["spacing"][0])
    L = 16.8

    # PSDs
    k_obs, F_obs = obs_psd_to_wavenumber(obs_speeds, U_mean)
    k_synth, F_synth = synth_streamwise_psd(synth["u"], dx)

    # Print comparison
    print(f"--- observed REA Point {YEAR} ---")
    print(f"  mean U      : {obs_summary.mean_ms:.3f} m/s")
    print(f"  std (all)   : {obs_summary.std_ms:.3f} m/s   (synoptic + all variability)")
    print(f"  Weibull     : k={k_wb:.3f}, c={c_wb:.3f}")
    print()
    print(f"--- synthetic Mann anisotropic ---")
    u = synth["u"]
    print(f"  mean U      : {u.mean():.3f} m/s   (imposed)")
    print(f"  std (turb)  : {u.std():.3f} m/s   (turbulence-only, matches IEC NTM at V_hub)")
    print(f"  generator   : {synth['generator']}")
    print()
    print(f"note: observed std includes weeks-to-year weather variability;")
    print(f"      synthetic std is turbulence-only at a fixed IEC atmospheric state.")
    print(f"      These are complementary, not directly comparable.")

    plot_combined_psd(k_obs, F_obs, k_synth, F_synth, U_mean, L, dx)
    plot_distributions(obs_speeds, u, k_wb, c_wb)
    print(f"wrote plots to {PLOTS}")


if __name__ == "__main__":
    main()
