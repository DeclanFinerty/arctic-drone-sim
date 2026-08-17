"""Derive Mann parameters for REA Point 2024 and print a TOML block."""
from __future__ import annotations

from arctic_sim.ingest.env_canada import load_rea_point
from arctic_sim.process.wind_stats import (
    clean,
    derive_mann_params,
    subset_year,
    summary,
)

YEAR = 2024
Z_MEASURED = 10.0
Z_HUB = 30.0
Z0 = 0.001
IEC_CLASS = "C"


def main() -> None:
    df = clean(subset_year(load_rea_point(), YEAR))
    stats_ = summary(df)
    p = derive_mann_params(
        stats_.mean_ms,
        z_measured_m=Z_MEASURED,
        z_hub_m=Z_HUB,
        z0_m=Z0,
        iec_class=IEC_CLASS,
    )
    print(f"--- REA Point {YEAR} Mann parameters (IEC 61400-1 Ed 4, class {p.iec_class}) ---")
    print(f"observed mean wind at {Z_MEASURED} m : {stats_.mean_ms:.3f} m/s")
    print(f"extrapolated V_hub at {p.z_hub_m} m : {p.v_hub_ms:.3f} m/s   (z0 = {p.z0_m} m)")
    print(f"IEC NTM sigma_u target             : {p.sigma_u_target_ms:.3f} m/s")
    print(f"target TI at hub                   : {p.ti_target * 100:.1f} %")
    print()
    print("copy into configs/default.toml under [wind_field.mann]:")
    print()
    print(f"alpha_epsilon_23 = {p.alpha_epsilon_23:.4f}   # m^(4/3) / s^2  (initial estimate)")
    print(f"length_scale_m   = {p.length_scale_m:.3f}")
    print(f"gamma            = {p.gamma}")
    print()
    print(f"# derived from REA Point {YEAR} climatology; z_hub={p.z_hub_m} m, IEC class {p.iec_class}")
    print(f"# note: {p.notes}")


if __name__ == "__main__":
    main()
