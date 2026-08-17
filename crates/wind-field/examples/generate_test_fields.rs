//! Sample the three analytical wind fields onto grids and write them to
//! `output/wind_fields/{uniform,shear,single_mode}/`. Consumed by
//! `python/scripts/validate_test_fields.py` for visualization.

use std::path::PathBuf;

use anyhow::Result;
use wind_field::analytical::{PowerLawShear, SingleFourierMode, Uniform};
use wind_field::grid::sample_to_grid;
use wind_field::io::save_grid;

fn main() -> Result<()> {
    let root = PathBuf::from("output").join("wind_fields");

    // --- Test 1: uniform ----------------------------------------------------
    let uniform = Uniform {
        velocity: [8.0, 0.0, 0.0],
        bounds: ([0.0; 3], [1000.0, 1000.0, 100.0]),
    };
    let uniform_grid = sample_to_grid(
        &uniform,
        [0.0; 3],
        [10.0, 10.0, 5.0],
        (101, 101, 21),
    );
    save_grid(
        &uniform_grid,
        &root.join("uniform"),
        "wind-field::analytical::Uniform  velocity=[8,0,0] m/s",
    )?;

    // --- Test 2: power-law shear -------------------------------------------
    let shear = PowerLawShear {
        u_ref_ms: 5.715,
        z_ref_m: 10.0,
        alpha: 1.0 / 7.0,
        bounds: ([0.0, 0.0, 0.5], [1000.0, 1000.0, 200.5]),
    };
    let shear_grid = sample_to_grid(
        &shear,
        [0.0, 0.0, 0.5],
        [10.0, 10.0, 2.0],
        (101, 101, 101),
    );
    save_grid(
        &shear_grid,
        &root.join("shear"),
        "wind-field::analytical::PowerLawShear  u_ref=5.715 m/s at 10m, alpha=1/7",
    )?;

    // --- Test 3: single Fourier mode ---------------------------------------
    // Wavelength 200 m in x; 5 m spacing gives 40 samples per wavelength.
    let mode = SingleFourierMode {
        amplitude: [2.0, 0.0, 0.0],
        wavenumber: [1.0 / 200.0, 0.0, 0.0],
        bounds: ([0.0; 3], [1000.0, 500.0, 100.0]),
    };
    let mode_grid = sample_to_grid(
        &mode,
        [0.0; 3],
        [5.0, 5.0, 5.0],
        (201, 101, 21),
    );
    save_grid(
        &mode_grid,
        &root.join("single_mode"),
        "wind-field::analytical::SingleFourierMode  amplitude=2 m/s, wavelength=200 m along x",
    )?;

    println!("wrote 3 test grids under {}", root.display());
    Ok(())
}
