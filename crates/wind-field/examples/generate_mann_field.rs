//! Generate an isotropic Mann turbulence field with the config's Mann
//! parameters and write it under `output/wind_fields/mann_isotropic/`.

use std::path::PathBuf;

use anyhow::Result;
use wind_field::io::save_grid;
use wind_field::mann::{MannConfig, generate_isotropic_field};

fn main() -> Result<()> {
    let cfg = MannConfig {
        alpha_epsilon_23: 0.585,
        length_scale_m: 16.8,
        gamma: 3.9,
    };

    let origin = [0.0; 3];
    let spacing = [10.0, 10.0, 10.0];
    let shape = (128, 128, 32);
    let mean_u_ms = 6.4;
    let seed: u64 = 20260817;

    let grid = generate_isotropic_field(&cfg, origin, spacing, shape, mean_u_ms, seed);

    let target_dir = PathBuf::from("output")
        .join("wind_fields")
        .join("mann_isotropic");
    save_grid(
        &grid,
        &target_dir,
        format!(
            "mann::generate_isotropic_field  A={}, L={}, mean_u={} m/s, seed={}",
            cfg.alpha_epsilon_23, cfg.length_scale_m, mean_u_ms, seed
        ),
    )?;

    let (mu, sigma_u) = mean_std(&grid.u);
    let (mv, sigma_v) = mean_std(&grid.v);
    let (mw, sigma_w) = mean_std(&grid.w);
    println!(
        "u: mean={:.3}, sigma={:.3}  |  v: mean={:.3}, sigma={:.3}  |  w: mean={:.3}, sigma={:.3}",
        mu, sigma_u, mv, sigma_v, mw, sigma_w
    );
    println!("wrote {}", target_dir.display());
    Ok(())
}

fn mean_std(arr: &ndarray::Array3<f64>) -> (f64, f64) {
    let mean = arr.mean().unwrap();
    let var = arr.mapv(|v| (v - mean).powi(2)).mean().unwrap();
    (mean, var.sqrt())
}
