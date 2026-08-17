//! Generate an isotropic Mann turbulence field with the config's Mann
//! parameters and write it under `output/wind_fields/mann_isotropic/`.

use std::path::PathBuf;

use anyhow::Result;
use wind_field::io::save_grid;
use wind_field::mann::{MannConfig, generate_anisotropic_field, generate_isotropic_field};

fn main() -> Result<()> {
    let cfg = MannConfig {
        alpha_epsilon_23: 0.163,
        length_scale_m: 16.8,
        gamma: 3.9,
    };

    let origin = [0.0; 3];
    let spacing = [10.0, 10.0, 10.0];
    let shape = (128, 128, 32);
    let mean_u_ms = 6.4;
    let seed: u64 = 20260817;

    // Isotropic (reference; kept for comparison with the anisotropic run).
    let iso_cfg = MannConfig {
        alpha_epsilon_23: 0.585,
        ..cfg.clone()
    };
    let iso_grid = generate_isotropic_field(&iso_cfg, origin, spacing, shape, mean_u_ms, seed);
    save_grid(
        &iso_grid,
        &PathBuf::from("output").join("wind_fields").join("mann_isotropic"),
        format!(
            "mann::generate_isotropic_field  A={}, L={}, mean_u={} m/s, seed={}",
            iso_cfg.alpha_epsilon_23, iso_cfg.length_scale_m, mean_u_ms, seed
        ),
    )?;
    print_stats("iso ", &iso_grid);

    // Anisotropic Mann field (RDT).
    let grid = generate_anisotropic_field(&cfg, origin, spacing, shape, mean_u_ms, seed);
    let target_dir = PathBuf::from("output")
        .join("wind_fields")
        .join("mann_anisotropic");
    save_grid(
        &grid,
        &target_dir,
        format!(
            "mann::generate_anisotropic_field  A={}, L={}, Gamma={}, mean_u={} m/s, seed={}",
            cfg.alpha_epsilon_23, cfg.length_scale_m, cfg.gamma, mean_u_ms, seed
        ),
    )?;
    print_stats("anis", &grid);

    println!("wrote {}", target_dir.display());
    Ok(())
}

fn print_stats(label: &str, grid: &wind_field::grid::WindGrid) {
    let (mu, sigma_u) = mean_std(&grid.u);
    let (_, sigma_v) = mean_std(&grid.v);
    let (_, sigma_w) = mean_std(&grid.w);
    let ratio_v = sigma_v / sigma_u;
    let ratio_w = sigma_w / sigma_u;
    println!(
        "{label}  mean_u={mu:.3}  sigma_u={sigma_u:.3}  sigma_v={sigma_v:.3}  sigma_w={sigma_w:.3}   ratios v/u={ratio_v:.2}, w/u={ratio_w:.2}"
    );
}

fn mean_std(arr: &ndarray::Array3<f64>) -> (f64, f64) {
    let mean = arr.mean().unwrap();
    let var = arr.mapv(|v| (v - mean).powi(2)).mean().unwrap();
    (mean, var.sqrt())
}
