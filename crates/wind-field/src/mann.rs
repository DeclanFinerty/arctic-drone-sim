//! Isotropic von Karman turbulence field, step 7a of the Mann pipeline.
//!
//! Generates a divergence-free 3D vector field on a periodic Cartesian grid
//! from the isotropic von Karman spectral tensor. Full anisotropic RDT
//! (Mann's Gamma parameter) is a follow-up: this variant gives correct
//! `sigma_u` and spectral shape, but `sigma_v = sigma_w = sigma_u` instead of
//! the atmospheric ratios `~0.8` and `~0.5`.
//!
//! Method:
//!   1. Discretize k-space on the grid dual to the physical domain.
//!   2. At each k with |k| > 0, evaluate the von Karman energy `E(k)` and the
//!      isotropic projection `P_ij = delta_ij - k_i k_j / |k|^2`.
//!   3. Draw independent complex Gaussians eta_j(k), then set
//!         U_i(k) = sqrt(E(k) / (4 pi |k|^2)) * sqrt(dk1 dk2 dk3) * P_ij eta_j
//!      Because P is a rank-2 idempotent, sqrt(Phi_ij) = sqrt(E/(4 pi k^2)) * P_ij.
//!   4. 3D inverse FFT each component, take the real part.
//!   5. Add a spatially constant mean streamwise wind.

use std::f64::consts::PI;

use ndarray::Array3;
use ndrustfft::{FftHandler, ndifft};
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::StandardNormal;

use crate::grid::WindGrid;

#[derive(Debug, Clone)]
pub struct MannConfig {
    pub alpha_epsilon_23: f64,
    pub length_scale_m: f64,
    /// Anisotropy parameter (unused in the isotropic 7a variant).
    pub gamma: f64,
}

/// Von Karman isotropic energy spectrum `E(k)`.
pub fn von_karman_energy(k_mag: f64, alpha_eps_23: f64, length: f64) -> f64 {
    if k_mag == 0.0 {
        return 0.0;
    }
    let kl = k_mag * length;
    let kl2 = kl * kl;
    alpha_eps_23 * length.powf(5.0 / 3.0) * kl.powi(4) / (1.0 + kl2).powf(17.0 / 6.0)
}

/// Generate a divergence-free isotropic turbulence field on a periodic grid,
/// added to a spatially constant streamwise mean wind.
pub fn generate_isotropic_field(
    config: &MannConfig,
    origin: [f64; 3],
    spacing: [f64; 3],
    shape: (usize, usize, usize),
    mean_u_ms: f64,
    seed: u64,
) -> WindGrid {
    let (nx, ny, nz) = shape;
    let lx = nx as f64 * spacing[0];
    let ly = ny as f64 * spacing[1];
    let lz = nz as f64 * spacing[2];
    let dkx = 2.0 * PI / lx;
    let dky = 2.0 * PI / ly;
    let dkz = 2.0 * PI / lz;
    // sqrt(2) compensates for the factor-of-1/2 lost by taking Re(IFFT(c))
    // when c(k) are independent complex Gaussians (no Hermitian symmetry).
    let sqrt_dk3 = (2.0 * dkx * dky * dkz).sqrt();

    let mut uk = Array3::<Complex64>::zeros(shape);
    let mut vk = Array3::<Complex64>::zeros(shape);
    let mut wk = Array3::<Complex64>::zeros(shape);

    let mut rng = StdRng::seed_from_u64(seed);

    for i in 0..nx {
        let kx = fft_freq(i, nx) * dkx;
        for j in 0..ny {
            let ky = fft_freq(j, ny) * dky;
            for k in 0..nz {
                let kz = fft_freq(k, nz) * dkz;
                let k_mag2 = kx * kx + ky * ky + kz * kz;
                if k_mag2 < 1e-24 {
                    continue;
                }
                let k_mag = k_mag2.sqrt();
                let e_k = von_karman_energy(k_mag, config.alpha_epsilon_23, config.length_scale_m);
                let s = (e_k / (4.0 * PI * k_mag2)).sqrt() * sqrt_dk3;

                let nx_u = kx / k_mag;
                let ny_u = ky / k_mag;
                let nz_u = kz / k_mag;

                let eta_x = complex_gaussian(&mut rng);
                let eta_y = complex_gaussian(&mut rng);
                let eta_z = complex_gaussian(&mut rng);

                let dot = nx_u * eta_x + ny_u * eta_y + nz_u * eta_z;
                uk[[i, j, k]] = s * (eta_x - dot * nx_u);
                vk[[i, j, k]] = s * (eta_y - dot * ny_u);
                wk[[i, j, k]] = s * (eta_z - dot * nz_u);
            }
        }
    }

    let u_phys = ifft_3d(uk, shape).mapv(|z| z.re);
    let v_phys = ifft_3d(vk, shape).mapv(|z| z.re);
    let w_phys = ifft_3d(wk, shape).mapv(|z| z.re);

    let u_total = &u_phys + mean_u_ms;

    WindGrid {
        u: u_total,
        v: v_phys,
        w: w_phys,
        origin,
        spacing,
    }
}

/// Integer wavenumber index for FFT bin `i` of length `n`: 0..n/2 map to
/// positive frequencies, the rest to negative frequencies (numpy convention).
fn fft_freq(i: usize, n: usize) -> f64 {
    let n_i = n as isize;
    let i_i = i as isize;
    if 2 * i_i <= n_i {
        i as f64
    } else {
        (i_i - n_i) as f64
    }
}

/// Complex standard normal with `E[|eta|^2] = 1` (Re, Im each variance 1/2).
fn complex_gaussian<R: Rng>(rng: &mut R) -> Complex64 {
    let re: f64 = rng.sample(StandardNormal);
    let im: f64 = rng.sample(StandardNormal);
    Complex64::new(re, im) / std::f64::consts::SQRT_2
}

/// 3D unnormalized inverse FFT via sequential 1D IFFTs along each axis.
/// `ndrustfft` applies a 1/N factor per axis, so the total scaling is
/// 1/(nx*ny*nz). We compensate by multiplying the result at the end so the
/// output matches the "synthesis by discrete sum" convention.
fn ifft_3d(mut arr: Array3<Complex64>, shape: (usize, usize, usize)) -> Array3<Complex64> {
    let (nx, ny, nz) = shape;
    let mut handler_x = FftHandler::<f64>::new(nx);
    let mut handler_y = FftHandler::<f64>::new(ny);
    let mut handler_z = FftHandler::<f64>::new(nz);

    let mut tmp = Array3::<Complex64>::zeros(shape);
    ndifft(&arr, &mut tmp, &mut handler_x, 0);
    ndifft(&tmp, &mut arr, &mut handler_y, 1);
    ndifft(&arr, &mut tmp, &mut handler_z, 2);

    let scale = (nx * ny * nz) as f64;
    tmp.mapv_inplace(|c| c * scale);
    tmp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_spectrum_zero_at_origin_and_positive_elsewhere() {
        assert_eq!(von_karman_energy(0.0, 0.1, 20.0), 0.0);
        let e = von_karman_energy(0.05, 0.1, 20.0);
        assert!(e > 0.0);
    }

    #[test]
    fn isotropic_field_mean_matches_input() {
        let cfg = MannConfig {
            alpha_epsilon_23: 0.0742,
            length_scale_m: 16.8,
            gamma: 3.9,
        };
        let grid = generate_isotropic_field(
            &cfg,
            [0.0; 3],
            [10.0, 10.0, 10.0],
            (32, 32, 16),
            6.4,
            123,
        );
        let mean_u = grid.u.mean().unwrap();
        assert!(
            (mean_u - 6.4).abs() < 0.05,
            "mean u {} deviates from target 6.4",
            mean_u
        );
        let mean_v = grid.v.mean().unwrap();
        assert!(mean_v.abs() < 0.1, "mean v {} should be near zero", mean_v);
    }
}
