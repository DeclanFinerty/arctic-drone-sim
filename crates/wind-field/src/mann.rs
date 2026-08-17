//! Mann turbulence field: isotropic (7a) and anisotropic RDT (7b) variants.
//!
//! Isotropic method (`generate_isotropic_field`):
//!   1. Discretize k-space on the grid dual to the physical domain.
//!   2. At each k with |k| > 0, evaluate the von Karman energy `E(k)` and the
//!      isotropic projection `P_ij = delta_ij - k_i k_j / |k|^2`.
//!   3. Draw independent complex Gaussians eta_j(k), then set
//!         U_i(k) = sqrt(E(k) / (4 pi |k|^2)) * sqrt(2 dk1 dk2 dk3) * P_ij eta_j
//!      Because P is a rank-2 idempotent, sqrt(Phi_ij) = sqrt(E/(4 pi k^2)) * P_ij.
//!   4. 3D inverse FFT each component, take the real part.
//!   5. Add a spatially constant mean streamwise wind.
//!
//! Anisotropic method (`generate_anisotropic_field`):
//!   Same as isotropic, but before applying the isotropic projection we map
//!   `k -> k_0 = (k_1, k_2, k_3 + beta*k_1)` (undistorted wavenumber) and
//!   after the projection we apply the RDT distortion operator D(k, beta):
//!     D_11 = 1, D_13 = zeta_1,
//!     D_22 = 1, D_23 = zeta_2,
//!     D_33 = k_0^2 / k^2, else 0.
//!   The parameter `beta = Gamma * (kL)^(-2/3) / sqrt(F(kL))` is the eddy
//!   lifetime scaled by shear; `F(kL) = 2F1(1/3, 17/6; 4/3; -(kL)^(-2))` is
//!   computed via Pfaff's transformation so the series argument is in (0, 1).
//!   Gamma = 0 recovers the isotropic case exactly.
//!
//! Reference: Mann, J. (1994). "The spatial structure of neutral atmospheric
//! surface-layer turbulence." J. Fluid Mech. 273, 141-168.

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

/// Gauss's hypergeometric series `2F1(a, b; c; z)` for `|z| < 1`.
///
/// Truncated when successive terms fall below `1e-14` of the accumulated sum.
pub fn hyp_2f1(a: f64, b: f64, c: f64, z: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    for n in 0..5000 {
        let nf = n as f64;
        term *= (a + nf) * (b + nf) / ((c + nf) * (nf + 1.0)) * z;
        sum += term;
        if term.abs() < 1e-14 * sum.abs().max(1.0) {
            break;
        }
    }
    sum
}

/// Mann's hypergeometric `F(kL) = 2F1(1/3, 17/6; 4/3; -(kL)^(-2))`.
///
/// Uses Pfaff's transformation so the effective argument sits in `(0, 1)`,
/// keeping the series convergent for all kL > 0.
pub fn mann_hyper_f(k_l: f64) -> f64 {
    if k_l <= 0.0 {
        return 0.0;
    }
    let x2 = k_l * k_l;
    let y = 1.0 / (x2 + 1.0);
    let prefactor = (x2 / (x2 + 1.0)).powf(1.0 / 3.0);
    prefactor * hyp_2f1(1.0 / 3.0, -1.5, 4.0 / 3.0, y)
}

/// Mann's dimensionless distortion parameter `beta(k) = Gamma * (kL)^(-2/3) / sqrt(F(kL))`.
/// Returns 0 at k = 0 or Gamma = 0, so no RDT distortion is applied there.
pub fn mann_beta(k_mag: f64, length: f64, gamma: f64) -> f64 {
    if k_mag <= 0.0 || gamma == 0.0 {
        return 0.0;
    }
    let k_l = k_mag * length;
    let f = mann_hyper_f(k_l);
    if !(f > 0.0) {
        return 0.0;
    }
    gamma * k_l.powf(-2.0 / 3.0) / f.sqrt()
}

/// Generate a Mann anisotropic turbulence field via RDT applied to isotropic
/// von Karman turbulence. Reduces to `generate_isotropic_field` when `gamma = 0`.
pub fn generate_anisotropic_field(
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
    let sqrt_2_dk3 = (2.0 * dkx * dky * dkz).sqrt();

    let mut uk = Array3::<Complex64>::zeros(shape);
    let mut vk = Array3::<Complex64>::zeros(shape);
    let mut wk = Array3::<Complex64>::zeros(shape);

    let mut rng = StdRng::seed_from_u64(seed);

    for i in 0..nx {
        let k1 = fft_freq(i, nx) * dkx;
        for j in 0..ny {
            let k2 = fft_freq(j, ny) * dky;
            for k in 0..nz {
                let k3 = fft_freq(k, nz) * dkz;
                let k_mag2 = k1 * k1 + k2 * k2 + k3 * k3;
                if k_mag2 < 1e-24 {
                    continue;
                }
                let k_mag = k_mag2.sqrt();

                let beta = mann_beta(k_mag, config.length_scale_m, config.gamma);
                let k30 = k3 + beta * k1;
                let k0_mag2 = k1 * k1 + k2 * k2 + k30 * k30;
                let k0_mag = k0_mag2.sqrt();

                let e_k0 = von_karman_energy(k0_mag, config.alpha_epsilon_23, config.length_scale_m);
                let s_iso = (e_k0 / (4.0 * PI * k0_mag2)).sqrt() * sqrt_2_dk3;

                // Isotropic projection at k0: n0 = eta - (k0/|k0| . eta) k0/|k0|
                let n0x = k1 / k0_mag;
                let n0y = k2 / k0_mag;
                let n0z = k30 / k0_mag;

                let eta_x = complex_gaussian(&mut rng);
                let eta_y = complex_gaussian(&mut rng);
                let eta_z = complex_gaussian(&mut rng);
                let dot = n0x * eta_x + n0y * eta_y + n0z * eta_z;
                let iso_x = s_iso * (eta_x - dot * n0x);
                let iso_y = s_iso * (eta_y - dot * n0y);
                let iso_z = s_iso * (eta_z - dot * n0z);

                // RDT distortion operator D(k, beta). See Mann 1994 Eqs. (46)-(48).
                let (zeta1, zeta2) = distortion_zetas(k1, k2, k3, k30, k_mag2, k0_mag2, beta);
                let d33 = k0_mag2 / k_mag2;

                uk[[i, j, k]] = iso_x + zeta1 * iso_z;
                vk[[i, j, k]] = iso_y + zeta2 * iso_z;
                wk[[i, j, k]] = d33 * iso_z;
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

/// Compute `(zeta_1, zeta_2)` for the RDT distortion operator at wavenumber
/// `k = (k1, k2, k3)`, with `k30 = k3 + beta*k1` and pre-computed magnitudes.
///
/// Handles the `k1 -> 0`, `k1^2 + k2^2 -> 0`, and `beta = 0` degeneracies
/// analytically so the operator reduces smoothly.
fn distortion_zetas(
    k1: f64,
    k2: f64,
    _k3: f64,
    k30: f64,
    k_mag2: f64,
    k0_mag2: f64,
    beta: f64,
) -> (f64, f64) {
    if beta == 0.0 {
        return (0.0, 0.0);
    }
    let k12_horiz = k1 * k1 + k2 * k2;
    if k12_horiz < 1e-24 {
        return (0.0, 0.0);
    }

    // Argument of arctan is (beta * k1 * sqrt(k12_horiz)) / (k0^2 - k30*k1*beta).
    let arctan_num = beta * k1 * k12_horiz.sqrt();
    let arctan_den = k0_mag2 - k30 * k1 * beta;
    // Use atan2 so the branch is chosen correctly across the singular denominator.
    let c2_arctan = arctan_num.atan2(arctan_den);
    let c2_prefactor = k2 * k0_mag2 / k12_horiz.powf(1.5);
    let c2 = c2_prefactor * c2_arctan;

    if k1.abs() < 1e-12 {
        // k1 -> 0: C_1 -> 0 (has explicit k1^2), zeta_1 = -k2/k1 * C_2 which is
        // regular in the k1 -> 0 limit because C_2 also carries k2 * arctan ~ beta k1.
        // Evaluate the limit: arctan(beta*k1*sqrt(k12h)/(k0^2)) ~ beta*k1*sqrt(k12h)/k0^2
        // so C_2 ~ k2 * beta * k1 / k12h, and zeta_1 = -k2/k1 * C_2 = -k2^2 * beta / k12h.
        let zeta1_limit = -k2 * k2 * beta / k12_horiz;
        return (zeta1_limit, c2);
    }

    let c1 = beta * k1 * k1 * (k0_mag2 - 2.0 * k30 * k30 + beta * k1 * k30)
        / (k_mag2 * k12_horiz);
    let zeta1 = c1 - (k2 / k1) * c2;
    let zeta2 = (k2 / k1) * c1 + c2;
    (zeta1, zeta2)
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
    fn hyp_2f1_matches_known_values() {
        // 2F1(1, 1; 2; z) = -ln(1 - z) / z
        let z = 0.5;
        let expected = -(1.0f64 - z).ln() / z;
        let got = hyp_2f1(1.0, 1.0, 2.0, z);
        assert!((got - expected).abs() < 1e-10, "expected {expected}, got {got}");
    }

    #[test]
    fn mann_hyper_f_is_positive_and_bounded() {
        for &k_l in &[0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0] {
            let f = mann_hyper_f(k_l);
            assert!(f > 0.0 && f < 10.0, "F({k_l}) = {f} out of expected range");
        }
    }

    #[test]
    fn mann_beta_zero_when_gamma_zero() {
        assert_eq!(mann_beta(0.1, 20.0, 0.0), 0.0);
    }

    #[test]
    fn anisotropic_reduces_to_isotropic_when_gamma_zero() {
        let cfg_iso = MannConfig {
            alpha_epsilon_23: 0.585,
            length_scale_m: 16.8,
            gamma: 0.0,
        };
        let iso = generate_isotropic_field(&cfg_iso, [0.0; 3], [10.0, 10.0, 10.0], (16, 16, 16), 0.0, 42);
        let aniso = generate_anisotropic_field(&cfg_iso, [0.0; 3], [10.0, 10.0, 10.0], (16, 16, 16), 0.0, 42);
        // Same seed, same Gamma=0: should be numerically identical.
        for (a, b) in iso.u.iter().zip(aniso.u.iter()) {
            assert!((a - b).abs() < 1e-10, "u mismatch: {a} vs {b}");
        }
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
