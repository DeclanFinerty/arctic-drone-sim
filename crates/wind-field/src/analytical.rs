//! Closed-form wind fields used to validate the grid, interpolation, and I/O
//! infrastructure before the Mann turbulence model is introduced.

use std::f64::consts::PI;

use crate::WindFieldQuery;

/// Constant wind vector at every point in the domain.
#[derive(Debug, Clone)]
pub struct Uniform {
    pub velocity: [f64; 3],
    pub bounds: ([f64; 3], [f64; 3]),
}

impl WindFieldQuery for Uniform {
    fn wind_at(&self, _position: [f64; 3], _time: f64) -> [f64; 3] {
        self.velocity
    }

    fn domain_bounds(&self) -> ([f64; 3], [f64; 3]) {
        self.bounds
    }
}

/// Power-law shear profile aligned with the x-axis:
///
/// ```text
/// u(z) = u_ref * (z / z_ref)^alpha
/// v(z) = w(z) = 0
/// ```
///
/// Values at z <= 0 are clamped to a small positive height to avoid a
/// singularity when alpha < 1.
#[derive(Debug, Clone)]
pub struct PowerLawShear {
    pub u_ref_ms: f64,
    pub z_ref_m: f64,
    pub alpha: f64,
    pub bounds: ([f64; 3], [f64; 3]),
}

impl WindFieldQuery for PowerLawShear {
    fn wind_at(&self, position: [f64; 3], _time: f64) -> [f64; 3] {
        let z = position[2].max(1e-6);
        let u = self.u_ref_ms * (z / self.z_ref_m).powf(self.alpha);
        [u, 0.0, 0.0]
    }

    fn domain_bounds(&self) -> ([f64; 3], [f64; 3]) {
        self.bounds
    }
}

/// Single sinusoidal Fourier mode. Useful as a plumbing check for the grid
/// sampling machinery and, later, the FFT round-trip in the Mann pipeline.
///
/// ```text
/// [u, v, w](x, y, z) = amplitude * sin(2*pi * (kx*x + ky*y + kz*z))
/// ```
///
/// `wavenumber` components are in 1/m (i.e. spatial frequency).
#[derive(Debug, Clone)]
pub struct SingleFourierMode {
    pub amplitude: [f64; 3],
    pub wavenumber: [f64; 3],
    pub bounds: ([f64; 3], [f64; 3]),
}

impl WindFieldQuery for SingleFourierMode {
    fn wind_at(&self, position: [f64; 3], _time: f64) -> [f64; 3] {
        let phase = 2.0 * PI
            * (position[0] * self.wavenumber[0]
                + position[1] * self.wavenumber[1]
                + position[2] * self.wavenumber[2]);
        let s = phase.sin();
        [
            self.amplitude[0] * s,
            self.amplitude[1] * s,
            self.amplitude[2] * s,
        ]
    }

    fn domain_bounds(&self) -> ([f64; 3], [f64; 3]) {
        self.bounds
    }
}
