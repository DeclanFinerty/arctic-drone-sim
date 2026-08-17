//! 3D grid representation for sampled wind fields with trilinear interpolation.

use ndarray::Array3;
use serde::{Deserialize, Serialize};

use crate::WindFieldQuery;

/// A regular 3D grid of wind vectors on a Cartesian domain.
///
/// The u/v/w arrays share shape `(nx, ny, nz)`. Grid point `(i, j, k)` lies at
/// `origin + [i, j, k] * spacing`. Time is not represented — this is a snapshot.
#[derive(Debug, Clone)]
pub struct WindGrid {
    pub u: Array3<f64>,
    pub v: Array3<f64>,
    pub w: Array3<f64>,
    pub origin: [f64; 3],
    pub spacing: [f64; 3],
}

impl WindGrid {
    pub fn shape(&self) -> (usize, usize, usize) {
        self.u.dim()
    }

    /// Coordinates of the far corner of the grid.
    pub fn max_corner(&self) -> [f64; 3] {
        let (nx, ny, nz) = self.shape();
        [
            self.origin[0] + (nx.saturating_sub(1)) as f64 * self.spacing[0],
            self.origin[1] + (ny.saturating_sub(1)) as f64 * self.spacing[1],
            self.origin[2] + (nz.saturating_sub(1)) as f64 * self.spacing[2],
        ]
    }
}

/// Sample any analytical wind field onto a fresh regular grid.
pub fn sample_to_grid<F: WindFieldQuery + ?Sized>(
    field: &F,
    origin: [f64; 3],
    spacing: [f64; 3],
    shape: (usize, usize, usize),
) -> WindGrid {
    let (nx, ny, nz) = shape;
    let mut u = Array3::<f64>::zeros(shape);
    let mut v = Array3::<f64>::zeros(shape);
    let mut w = Array3::<f64>::zeros(shape);
    for i in 0..nx {
        let x = origin[0] + i as f64 * spacing[0];
        for j in 0..ny {
            let y = origin[1] + j as f64 * spacing[1];
            for k in 0..nz {
                let z = origin[2] + k as f64 * spacing[2];
                let vec = field.wind_at([x, y, z], 0.0);
                u[[i, j, k]] = vec[0];
                v[[i, j, k]] = vec[1];
                w[[i, j, k]] = vec[2];
            }
        }
    }
    WindGrid {
        u,
        v,
        w,
        origin,
        spacing,
    }
}

impl WindFieldQuery for WindGrid {
    fn wind_at(&self, position: [f64; 3], _time: f64) -> [f64; 3] {
        let (nx, ny, nz) = self.shape();
        // Fractional index in each axis, clamped so out-of-domain queries
        // return the nearest edge value rather than panicking.
        let fx = index_frac(position[0], self.origin[0], self.spacing[0], nx);
        let fy = index_frac(position[1], self.origin[1], self.spacing[1], ny);
        let fz = index_frac(position[2], self.origin[2], self.spacing[2], nz);

        [
            trilinear(&self.u, fx, fy, fz),
            trilinear(&self.v, fx, fy, fz),
            trilinear(&self.w, fx, fy, fz),
        ]
    }

    fn domain_bounds(&self) -> ([f64; 3], [f64; 3]) {
        (self.origin, self.max_corner())
    }
}

/// Turns a physical coordinate into `(i, t)` where `i` is the lower grid index
/// and `t` in `[0, 1]` is the fractional distance to `i + 1`. Clamps to the
/// valid interpolation range `[0, n-1]`.
fn index_frac(coord: f64, origin: f64, spacing: f64, n: usize) -> (usize, f64) {
    if n == 0 {
        return (0, 0.0);
    }
    let raw = (coord - origin) / spacing;
    let max_lower = n.saturating_sub(1) as f64;
    let clamped = raw.clamp(0.0, max_lower);
    let i = clamped.floor() as usize;
    let i = i.min(n.saturating_sub(2).max(0));
    let t = clamped - i as f64;
    (i, t)
}

fn trilinear(arr: &Array3<f64>, fx: (usize, f64), fy: (usize, f64), fz: (usize, f64)) -> f64 {
    let (i, tx) = fx;
    let (j, ty) = fy;
    let (k, tz) = fz;
    let (nx, ny, nz) = arr.dim();
    // For 1-cell axes, tx/ty/tz are 0 and i+1 index would be out of bounds — clamp.
    let i1 = (i + 1).min(nx - 1);
    let j1 = (j + 1).min(ny - 1);
    let k1 = (k + 1).min(nz - 1);

    let c000 = arr[[i, j, k]];
    let c100 = arr[[i1, j, k]];
    let c010 = arr[[i, j1, k]];
    let c110 = arr[[i1, j1, k]];
    let c001 = arr[[i, j, k1]];
    let c101 = arr[[i1, j, k1]];
    let c011 = arr[[i, j1, k1]];
    let c111 = arr[[i1, j1, k1]];

    let c00 = c000 * (1.0 - tx) + c100 * tx;
    let c10 = c010 * (1.0 - tx) + c110 * tx;
    let c01 = c001 * (1.0 - tx) + c101 * tx;
    let c11 = c011 * (1.0 - tx) + c111 * tx;

    let c0 = c00 * (1.0 - ty) + c10 * ty;
    let c1 = c01 * (1.0 - ty) + c11 * ty;

    c0 * (1.0 - tz) + c1 * tz
}

/// Taylor's frozen-turbulence wrapper: exposes a static field as if it were
/// advecting past the query point at a constant `mean_flow_ms`. Common
/// approximation for atmospheric turbulence at scales small compared to the
/// eddy lifetime (usually valid for drone dynamics).
#[derive(Debug, Clone)]
pub struct AdvectedField<F> {
    pub inner: F,
    pub mean_flow_ms: [f64; 3],
}

impl<F: WindFieldQuery> WindFieldQuery for AdvectedField<F> {
    fn wind_at(&self, position: [f64; 3], time: f64) -> [f64; 3] {
        let advected = [
            position[0] - self.mean_flow_ms[0] * time,
            position[1] - self.mean_flow_ms[1] * time,
            position[2] - self.mean_flow_ms[2] * time,
        ];
        self.inner.wind_at(advected, 0.0)
    }

    fn domain_bounds(&self) -> ([f64; 3], [f64; 3]) {
        self.inner.domain_bounds()
    }
}

/// JSON-serializable metadata sidecar written next to the .npy arrays.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridMetadata {
    pub shape: [usize; 3],
    pub origin: [f64; 3],
    pub spacing: [f64; 3],
    pub generator: String,
    pub notes: Option<String>,
}

impl GridMetadata {
    pub fn from_grid(grid: &WindGrid, generator: impl Into<String>) -> Self {
        let (nx, ny, nz) = grid.shape();
        Self {
            shape: [nx, ny, nz],
            origin: grid.origin,
            spacing: grid.spacing,
            generator: generator.into(),
            notes: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytical::{PowerLawShear, SingleFourierMode, Uniform};
    use std::f64::consts::PI;

    const TOL: f64 = 1e-10;

    #[test]
    fn uniform_sampling_and_interpolation() {
        let field = Uniform {
            velocity: [7.0, -1.0, 0.5],
            bounds: ([0.0; 3], [100.0, 100.0, 50.0]),
        };
        let grid = sample_to_grid(&field, [0.0; 3], [10.0, 10.0, 5.0], (11, 11, 11));
        for &wind in grid.u.iter() {
            assert!((wind - 7.0).abs() < TOL);
        }
        // Interpolation at an arbitrary interior point returns the constant.
        let interp = grid.wind_at([37.5, 42.1, 13.3], 0.0);
        assert!((interp[0] - 7.0).abs() < TOL);
        assert!((interp[1] + 1.0).abs() < TOL);
        assert!((interp[2] - 0.5).abs() < TOL);
    }

    #[test]
    fn power_law_shear_matches_formula_at_grid_points_and_midpoints() {
        let field = PowerLawShear {
            u_ref_ms: 6.0,
            z_ref_m: 10.0,
            alpha: 1.0 / 7.0,
            bounds: ([0.0; 3], [100.0, 100.0, 100.0]),
        };
        let grid = sample_to_grid(&field, [0.0, 0.0, 1.0], [10.0, 10.0, 1.0], (11, 11, 100));

        for k in 0..grid.shape().2 {
            let z: f64 = 1.0 + k as f64;
            let expected: f64 = 6.0 * (z / 10.0).powf(1.0f64 / 7.0);
            assert!((grid.u[[0, 0, k]] - expected).abs() < TOL);
        }
        // Trilinear interpolation is linear in z; the power-law is smooth,
        // so midpoint error is small but non-zero.
        let z: f64 = 12.5;
        let expected: f64 = 6.0 * (z / 10.0).powf(1.0f64 / 7.0);
        let got = grid.wind_at([0.0, 0.0, z], 0.0);
        assert!(
            (got[0] - expected).abs() < 1e-3,
            "expected {expected}, got {}",
            got[0]
        );
    }

    #[test]
    fn single_mode_grid_matches_analytical_at_grid_points() {
        // Wavelength = 20m, spacing = 1m -> 20 samples/wavelength, well resolved.
        let field = SingleFourierMode {
            amplitude: [2.0, 0.0, 0.0],
            wavenumber: [1.0 / 20.0, 0.0, 0.0],
            bounds: ([0.0; 3], [40.0, 10.0, 10.0]),
        };
        let grid = sample_to_grid(&field, [0.0; 3], [1.0, 1.0, 1.0], (41, 11, 11));
        for i in 0..41 {
            let x = i as f64;
            let expected = 2.0 * (2.0 * PI * x / 20.0).sin();
            assert!((grid.u[[i, 0, 0]] - expected).abs() < TOL);
        }
        // Maximum should be near x = 5m (quarter wavelength).
        let peak = grid.wind_at([5.0, 0.0, 0.0], 0.0);
        assert!((peak[0] - 2.0).abs() < TOL);
    }

    #[test]
    fn out_of_bounds_query_clamps_to_edge() {
        let field = Uniform {
            velocity: [3.0, 0.0, 0.0],
            bounds: ([0.0; 3], [10.0, 10.0, 10.0]),
        };
        let grid = sample_to_grid(&field, [0.0; 3], [1.0, 1.0, 1.0], (11, 11, 11));
        let far = grid.wind_at([1000.0, -50.0, 999.0], 0.0);
        assert!((far[0] - 3.0).abs() < TOL);
    }
}
