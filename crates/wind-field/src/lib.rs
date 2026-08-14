pub mod fitting;
pub mod grid;
pub mod io;
pub mod mann;

/// Query a wind field at arbitrary continuous coordinates.
///
/// Implementations interpolate between the underlying discrete samples.
/// Positions are `[x, y, z]` in meters, time in seconds. Returned wind
/// vector is `[u, v, w]` in m/s.
pub trait WindFieldQuery {
    fn wind_at(&self, position: [f64; 3], time: f64) -> [f64; 3];

    /// Returns `(min, max)` corner coordinates of the domain in meters.
    fn domain_bounds(&self) -> ([f64; 3], [f64; 3]);
}
