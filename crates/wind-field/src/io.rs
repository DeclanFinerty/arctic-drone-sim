//! Serialize wind grids to disk as three `.npy` files (one per component) plus
//! a JSON metadata sidecar. Python readers can load the arrays with
//! `numpy.load` and the sidecar with `json.load`.

use std::fs;
use std::path::Path;

use ndarray_npy::{ReadNpyError, ReadableElement, WriteNpyError, read_npy, write_npy};

use crate::grid::{GridMetadata, WindGrid};
use ndarray::Array3;

#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("npy write: {0}")]
    NpyWrite(#[from] WriteNpyError),
    #[error("npy read: {0}")]
    NpyRead(#[from] ReadNpyError),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("component shape mismatch: {0:?} vs {1:?}")]
    ShapeMismatch([usize; 3], [usize; 3]),
}

/// Write `grid` into `dir/` as `u.npy`, `v.npy`, `w.npy`, `metadata.json`.
///
/// Creates `dir` if it does not exist. Overwrites existing files.
pub fn save_grid(grid: &WindGrid, dir: &Path, generator: impl Into<String>) -> Result<(), IoError> {
    fs::create_dir_all(dir)?;
    write_npy(dir.join("u.npy"), &grid.u)?;
    write_npy(dir.join("v.npy"), &grid.v)?;
    write_npy(dir.join("w.npy"), &grid.w)?;
    let meta = GridMetadata::from_grid(grid, generator);
    fs::write(dir.join("metadata.json"), serde_json::to_string_pretty(&meta)?)?;
    Ok(())
}

/// Load a grid previously written by [`save_grid`].
pub fn load_grid(dir: &Path) -> Result<WindGrid, IoError> {
    let u: Array3<f64> = read_npy(dir.join("u.npy"))?;
    let v: Array3<f64> = read_npy_matching(&u, dir.join("v.npy"))?;
    let w: Array3<f64> = read_npy_matching(&u, dir.join("w.npy"))?;
    let meta: GridMetadata = serde_json::from_slice(&fs::read(dir.join("metadata.json"))?)?;
    Ok(WindGrid {
        u,
        v,
        w,
        origin: meta.origin,
        spacing: meta.spacing,
    })
}

fn read_npy_matching<T: ReadableElement>(
    reference: &Array3<T>,
    path: impl AsRef<Path>,
) -> Result<Array3<T>, IoError> {
    let arr: Array3<T> = read_npy(path)?;
    if arr.dim() != reference.dim() {
        let (a, b, c) = arr.dim();
        let (ra, rb, rc) = reference.dim();
        return Err(IoError::ShapeMismatch([a, b, c], [ra, rb, rc]));
    }
    Ok(arr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytical::PowerLawShear;
    use crate::grid::sample_to_grid;

    fn scratch_dir(name: &str) -> std::path::PathBuf {
        let base = std::env::temp_dir().join("arctic-drone-sim-tests").join(name);
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        base
    }

    #[test]
    fn round_trip_preserves_grid() {
        let field = PowerLawShear {
            u_ref_ms: 6.0,
            z_ref_m: 10.0,
            alpha: 1.0 / 7.0,
            bounds: ([0.0; 3], [50.0, 50.0, 50.0]),
        };
        let grid = sample_to_grid(&field, [0.0, 0.0, 1.0], [5.0, 5.0, 1.0], (11, 11, 50));
        let dir = scratch_dir("roundtrip");
        save_grid(&grid, &dir, "test").unwrap();
        let loaded = load_grid(&dir).unwrap();
        assert_eq!(loaded.shape(), grid.shape());
        assert_eq!(loaded.origin, grid.origin);
        assert_eq!(loaded.spacing, grid.spacing);
        for (a, b) in loaded.u.iter().zip(grid.u.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
