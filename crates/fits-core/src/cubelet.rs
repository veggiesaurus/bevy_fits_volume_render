//! Cubes travel as cubelets of up to CUBELET³ voxels, cut from a fixed list of levels per file.

use crate::{Cube, Region, downsample, downsampled_dims, find_downsample_factors};

pub const CUBELET: usize = 64;

#[derive(Debug, PartialEq)]
pub struct Levels {
    /// Downsampling factors, level 0 first. Each level halves x and y while they're wider than a cubelet, and z
    /// while it's the longest axis, so it nests in and can be built from the one before. The last fits in one
    /// cubelet.
    pub chain: Vec<[usize; 3]>,
    /// The factors that fit the GPU budget and the device's layers, which may be one of the chain's.
    pub overview: [usize; 3],
}

pub fn levels(dims: [usize; 3], gpu_memory_limit_mb: u64, max_depth: usize) -> Levels {
    let mut chain = vec![[1; 3]];
    loop {
        let f = chain[chain.len() - 1];
        let [x, y, z] = downsampled_dims(Region::whole(dims), f);
        if x.max(y).max(z) <= CUBELET {
            break;
        }
        let halve = [x > CUBELET, y > CUBELET, z > x && z > y];
        chain.push(std::array::from_fn(|i| f[i] << usize::from(halve[i])));
    }
    Levels {
        chain,
        overview: find_downsample_factors(gpu_memory_limit_mb, dims, max_depth),
    }
}

/// A cubelet at its true size, which is smaller than CUBELET³ at the cube's far edges, and its `Cell`.
#[derive(Debug)]
pub struct Cubelet {
    pub dims: [usize; 3],
    pub voxels: Voxels,
    pub cell: Cell,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Voxels {
    AllNan,
    Constant(f32),
    Data(Vec<f32>),
}

impl Cubelet {
    fn new(dims: [usize; 3], data: Vec<f32>, cell: Cell) -> Self {
        let voxels = if data.iter().all(|v| v.is_nan()) {
            Voxels::AllNan
        } else if data.iter().all(|v| v.to_bits() == data[0].to_bits()) {
            Voxels::Constant(data[0])
        } else {
            Voxels::Data(data)
        };
        Self { dims, voxels, cell }
    }
}

/// What the cubelet at `at` of a cube of `dims` downsampled by `factors` covers of it, or None when `at` is outside
/// that level's grid of cubelets.
pub fn cubelet_region(dims: [usize; 3], factors: [usize; 3], at: [usize; 3]) -> Option<Region> {
    let mut region = Region::whole(dims);
    for i in 0..3 {
        let span = CUBELET.checked_mul(factors[i]).filter(|&span| span > 0)?;
        region.min[i] = at[i].checked_mul(span).filter(|&min| min < dims[i])?;
        region.max[i] = region.min[i].saturating_add(span).min(dims[i]) - 1;
    }
    Some(region)
}

/// The voxels of `region` of `data`, a cube of `dims`, a row at a time.
fn copy(data: &[f32], dims: [usize; 3], region: Region) -> Vec<f32> {
    let Region { min, max } = region;
    let mut out = Vec::with_capacity(region.extent().iter().product());
    for z in min[2]..=max[2] {
        for y in min[1]..=max[1] {
            let start = (z * dims[1] + y) * dims[0];
            out.extend_from_slice(&data[start + min[0]..=start + max[0]]);
        }
    }
    out
}

/// `region` grown by `by` each side, clamped to `dims`.
pub fn grow(region: Region, dims: [usize; 3], by: [usize; 3]) -> Region {
    Region {
        min: std::array::from_fn(|i| region.min[i].saturating_sub(by[i])),
        max: std::array::from_fn(|i| (region.max[i] + by[i]).min(dims[i] - 1)),
    }
}

/// The cubelet at `at` of level 0 downsampled by `factors`, cut straight from level 0, or None when `at` is outside
/// that level's grid.
pub fn cubelet(
    cube: &Cube,
    factors: [usize; 3],
    at: [usize; 3],
    max_mode: bool,
) -> Option<Cubelet> {
    let dims = cube.info.dims;
    let region = cubelet_region(dims, factors, at)?;
    if factors == [1; 3] {
        let cell = Cell::of(&cube.data, dims, grow(region, dims, [1; 3]));
        return Some(Cubelet::new(
            region.extent(),
            copy(&cube.data, dims, region),
            cell,
        ));
    }
    // A texel more each side for the cell, then the cubelet from inside that.
    let grown = grow(region, dims, factors);
    let (grown_dims, data, _) = downsample(&cube.data, dims, None, grown, factors, max_mode, false);
    let min: [usize; 3] = std::array::from_fn(|i| (region.min[i] - grown.min[i]) / factors[i]);
    let extent = downsampled_dims(region, factors);
    let inner = Region {
        min,
        max: std::array::from_fn(|i| min[i] + extent[i] - 1),
    };
    let cell = Cell::of(&data, grown_dims, Region::whole(grown_dims));
    Some(Cubelet::new(extent, copy(&data, grown_dims, inner), cell))
}

/// The largest value in a cubelet, ignoring NaN (-inf when there's nothing else), and whether it has a NaN. Both take
/// in a voxel past its faces as well, for samples that round across them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cell {
    pub max: f32,
    pub has_nan: bool,
}

impl Cell {
    /// Over `region` of `data`, a level of `dims`.
    pub fn of(data: &[f32], dims: [usize; 3], region: Region) -> Self {
        let Region { min, max } = region;
        let mut cell = Cell {
            max: f32::NEG_INFINITY,
            has_nan: false,
        };
        for z in min[2]..=max[2] {
            for y in min[1]..=max[1] {
                let start = (z * dims[1] + y) * dims[0];
                for &v in &data[start + min[0]..=start + max[0]] {
                    cell.has_nan |= v.is_nan();
                    cell.max = cell.max.max(v);
                }
            }
        }
        cell
    }
}

/// A whole level held in memory, so that its cubelets are copies and a coarser level takes one pass over it. Level 0
/// is the cube itself, so a chain starts at the level after it, built from the cube.
pub struct Level {
    pub factors: [usize; 3],
    pub dims: [usize; 3],
    pub data: Vec<f32>,
    pub max_mode: bool,
    /// For a mean, the finite level 0 voxels behind each voxel, which keep the means of coarser levels exact. Only
    /// needed until the next level is built.
    pub(crate) counts: Vec<u32>,
}

impl Level {
    pub fn new(cube: &Cube, factors: [usize; 3], max_mode: bool) -> Self {
        assert_ne!(factors, [1; 3], "level 0 is the cube itself");
        let dims = cube.info.dims;
        let (dims, data, counts) = downsample(
            &cube.data,
            dims,
            None,
            Region::whole(dims),
            factors,
            max_mode,
            !max_mode,
        );
        Self {
            factors,
            dims,
            data,
            max_mode,
            counts,
        }
    }

    /// The level at `factors`, each a multiple of this level's.
    pub fn chain(&self, factors: [usize; 3]) -> Self {
        assert!(
            (0..3).all(|i| factors[i].is_multiple_of(self.factors[i])),
            "{factors:?} doesn't divide by {:?}",
            self.factors
        );
        let step = std::array::from_fn(|i| factors[i] / self.factors[i]);
        let max_mode = self.max_mode;
        let weights = (!max_mode).then_some(&self.counts[..]);
        let whole = Region::whole(self.dims);
        let (dims, data, counts) = downsample(
            &self.data, self.dims, weights, whole, step, max_mode, !max_mode,
        );
        Self {
            factors,
            dims,
            data,
            max_mode,
            counts,
        }
    }

    /// The cubelet at `at` in this level's grid, or None when `at` is outside it.
    pub fn cubelet(&self, at: [usize; 3]) -> Option<Cubelet> {
        let region = cubelet_region(self.dims, [1; 3], at)?;
        Some(Cubelet::new(
            region.extent(),
            copy(&self.data, self.dims, region),
            Cell::of(&self.data, self.dims, grow(region, self.dims, [1; 3])),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{crop_and_downsample, tests::cube};

    #[test]
    fn levels_halve_x_and_y_down_to_a_cubelet_and_z_when_its_longest() {
        // A tie leaves z alone, as find_downsample_factors does, and the overview is the chain's second level.
        assert_eq!(
            levels([512, 512, 512], 384, 2048),
            Levels {
                chain: vec![[1, 1, 1], [2, 2, 1], [4, 4, 2], [8, 8, 4], [8, 8, 8]],
                overview: [2, 2, 1],
            }
        );
        // x and y stay at 50 while z halves to 63, and the overview is today's 1 x 1 x 2.
        assert_eq!(
            levels([100, 100, 4000], 384, 2048),
            Levels {
                chain: vec![
                    [1, 1, 1],
                    [2, 2, 2],
                    [2, 2, 4],
                    [2, 2, 8],
                    [2, 2, 16],
                    [2, 2, 32],
                    [2, 2, 64]
                ],
                overview: [1, 1, 2],
            }
        );
        // The overview is today's 5 x 5 x 1, where the power-of-two levels skip from 4 to 8.
        assert_eq!(
            levels([4096, 4096, 100], 384, 2048),
            Levels {
                chain: vec![
                    [1, 1, 1],
                    [2, 2, 1],
                    [4, 4, 1],
                    [8, 8, 1],
                    [16, 16, 1],
                    [32, 32, 1],
                    [64, 64, 1],
                    [64, 64, 2]
                ],
                overview: [5, 5, 1],
            }
        );
    }

    #[test]
    fn cells_reach_a_voxel_past_their_faces() {
        let cell = |max, has_nan| Cell { max, has_nan };
        let cell_summary = |voxels: &[f32], dims: [usize; 3]| -> Vec<Cell> {
            let cube = cube(dims, voxels.to_vec());
            let [x, y, z] = dims.map(|d| d.div_ceil(CUBELET));
            (0..z)
                .flat_map(|z| (0..y).flat_map(move |y| (0..x).map(move |x| [x, y, z])))
                .map(|at| cubelet(&cube, [1; 3], at, true).unwrap().cell)
                .collect()
        };
        // Two cells along x, the second 6 voxels wide; all NaN apart from the voxels named below.
        let dims = [70, 2, 1];
        let mut voxels = vec![f32::NAN; 140];
        let at = |x: usize, y: usize| y * 70 + x;
        // Cell 0's last voxel: its own max and, a voxel across the face, cell 1's.
        voxels[at(63, 1)] = 5.0;
        // Cell 1's first voxel, which cell 0 takes in across the face.
        voxels[at(64, 0)] = 7.0;
        voxels[at(66, 0)] = 3.0;
        let summary = cell_summary(&voxels, dims);
        assert_eq!(summary, [cell(7.0, true), cell(7.0, true)]);
        voxels[at(63, 1)] = f32::NAN;
        voxels[at(64, 0)] = f32::NAN;
        let summary = cell_summary(&voxels, dims);
        assert_eq!(summary, [cell(f32::NEG_INFINITY, true), cell(3.0, true)]);
        assert_eq!(cell_summary(&[1.0; 140], dims), [cell(1.0, false); 2]);

        // The same in y and z, on zeros: 9 on a y face, 5 on a z face, and a NaN that only its own cell takes in.
        let dims = [1, 66, 66];
        let mut voxels = vec![0.0; 66 * 66];
        let at = |y: usize, z: usize| z * 66 + y;
        voxels[at(64, 10)] = 9.0;
        voxels[at(30, 63)] = 5.0;
        voxels[at(65, 65)] = f32::NAN;
        let summary = cell_summary(&voxels, dims);
        let want = [
            cell(9.0, false),
            cell(9.0, false),
            cell(5.0, false),
            cell(0.0, true),
        ];
        assert_eq!(summary, want);
    }

    #[test]
    fn cubelets_from_level_0_and_from_chained_levels_agree() {
        // Dims that no factor divides, small integers so that sums are exact, scattered NaN and infinities, an
        // all-NaN corner that's the last level 0 cubelet, and a constant strip that's the cubelet at [0, 2, 0].
        let dims = [300, 140, 9];
        let mut seed = 12345u32;
        let data = (0..dims.iter().product::<usize>())
            .map(|i| {
                let [x, y] = [i % dims[0], i / dims[0] % dims[1]];
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                match (x, y) {
                    (256.., 128..) => f32::NAN,
                    (..64, 128..) => 7.0,
                    _ if seed.is_multiple_of(5) => f32::NAN,
                    _ if seed.is_multiple_of(89) => f32::INFINITY,
                    _ if seed.is_multiple_of(97) => f32::NEG_INFINITY,
                    _ => (seed >> 22) as f32,
                }
            })
            .collect();
        let cube = cube(dims, data);
        let same = |a: &[f32], b: &[f32], tolerance: f32| {
            a.len() == b.len()
                && a.iter().zip(b).all(|(a, b)| {
                    a == b || a.is_nan() && b.is_nan() || (a - b).abs() <= tolerance * b.abs()
                })
        };
        let grid = |dims: [usize; 3]| {
            let [x, y, z] = dims.map(|d| d.div_ceil(CUBELET));
            (0..z).flat_map(move |z| (0..y).flat_map(move |y| (0..x).map(move |x| [x, y, z])))
        };

        let edge = cubelet(&cube, [1; 3], [4, 2, 0], true).unwrap();
        assert_eq!((edge.dims, edge.voxels), ([44, 12, 9], Voxels::AllNan));
        assert_eq!(
            cubelet(&cube, [1; 3], [0, 2, 0], false).unwrap().voxels,
            Voxels::Constant(7.0)
        );
        for (factors, at) in [
            ([1; 3], [5, 0, 0]),
            ([2, 2, 1], [usize::MAX; 3]),
            ([0, 1, 1], [0; 3]),
        ] {
            assert!(
                cubelet(&cube, factors, at, true).is_none(),
                "{factors:?} {at:?}"
            );
        }

        for max_mode in [true, false] {
            // Max of max is exact; a mean of means only to rounding, and only because each is weighted.
            let tolerance = if max_mode { 0.0 } else { 1e-5 };
            let first = Level::new(&cube, [2, 2, 1], max_mode);
            let chained = [
                first.chain([4, 4, 2]),
                first.chain([6, 6, 3]).chain([12, 12, 9]),
            ];
            for level in [first].iter().chain(&chained) {
                let direct = crop_and_downsample(
                    &cube.data,
                    dims,
                    Region::whole(dims),
                    level.factors,
                    max_mode,
                );
                assert_eq!(level.dims, direct.0);
                let past = level.dims[2].div_ceil(CUBELET);
                assert!(level.cubelet([0, 0, past]).is_none());
                assert!(
                    same(&level.data, &direct.1, tolerance),
                    "{:?}",
                    level.factors
                );
                for at in grid(level.dims) {
                    let (ours, cut) = (
                        level.cubelet(at).unwrap(),
                        cubelet(&cube, level.factors, at, max_mode).unwrap(),
                    );
                    assert_eq!(ours.dims, cut.dims);
                    if max_mode {
                        assert_eq!(ours.cell, cut.cell, "{:?} {at:?}", level.factors);
                    }
                    let voxels = |c: Cubelet| match c.voxels {
                        Voxels::AllNan => vec![f32::NAN; c.dims.iter().product()],
                        Voxels::Constant(v) => vec![v; c.dims.iter().product()],
                        Voxels::Data(data) => data,
                    };
                    assert!(
                        same(&voxels(ours), &voxels(cut), tolerance),
                        "{:?} {at:?}",
                        level.factors
                    );
                }
            }
        }
    }
}
