//! Noise statistics, noise normalisation, smoothing and a smooth-and-clip source finder, over voxels held x fastest,
//! and the engine's versions of them for a level's texels: measured, cut as cubelets, or made into a mask.

use bevy_tasks::{ComputeTaskPool, TaskPool};

use crate::{
    Cell, Cube, Cubelet, Histogram, MAX_BINS, Region, count, cubelet::copy, cubelet_region,
    downsample, downsampled_dims, grow, median,
};

/// A zero point and standard deviation from medians, which emission barely moves, in data units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Noise {
    pub median: f32,
    pub sigma: f32,
}

/// What's done to a level's texels before they're shown.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Processing {
    pub normalise: bool,
    pub smoothing: Smoothing,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Smoothing {
    #[default]
    Off,
    /// Hanning along z.
    Spectral,
    /// The 3D binomial, its spatial taps two texels apart.
    Spatial,
}

impl Processing {
    pub fn is_none(self) -> bool {
        self == Self::default()
    }
}

/// About this many voxels a side of each channel's sample grid, in up to this many channels.
const GRID: usize = 64;
const CHANNELS: usize = 256;
/// Spectra sampled a side, for normalisation.
const SPECTRA: usize = 128;

/// The finder's box kernels, in voxels across, spatially and spectrally: every pair of them.
const SPATIAL: [usize; 2] = [1, 3];
const SPECTRAL: [usize; 3] = [1, 3, 7];
/// A component spanning fewer voxels than this along any axis is dropped, as SoFiA's linker drops those under its
/// minimum size (5 by default, for full-resolution cubes).
const MIN_EXTENT: u16 = 3;
/// Past this many runs the mask is mostly noise, and its components are left as they are.
const MAX_RUNS: usize = 1 << 22;

/// Hanning along z, and the 3D binomial's spatial taps two voxels apart.
const HANNING: [(isize, f32); 3] = [(-1, 0.25), (0, 0.5), (1, 0.25)];
const BINOMIAL_XY: [(isize, f32); 3] = [(-2, 0.25), (0, 0.5), (2, 0.25)];

/// About 64 x 64 of each channel's finite values, stepping x and y apart so that the step can't alias with the
/// width, in up to 256 channels or all of them.
fn channel_samples(
    dims: [usize; 3],
    every_channel: bool,
    value: impl Fn([usize; 3]) -> f32,
) -> impl Iterator<Item = Vec<f32>> {
    let [width, height, depth] = dims;
    let (step_x, step_y) = (width.div_ceil(GRID), height.div_ceil(GRID));
    let step_z = if every_channel {
        1
    } else {
        depth.div_ceil(CHANNELS)
    };
    (0..depth).step_by(step_z).map(move |z| {
        let points = (0..height)
            .step_by(step_y)
            .flat_map(|y| (0..width).step_by(step_x).map(move |x| [x, y]));
        points
            .map(|[x, y]| value([x, y, z]))
            .filter(|v| v.is_finite())
            .collect()
    })
}

/// The median and 1.4826 MAD of `values`, which it reorders; None where the MAD is zero, as in a zero-filled or
/// flagged channel, or there's nothing.
fn median_mad(values: &mut [f32]) -> Option<Noise> {
    let centre = median(values)?;
    values.iter_mut().for_each(|v| *v = (*v - centre).abs());
    let mad = median(values)?;
    (mad > 0.0).then_some(Noise {
        median: centre,
        sigma: 1.4826 * mad,
    })
}

/// The middle of the quietest quarter by σ, so that neither emission nor a few unusually flat channels set it.
fn quietest(channels: &[Option<Noise>]) -> Option<Noise> {
    let mut channels: Vec<Noise> = channels.iter().flatten().copied().collect();
    channels.sort_by(|a, b| a.sigma.total_cmp(&b.sigma));
    channels.get(channels.len().div_ceil(4) / 2).copied()
}

/// The quietest channels' noise, from `channel_samples`, or where every channel's MAD is zero, as in heavily
/// quantised data, `histogram_noise` of all the samples.
fn sampled_noise(dims: [usize; 3], value: impl Fn([usize; 3]) -> f32) -> Option<Noise> {
    let mut all = Vec::new();
    let channels: Vec<_> = channel_samples(dims, false, value)
        .map(|mut v| {
            all.extend_from_slice(&v);
            median_mad(&mut v)
        })
        .collect();
    quietest(&channels).or_else(|| histogram_noise(&all))
}

/// The median and IQR / 1.349 of finite `values`, from a histogram of them, which interpolates within a bin and so
/// isn't zero where most values are equal. None when they all are.
fn histogram_noise(values: &[f32]) -> Option<Noise> {
    let (min, max) = (values.iter()).fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
        (lo.min(v), hi.max(v))
    });
    let range = f64::from(max) - f64::from(min);
    if range <= 0.0 {
        return None;
    }
    let bins = ((values.len() as f64).sqrt().round() as usize).clamp(1, MAX_BINS);
    let histogram = Histogram::of_counts(&count(values, min, max, bins));
    let at = |rank| f64::from(min) + histogram.fraction_below(rank) * range;
    let sigma = (at(0.75) - at(0.25)) / 1.349;
    (sigma > 0.0).then(|| Noise {
        median: at(0.5) as f32,
        sigma: sigma as f32,
    })
}

fn index(dims: [usize; 3], [x, y, z]: [usize; 3]) -> usize {
    x + dims[0] * (y + dims[1] * z)
}

/// The noise of `voxels`, a cube of `dims`.
pub fn noise(voxels: &[f32], dims: [usize; 3]) -> Option<Noise> {
    sampled_noise(dims, |p| voxels[index(dims, p)])
}

/// A level of the cube: held in memory, or read from level 0 a block at a time, each a max ignoring NaN or its
/// finite voxels' mean.
#[derive(Clone, Copy)]
pub struct LevelOf<'a> {
    cube: &'a Cube,
    factors: [usize; 3],
    max_mode: bool,
    dims: [usize; 3],
    held: Option<&'a [f32]>,
}

impl<'a> LevelOf<'a> {
    /// `held` is the level's texels when it's been built. Level 0 is held as the cube.
    pub fn new(
        cube: &'a Cube,
        factors: [usize; 3],
        max_mode: bool,
        held: Option<&'a [f32]>,
    ) -> Self {
        Self {
            cube,
            factors,
            max_mode,
            dims: downsampled_dims(Region::whole(cube.info.dims), factors),
            held: held.or((factors == [1; 3]).then_some(&cube.data[..])),
        }
    }

    /// NaN outside the level.
    fn texel(&self, p: [usize; 3]) -> f32 {
        if (0..3).any(|i| p[i] >= self.dims[i]) {
            return f32::NAN;
        }
        if let Some(held) = self.held {
            return held[index(self.dims, p)];
        }
        let (dims, f) = (self.cube.info.dims, self.factors);
        let (mut acc, mut n) = (if self.max_mode { f32::NAN } else { 0.0 }, 0u32);
        for z in p[2] * f[2]..((p[2] + 1) * f[2]).min(dims[2]) {
            for y in p[1] * f[1]..((p[1] + 1) * f[1]).min(dims[1]) {
                let row = index(dims, [0, y, z]);
                for &v in &self.cube.data[row + p[0] * f[0]..row + ((p[0] + 1) * f[0]).min(dims[0])]
                {
                    if self.max_mode {
                        acc = acc.max(v);
                    } else if v.is_finite() {
                        (acc, n) = (acc + v, n + 1);
                    }
                }
            }
        }
        match (self.max_mode, n) {
            (true, _) => acc,
            (false, 0) => f32::NAN,
            (false, n) => acc / n as f32,
        }
    }

    /// The texels of `texels`, all inside the level.
    fn block(&self, texels: Region) -> Vec<f32> {
        if let Some(held) = self.held {
            return copy(held, self.dims, texels);
        }
        let (dims, f) = (self.cube.info.dims, self.factors);
        let voxels = Region {
            min: std::array::from_fn(|i| texels.min[i] * f[i]),
            max: std::array::from_fn(|i| ((texels.max[i] + 1) * f[i]).min(dims[i]) - 1),
        };
        downsample(&self.cube.data, dims, None, voxels, f, self.max_mode, false).1
    }
}

/// Noise normalisation after SoFiA-2, measured on a level: each channel and then each spectrum is rescaled about its
/// median so that its σ is the quietest channels'. The spectra are sampled on a grid, each σ averaged with the 3 x 3
/// around it, since a few hundred channels give a MAD good to only about 10%. Zeros and non-finite texels stay as they
/// are. Changes fluxes.
pub struct Normaliser {
    reference: Noise,
    channels: Vec<Option<Noise>>,
    step: usize,
    columns: usize,
    spectra: Vec<Option<Noise>>,
}

impl Normaliser {
    /// From a level read once: a level that isn't held is made in one pass, rather than a block of level 0 a texel.
    pub fn measure(level: &LevelOf) -> Option<Self> {
        let made;
        let level = match level.held {
            Some(_) => *level,
            None => {
                made = level.block(Region::whole(level.dims));
                LevelOf {
                    held: Some(&made),
                    ..*level
                }
            }
        };
        let [width, height, depth] = level.dims;
        let channels: Vec<_> = channel_samples(level.dims, true, |p| level.texel(p))
            .map(|mut v| median_mad(&mut v))
            .collect();
        let reference = quietest(&channels)?;
        let step = width.max(height).div_ceil(SPECTRA).max(1);
        let (columns, rows) = (width.div_ceil(step), height.div_ceil(step));
        let mut normaliser = Self {
            reference,
            channels,
            step,
            columns,
            spectra: Vec::new(),
        };
        let mut values = Vec::with_capacity(depth);
        let mut spectrum =
            |[x, y]: [usize; 2]| {
                values.clear();
                values.extend((0..depth).map(|z| (z, level.texel([x, y, z]))).filter_map(
                    |(z, v)| {
                        let v = normaliser.channel(v, z);
                        (v.is_finite() && v != 0.0).then_some(v)
                    },
                ));
                median_mad(&mut values)
            };
        // A block's first spectrum with a noise, from its corner, middle and far edge, since at the edge of a
        // primary beam the corner can be blank where the rest isn't.
        let mut offsets = vec![0, step / 2, step - 1];
        offsets.dedup();
        let sampled: Vec<Option<Noise>> = (0..rows * columns)
            .map(|i| {
                let [x0, y0] = [i % columns * step, i / columns * step];
                (offsets.iter())
                    .flat_map(|&dy| offsets.iter().map(move |&dx| [x0 + dx, y0 + dy]))
                    .filter(|&[x, y]| x < width && y < height)
                    .find_map(&mut spectrum)
            })
            .collect();
        // Each σ the mean of the 3 x 3 blocks around it, and a block with no spectrum of its own takes its
        // neighbours' median too.
        normaliser.spectra = (0..rows * columns)
            .map(|i| {
                let [cx, cy] = [(i % columns) as isize, (i / columns) as isize];
                let near: Vec<Noise> = (-1..=1)
                    .flat_map(|dy| (-1..=1).map(move |dx| [cx + dx, cy + dy]))
                    .filter(|&[x, y]| {
                        (0..columns as isize).contains(&x) && (0..rows as isize).contains(&y)
                    })
                    .filter_map(|[x, y]| sampled[x as usize + columns * y as usize])
                    .collect();
                let mean =
                    |of: fn(&Noise) -> f32| near.iter().map(of).sum::<f32>() / near.len() as f32;
                (!near.is_empty()).then(|| Noise {
                    median: sampled[i].map_or_else(|| mean(|n| n.median), |n| n.median),
                    sigma: mean(|n| n.sigma),
                })
            })
            .collect();
        Some(normaliser)
    }

    fn rescale(&self, v: f32, local: Option<Noise>) -> f32 {
        match local {
            Some(local) if v.is_finite() && v != 0.0 => {
                self.reference.median + (v - local.median) * (self.reference.sigma / local.sigma)
            }
            _ => v,
        }
    }

    fn channel(&self, v: f32, z: usize) -> f32 {
        self.rescale(v, self.channels.get(z).copied().flatten())
    }

    /// Texel `p`'s value `v`, normalised.
    pub fn apply(&self, v: f32, [x, y, z]: [usize; 3]) -> f32 {
        let spectrum = self
            .spectra
            .get(x / self.step + self.columns * (y / self.step));
        self.rescale(self.channel(v, z), spectrum.copied().flatten())
    }
}

/// A level as shown: normalised and smoothed where asked.
pub struct Shown<'a> {
    pub level: LevelOf<'a>,
    pub processing: Processing,
    pub normaliser: Option<&'a Normaliser>,
}

impl Shown<'_> {
    fn normalised(&self, p: [usize; 3]) -> f32 {
        let v = self.level.texel(p);
        match self.normaliser.filter(|_| self.processing.normalise) {
            Some(normaliser) => normaliser.apply(v, p),
            None => v,
        }
    }

    /// Texel `p` as `smooth` leaves it, its passes along z, x and then y worked out for it alone.
    fn texel(&self, [x, y, z]: [usize; 3]) -> f32 {
        let at = |[x, y, z]: [isize; 3]| match x >= 0 && y >= 0 && z >= 0 {
            true => self.normalised([x as usize, y as usize, z as usize]),
            false => f32::NAN,
        };
        let along_z = |[x, y, z]: [isize; 3]| taps_at(&HANNING, |dz| at([x, y, z + dz]));
        let p = [x as isize, y as isize, z as isize];
        match self.processing.smoothing {
            Smoothing::Off => self.normalised([x, y, z]),
            Smoothing::Spectral => along_z(p),
            Smoothing::Spatial => {
                let along_x =
                    |[x, y, z]: [isize; 3]| taps_at(&BINOMIAL_XY, |dx| along_z([x + dx, y, z]));
                taps_at(&BINOMIAL_XY, |dy| along_x([p[0], p[1] + dy, p[2]]))
            }
        }
    }

    /// The noise of `texels`, from the texels `noise` samples.
    pub fn measure(&self, texels: Region) -> Option<Noise> {
        sampled_noise(texels.extent(), |p| {
            self.texel(std::array::from_fn(|i| texels.min[i] + p[i]))
        })
    }

    /// The texels of `texels` as a block, with the texels `smoothing` takes in around them as far as the level goes.
    fn block(&self, texels: Region) -> (Vec<f32>, Region) {
        let halo = match self.processing.smoothing {
            Smoothing::Off => [0; 3],
            Smoothing::Spectral => [0, 0, 1],
            Smoothing::Spatial => [2, 2, 1],
        };
        let grown = grow(texels, self.level.dims, halo);
        let mut block = self.level.block(grown);
        let dims = grown.extent();
        if let Some(normaliser) = self.normaliser.filter(|_| self.processing.normalise) {
            for (i, v) in block.iter_mut().enumerate() {
                let p = [i % dims[0], i / dims[0] % dims[1], i / (dims[0] * dims[1])];
                *v = normaliser.apply(*v, std::array::from_fn(|a| grown.min[a] + p[a]));
            }
        }
        if self.processing.smoothing != Smoothing::Off {
            block = smooth(
                &block,
                dims,
                self.processing.smoothing == Smoothing::Spatial,
            );
        }
        (block, grown)
    }

    /// The cubelet at `at`, with its cell, which takes in a texel past its faces.
    pub fn cubelet(&self, at: [usize; 3]) -> Option<Cubelet> {
        let (dims, f) = (self.level.cube.info.dims, self.level.factors);
        let voxels = cubelet_region(dims, f, at)?;
        let level = self.level.dims;
        let texels = Region {
            min: std::array::from_fn(|i| voxels.min[i] / f[i]),
            max: std::array::from_fn(|i| voxels.max[i] / f[i]),
        };
        let (block, grown) = self.block(grow(texels, level, [1; 3]));
        let block_dims = grown.extent();
        let inner = Region {
            min: std::array::from_fn(|i| texels.min[i] - grown.min[i]),
            max: std::array::from_fn(|i| texels.max[i] - grown.min[i]),
        };
        let cell = Cell::of(&block, block_dims, grow(inner, block_dims, [1; 3]));
        Some(Cubelet::new(
            texels.extent(),
            copy(&block, block_dims, inner),
            cell,
        ))
    }

    /// The mask by `find_sources` at `threshold` over `texels` and the texels around them that smoothing takes in,
    /// 1 inside, and the region it's over.
    pub fn mask(&self, texels: Region, threshold: f32) -> (Vec<u8>, Region) {
        let (block, grown) = self.block(texels);
        let dims = grown.extent();
        let mask = match noise(&block, dims) {
            Some(noise) => find_sources(&block, dims, noise, threshold),
            None => vec![0; block.len()],
        };
        (mask, grown)
    }
}

/// Hanning smoothing along z, (1, 2, 1) / 4, and with `spatial` the same along x and y with the taps two voxels
/// apart: the 3D binomial. NaN is left out and stays NaN.
pub fn smooth(voxels: &[f32], dims: [usize; 3], spatial: bool) -> Vec<f32> {
    let mut out = vec![0f32; voxels.len()];
    smooth_z(voxels, &mut out, dims, &HANNING);
    if spatial {
        smooth_planes(&mut out, dims, &BINOMIAL_XY, &BINOMIAL_XY);
    }
    out
}

/// SoFiA-2's smooth-and-clip finder without its reliability step: the voxels smoothed with each pair of box kernels
/// in SPATIAL and SPECTRAL, each clipped at `threshold` times its own noise (the unsmoothed at `noise`), and OR'd.
/// Then `drop_small` drops the small components. 1 marks a voxel in the mask.
pub fn find_sources(voxels: &[f32], dims: [usize; 3], noise: Noise, threshold: f32) -> Vec<u8> {
    let mut mask = vec![0u8; voxels.len()];
    let mut smoothed = Vec::new();
    for spatial in SPATIAL {
        for spectral in SPECTRAL {
            let (data, noise) = if (spatial, spectral) == (1, 1) {
                (voxels, Some(noise))
            } else {
                smoothed.resize(voxels.len(), 0.0);
                smooth_z(voxels, &mut smoothed, dims, &box_taps(spectral));
                let taps = box_taps(spatial);
                smooth_planes(&mut smoothed, dims, &taps, &taps);
                (&smoothed[..], self::noise(&smoothed, dims))
            };
            let Some(noise) = noise else {
                continue;
            };
            let cut = noise.median + threshold * noise.sigma;
            for (m, &v) in mask.iter_mut().zip(data) {
                *m |= u8::from(v > cut);
            }
        }
    }
    drop_small(&mut mask, dims);
    mask
}

/// Taps along an axis: (offset, weight).
type Taps = [(isize, f32)];

fn box_taps(width: usize) -> Vec<(isize, f32)> {
    let half = (width / 2) as isize;
    (-half..=half).map(|offset| (offset, 1.0)).collect()
}

/// `tap_mean` of `value` at the taps' offsets, NaN past the ends.
fn taps_at(taps: &Taps, value: impl Fn(isize) -> f32) -> f32 {
    let (mut sum, mut weight) = (0.0, 0.0);
    for &(offset, w) in taps {
        let v = value(offset);
        if v.is_nan() {
            if offset == 0 {
                return v;
            }
        } else {
            sum += w * v;
            weight += w;
        }
    }
    sum / weight
}

/// The weighted mean of the taps around voxel `i` of a line of `n` voxels `stride` apart, leaving out NaN and voxels
/// past the ends. NaN stays NaN.
fn tap_mean(line: &[f32], n: usize, stride: usize, i: usize, taps: &Taps) -> f32 {
    let centre = line[i * stride];
    if centre.is_nan() {
        return centre;
    }
    let (mut sum, mut weight) = (0.0, 0.0);
    for &(offset, w) in taps {
        if let Some(j) = i.checked_add_signed(offset).filter(|&j| j < n) {
            let v = line[j * stride];
            if !v.is_nan() {
                sum += w * v;
                weight += w;
            }
        }
    }
    sum / weight
}

/// `out` is `src` smoothed along z, a task per plane.
fn smooth_z(src: &[f32], out: &mut [f32], dims: [usize; 3], taps: &Taps) {
    let (plane, depth) = (dims[0] * dims[1], dims[2]);
    if taps.len() == 1 {
        out.copy_from_slice(src);
        return;
    }
    ComputeTaskPool::get_or_init(TaskPool::default).scope(|s| {
        for (z, out) in out.chunks_mut(plane).enumerate() {
            s.spawn(async move {
                for (j, o) in out.iter_mut().enumerate() {
                    *o = tap_mean(&src[j..], depth, plane, z, taps);
                }
            });
        }
    });
}

/// Each plane smoothed along x with `x` and then along y with `y`, in place, a task per plane.
fn smooth_planes(data: &mut [f32], dims: [usize; 3], x: &Taps, y: &Taps) {
    let [width, height, _] = dims;
    ComputeTaskPool::get_or_init(TaskPool::default).scope(|s| {
        for plane in data.chunks_mut(width * height) {
            s.spawn(async move {
                let mut copy = plane.to_vec();
                // Rows of `width` voxels one apart, then columns of `height` voxels a row apart.
                for (taps, n, stride, lines, next) in
                    [(x, width, 1, height, width), (y, height, width, width, 1)]
                {
                    if taps.len() == 1 {
                        continue;
                    }
                    copy.copy_from_slice(plane);
                    for start in (0..lines).map(|line| line * next) {
                        for i in 0..n {
                            plane[start + i * stride] =
                                tap_mean(&copy[start..], n, stride, i, taps);
                        }
                    }
                }
            });
        }
    });
}

/// Drops the mask's 26-connected components spanning fewer than MIN_EXTENT voxels along any axis that long. It works
/// on runs of set voxels along x, so it needs little memory unless the mask is mostly noise.
fn drop_small(mask: &mut [u8], dims: [usize; 3]) {
    let [width, height, depth] = dims;
    if dims.iter().any(|&d| d > usize::from(u16::MAX)) {
        return;
    }
    // Each row's runs as [start, end) in x, a row's from first[row] on.
    let (mut runs, mut first) = (
        Vec::<[u16; 2]>::new(),
        Vec::with_capacity(height * depth + 1),
    );
    for row in mask.chunks(width) {
        first.push(runs.len());
        let mut x = 0;
        while x < width {
            if row[x] == 0 {
                x += 1;
                continue;
            }
            let start = x;
            while x < width && row[x] != 0 {
                x += 1;
            }
            runs.push([start as u16, x as u16]);
        }
        if runs.len() > MAX_RUNS {
            return;
        }
    }
    first.push(runs.len());

    let mut parent: Vec<u32> = (0..runs.len() as u32).collect();
    for z in 0..depth {
        for y in 0..height {
            let row = y + height * z;
            // The rows before this one that can touch it: the one above, and three in the plane before.
            let before = [
                (y > 0).then(|| row - 1),
                (z > 0 && y > 0).then(|| row - height - 1),
                (z > 0).then(|| row - height),
                (z > 0 && y + 1 < height).then(|| row - height + 1),
            ];
            for other in before.into_iter().flatten() {
                // Both rows' runs in step, joining those within a voxel of each other along x.
                let (mut i, mut j) = (first[row], first[other]);
                while i < first[row + 1] && j < first[other + 1] {
                    let ([a0, a1], [b0, b1]) = (runs[i], runs[j]);
                    if a0 <= b1 && b0 <= a1 {
                        union(&mut parent, i, j);
                    }
                    if a1 < b1 { i += 1 } else { j += 1 }
                }
            }
        }
    }

    // Each component's lowest and highest voxel along each axis, kept at its root.
    let mut extent = vec![[u16::MAX, u16::MAX, u16::MAX, 0, 0, 0]; runs.len()];
    let rows = first.windows(2).enumerate();
    for (row, span) in rows.clone() {
        let at = [0, (row % height) as u16, (row / height) as u16];
        for i in span[0]..span[1] {
            let e = &mut extent[find(&mut parent, i)];
            let [start, end] = runs[i];
            for (axis, (lo, hi)) in [(start, end - 1), (at[1], at[1]), (at[2], at[2])]
                .into_iter()
                .enumerate()
            {
                e[axis] = e[axis].min(lo);
                e[axis + 3] = e[axis + 3].max(hi);
            }
        }
    }
    // No more than the mask spans, so that a plane or a few channels keep what's in them.
    let min_extent = dims.map(|d| MIN_EXTENT.min(d as u16));
    for (row, span) in rows {
        for i in span[0]..span[1] {
            let e = extent[find(&mut parent, i)];
            if (0..3).any(|axis| e[axis + 3] - e[axis] + 1 < min_extent[axis]) {
                let [start, end] = runs[i].map(usize::from);
                mask[row * width + start..row * width + end].fill(0);
            }
        }
    }
}

fn find(parent: &mut [u32], mut i: usize) -> usize {
    while parent[i] as usize != i {
        parent[i] = parent[parent[i] as usize];
        i = parent[i] as usize;
    }
    i
}

fn union(parent: &mut [u32], a: usize, b: usize) {
    let (a, b) = (find(parent, a), find(parent, b));
    if a != b {
        parent[a.max(b)] = a.min(b) as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_finder_keeps_faint_extended_emission_and_drops_a_lone_spike() {
        // Noise from sums of uniforms, a cloud at 1.5σ a voxel that only the smoothed copies can find, and a 6σ
        // spike that only the unsmoothed one does.
        let dims = [40, 40, 40];
        let mut seed = 7u32;
        let mut uniform = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1 << 24) as f32 - 0.5
        };
        let at = |[x, y, z]: [usize; 3]| x + 40 * (y + 40 * z);
        let in_cloud = |i: usize| {
            [i % 40, i / 40 % 40, i / 1600]
                .iter()
                .all(|c| (14..26).contains(c))
        };
        let mut data: Vec<f32> = (0..64_000)
            .map(|i| (0..12).map(|_| uniform()).sum::<f32>() + if in_cloud(i) { 1.5 } else { 0.0 })
            .collect();
        data[at([5, 5, 5])] = 6.0;
        let noise = noise(&data, dims).unwrap();
        assert!((noise.sigma - 1.0).abs() < 0.1, "{noise:?}");

        let mask = find_sources(&data, dims, noise, 5.0);
        assert_eq!(mask[at([20, 20, 20])], 1);
        assert_eq!(mask[at([5, 5, 5])], 0);
        // The smoothing spreads the cloud by a few voxels, and nothing else is found.
        let near_cloud = |i: usize| {
            [i % 40, i / 40 % 40, i / 1600]
                .iter()
                .all(|c| (9..31).contains(c))
        };
        let stray = (0..64_000)
            .filter(|&i| mask[i] == 1 && !near_cloud(i))
            .count();
        assert_eq!(stray, 0);

        // A plane, and two channels, are shorter along z than MIN_EXTENT but keep a 3σ square that only the
        // smoothed copies find.
        for depth in [1, 2] {
            let dims = [40, 40, depth];
            let square = |i: usize| [i % 40, i / 40 % 40].iter().all(|c| (14..26).contains(c));
            let plane: Vec<f32> = (0..1600 * depth)
                .map(|i| data[at([0, 0, 30]) + i] + if square(i) { 3.0 } else { 0.0 })
                .collect();
            let mask = find_sources(&plane, dims, super::noise(&plane, dims).unwrap(), 5.0);
            assert_eq!(mask[at([20, 20, 0])], 1, "{depth} channels");
        }
    }

    #[test]
    fn the_normaliser_finds_a_blocks_spectrum_past_a_blank_corner_or_takes_its_neighbours() {
        // Blocks of 4 x 4 spectra. σ is 1 for x < 256 and 2 from there, where the blocks' corners are blank, and
        // from x = 384 only every fourth spectrum, one that no block samples, is left.
        let dims = [512, 8, 40];
        let mut seed = 11u32;
        let mut uniform = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1 << 24) as f32 - 0.5
        };
        let data: Vec<f32> = (0..dims.iter().product::<usize>())
            .map(|i| {
                let x = i % 512;
                let noise = (0..12).map(|_| uniform()).sum::<f32>();
                match x {
                    256.. if x % 4 == 0 || (x >= 384 && x % 4 != 1) => f32::NAN,
                    256.. => 2.0 * noise,
                    _ => noise,
                }
            })
            .collect();
        let cube = crate::tests::cube(dims, data);
        let normaliser = Normaliser::measure(&LevelOf::new(&cube, [1; 3], true, None)).unwrap();
        let sigma = |columns: &mut dyn Iterator<Item = usize>| {
            let mut values: Vec<f32> = columns
                .flat_map(|x| (0..8 * 40).map(move |j| [x, j % 8, j / 8]))
                .map(|p| normaliser.apply(cube.data[index(dims, p)], p))
                .collect();
            median_mad(&mut values).unwrap().sigma
        };
        let reference = normaliser.reference.sigma;
        // Away from where the σ changes, which the 3 x 3 mean blurs.
        let past_corners = sigma(&mut (272..368).filter(|x| x % 4 != 0));
        assert!(
            (past_corners / reference - 1.0).abs() < 0.1,
            "{past_corners} for {reference}"
        );
        // The first block with nothing it samples takes its neighbours'.
        let from_neighbours = sigma(&mut [385].into_iter());
        assert!(
            (from_neighbours / reference - 1.0).abs() < 0.25,
            "{from_neighbours} for {reference}"
        );
    }
}
