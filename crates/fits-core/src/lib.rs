use std::io::{self, ErrorKind};
use std::str::FromStr;

use bevy_tasks::{ComputeTaskPool, TaskPool};
#[cfg(not(target_arch = "wasm32"))]
use bevy_tasks::{ParallelSlice, ParallelSliceMut};

mod cubelet;
pub use cubelet::{CUBELET, Cubelet, Level, Levels, Voxels, cubelet, levels};

const BLOCK: usize = 2880;
/// Voxels read, decoded and summarised per task: 4 MB of output, small enough to stay in cache.
const CHUNK_VOXELS: usize = 1 << 20;

/// Bytes that can be read at any offset, so the data can be read in chunks on several threads.
pub trait Source: MaybeSync {
    fn size(&self) -> io::Result<u64>;
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()>;
}

/// Sync where the chunks are read on several threads; wasm reads them in turn, so a JS array will do.
#[cfg(not(target_arch = "wasm32"))]
pub trait MaybeSync: Sync {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Sync + ?Sized> MaybeSync for T {}
#[cfg(target_arch = "wasm32")]
pub trait MaybeSync {}
#[cfg(target_arch = "wasm32")]
impl<T: ?Sized> MaybeSync for T {}

impl Source for [u8] {
    fn size(&self) -> io::Result<u64> {
        Ok(self.len() as u64)
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let start = usize::try_from(offset).map_err(|_| ErrorKind::UnexpectedEof)?;
        let bytes = self
            .get(start..start + buf.len())
            .ok_or(ErrorKind::UnexpectedEof)?;
        buf.copy_from_slice(bytes);
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Source for std::fs::File {
    fn size(&self) -> io::Result<u64> {
        Ok(self.metadata()?.len())
    }

    #[cfg(unix)]
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        std::os::unix::fs::FileExt::read_exact_at(self, buf, offset)
    }

    #[cfg(windows)]
    fn read_at(&self, mut offset: u64, mut buf: &mut [u8]) -> io::Result<()> {
        while !buf.is_empty() {
            match std::os::windows::fs::FileExt::seek_read(self, buf, offset)? {
                0 => return Err(ErrorKind::UnexpectedEof.into()),
                n => (buf, offset) = (&mut buf[n..], offset + n as u64),
            }
        }
        Ok(())
    }
}

/// 0-based inclusive voxel corners.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Region {
    pub min: [usize; 3],
    pub max: [usize; 3],
}

impl Region {
    pub fn whole(dims: [usize; 3]) -> Self {
        Self {
            min: [0; 3],
            max: dims.map(|d| d - 1),
        }
    }

    pub fn extent(&self) -> [usize; 3] {
        std::array::from_fn(|i| self.max[i] - self.min[i] + 1)
    }
}

pub struct Cube {
    pub info: FileInfo,
    pub data: Vec<f32>,
}

impl Cube {
    /// The noise σ: from the median absolute deviation of the quietest channels, sampled on a grid, leaving out
    /// channels where it's zero, such as zero-filled or flagged ones. Where every channel is like that, the
    /// histogram's IQR / 1.349, as for heavily quantised data. None when the cube has no finite voxels.
    pub fn sigma(&self) -> Option<f64> {
        // About 64 x 64 voxels a channel, stepping x and y apart so that the step can't alias with the width.
        const GRID: usize = 64;
        let [width, height, _] = self.info.dims;
        let (step_x, step_y) = (width.div_ceil(GRID), height.div_ceil(GRID));
        let mut mads: Vec<f32> = (self.data.chunks(width * height))
            .filter_map(|channel| {
                let mut values: Vec<f32> = (channel.chunks(width).step_by(step_y))
                    .flat_map(|row| row.iter().step_by(step_x))
                    .copied()
                    .filter(|v| v.is_finite())
                    .collect();
                let centre = median(&mut values)?;
                values.iter_mut().for_each(|v| *v = (*v - centre).abs());
                median(&mut values).filter(|&mad| mad > 0.0)
            })
            .collect();
        mads.sort_by(f32::total_cmp);
        // The middle of the quietest quarter, so that neither emission nor a few unusually flat channels set it.
        if let Some(&mad) = mads.get(mads.len().div_ceil(4) / 2) {
            return Some(1.4826 * f64::from(mad));
        }
        let Stats { min, max, .. } = self.info.stats;
        let range = f64::from(max) - f64::from(min);
        range.is_finite().then(|| {
            let (lo, hi) = self.info.histogram.clip(50.0);
            f64::from(hi - lo) * range / 1.349
        })
    }
}

fn median(values: &mut [f32]) -> Option<f32> {
    let middle = values.len() / 2;
    (!values.is_empty()).then(|| *values.select_nth_unstable_by(middle, f32::total_cmp).1)
}

/// Everything about a cube but its voxels, which is what a viewer is sent when it opens one.
#[derive(Clone, Debug)]
pub struct FileInfo {
    pub dims: [usize; 3],
    /// The FITS axes (0-based) shown as x, y and z. One past NAXIS pads a 2D image.
    pub axes: [usize; 3],
    pub cards: Vec<String>,
    pub hdu: usize,
    pub stats: Stats,
    pub histogram: Histogram,
}

impl FileInfo {
    /// The data unit from BUNIT, e.g. "Jy/beam" (or CASA's "beam-1 Jy").
    pub fn unit(&self) -> Option<String> {
        Some(value(&self.cards, "BUNIT")?.trim_start().to_string()).filter(|u| !u.is_empty())
    }

    /// "1 of 3 along STOKES" when the HDU holds more than the one plane of its other axes that's shown.
    pub fn other_planes(&self) -> Option<String> {
        let naxis: usize = num(&self.cards, "NAXIS").ok()??;
        let others: Vec<(u64, String)> = (1..=naxis)
            .filter(|a| !self.axes.contains(&(a - 1)))
            .map(|a| {
                let n = num(&self.cards, &format!("NAXIS{a}")).ok().flatten();
                let name = value(&self.cards, &format!("CTYPE{a}"));
                (n.unwrap_or(1), name.unwrap_or(format!("axis {a}")))
            })
            .filter(|(n, _)| *n > 1)
            .collect();
        let planes = others
            .iter()
            .try_fold(1u64, |p, (n, _)| p.checked_mul(*n))?;
        let names: Vec<&str> = others.iter().map(|(_, name)| name.as_str()).collect();
        (planes > 1).then(|| format!("1 of {planes} along {}", names.join(", ")))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub min: f32,
    pub max: f32,
    pub mean: f64,
    pub std: f64,
    /// NaN and infinite voxels, which the other statistics leave out.
    pub non_finite: usize,
}

pub fn read_fits<S: Source + ?Sized>(source: &S, hdu: Option<usize>) -> Result<Cube, String> {
    let size = source.size().map_err(|e| e.to_string())?;
    let (mut index, mut offset) = (0, 0);
    let (mut primary, mut compressed): (Vec<String>, _) = (Vec::new(), None);
    let tile_compressed = |n| {
        format!(
            "HDU {n} is a tile-compressed image, which isn't supported; decompress it with funpack"
        )
    };
    loop {
        let Some(mut cards) = read_header(source, &mut offset)? else {
            return Err(match (hdu, compressed) {
                (Some(n), _) => format!("HDU {n} not found, the file has {index}"),
                (None, Some(n)) => tile_compressed(n),
                (None, None) => "no image HDU with NAXIS >= 2 found".into(),
            });
        };
        let bitpix: i64 = num(&cards, "BITPIX")?.ok_or("missing BITPIX")?;
        let naxis: u64 = num(&cards, "NAXIS")?.ok_or("missing NAXIS")?;
        let axes = (1..=naxis)
            .map(|i| num::<u64>(&cards, &format!("NAXIS{i}"))?.ok_or(format!("missing NAXIS{i}")))
            .collect::<Result<Vec<_>, _>>()?;
        if ![8, 16, 32, 64, -32, -64].contains(&bitpix) {
            return Err(format!("HDU {index}: unsupported BITPIX {bitpix}"));
        }
        let width = bitpix.unsigned_abs() / 8;
        // A random-groups primary (UVFITS) has NAXIS1 = 0 and is no image.
        let groups = value(&cards, "GROUPS").as_deref() == Some("T");
        let tiled = value(&cards, "ZIMAGE").as_deref() == Some("T");
        compressed = compressed.or(tiled.then_some(index));
        let is_cube = naxis >= 2
            && axes.iter().all(|&n| n > 0)
            && !groups
            && (index == 0 || value(&cards, "XTENSION").as_deref() == Some("IMAGE"));

        if hdu == Some(index) || (hdu.is_none() && is_cube) {
            if !is_cube {
                return Err(match tiled {
                    true => tile_compressed(index),
                    false => format!("HDU {index} is not an image with two or more non-empty axes"),
                });
            }
            let picked = pick_axes(&axes);
            let lengths = picked.map(|a| axes.get(a).copied().unwrap_or(1));
            let len = checked_product(&lengths)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or(format!("HDU {index}: image too large"))?;
            // Each axis is at most `len`, which fits.
            let dims = lengths.map(|n| n as usize);
            // Checked before allocating, so a header can't claim more memory than its file backs.
            if (len as u64)
                .checked_mul(width)
                .and_then(|bytes| bytes.checked_add(offset))
                .is_none_or(|end| end > size)
            {
                return Err(format!("HDU {index}: data truncated"));
            }
            // The keyword inheritance convention. The extension's cards come first, so its values win.
            if index > 0 && value(&cards, "INHERIT").as_deref() == Some("T") {
                let structural = ["SIMPLE", "BITPIX", "EXTEND", "GROUPS", "PCOUNT", "GCOUNT"];
                let commentary = ["COMMENT", "HISTORY", ""];
                cards.extend(primary.into_iter().filter(|c| {
                    let key = c[..8].trim_end();
                    !key.starts_with("NAXIS")
                        && !structural.contains(&key)
                        && !commentary.contains(&key)
                }));
            }
            let (data, stats, histogram) =
                read_data(source, offset, &cards, bitpix, width as usize, len)?;
            return Ok(Cube {
                info: FileInfo {
                    dims,
                    axes: picked,
                    cards,
                    hdu: index,
                    stats,
                    histogram,
                },
                data,
            });
        }

        let pcount: u64 = num(&cards, "PCOUNT")?.unwrap_or(0);
        let gcount: u64 = num(&cards, "GCOUNT")?.unwrap_or(1);
        // Checked, so a crafted size can't wrap the offset back to an earlier header and loop forever.
        let padded_bytes = |values: &[u64]| {
            (checked_product(values)?.checked_add(pcount)?)
                .checked_mul(gcount)?
                .checked_mul(width)?
                .checked_next_multiple_of(BLOCK as u64)
        };
        let padded = match &axes[..] {
            [] => Some(0),
            [_, rest @ ..] if groups => padded_bytes(rest),
            all => padded_bytes(all),
        };
        offset = padded
            .and_then(|padded| offset.checked_add(padded))
            .filter(|&end| end <= size)
            .ok_or(format!("HDU {index}: data truncated"))?;
        if index == 0 {
            primary = cards;
        }
        index += 1;
    }
}

/// The first three axes longer than 1, then length-1 axes, then ones past NAXIS. Leaving out length-1 axes
/// keeps the first x * y * z values in order, so CASA's RA, DEC, STOKES(1), FREQ loads every channel.
fn pick_axes(lengths: &[u64]) -> [usize; 3] {
    let n = lengths.len();
    let mut picked = (0..n)
        .filter(|&a| lengths[a] > 1)
        .chain((0..n).filter(|&a| lengths[a] == 1))
        .chain(n..);
    std::array::from_fn(|_| picked.next().unwrap())
}

fn checked_product(values: &[u64]) -> Option<u64> {
    values.iter().try_fold(1u64, |n, &v| n.checked_mul(v))
}

/// Reads the header starting at `offset` and moves it past the header. Returns None at a clean end of file.
fn read_header<S: Source + ?Sized>(
    source: &S,
    offset: &mut u64,
) -> Result<Option<Vec<String>>, String> {
    let mut cards = Vec::new();
    let mut block = [0u8; BLOCK];
    loop {
        match source.read_at(*offset, &mut block) {
            Err(e) if e.kind() == ErrorKind::UnexpectedEof && cards.is_empty() => return Ok(None),
            r => r.map_err(|e| format!("reading FITS header: {e}"))?,
        }
        *offset += BLOCK as u64;
        if cards.is_empty() && !block.starts_with(b"SIMPLE  =") && !block.starts_with(b"XTENSION=")
        {
            return Err("not an uncompressed FITS file".into());
        }
        for card in block.chunks(80) {
            if card.starts_with(b"END     ") {
                return Ok(Some(cards));
            }
            cards.push(
                card.iter()
                    .map(|&b| if b.is_ascii() { b as char } else { '?' })
                    .collect(),
            );
        }
    }
}

/// A card's value: a string without its quotes and trailing spaces, joined across the CONTINUE cards of the
/// long-string convention, or the text before any comment. None when the value field is empty.
fn value(cards: &[String], key: &str) -> Option<String> {
    let at = cards
        .iter()
        .position(|c| c[..8].trim_end() == key && &c[8..10] == "= ")?;
    let field = cards[at][10..].trim_start();
    let Some(quoted) = field.strip_prefix('\'') else {
        return Some(field.split('/').next()?.trim().to_string()).filter(|v| !v.is_empty());
    };
    let mut text = string(quoted)?;
    let mut next = cards[at + 1..].iter();
    while text.ends_with('&') {
        let Some(more) = next
            .next()
            .filter(|c| c.starts_with("CONTINUE"))
            .and_then(|c| string(c[8..].trim_start().strip_prefix('\'')?))
        else {
            break;
        };
        text.pop();
        text.push_str(&more);
    }
    Some(text.trim_end().to_string())
}

/// The string after an opening quote, up to the closing one, with '' read as a quote. None if unterminated.
fn string(after_quote: &str) -> Option<String> {
    let mut text = String::new();
    let mut chars = after_quote.chars();
    while let Some(c) = chars.next() {
        if c == '\'' && chars.next() != Some('\'') {
            return Some(text);
        }
        text.push(c);
    }
    None
}

fn num<T: FromStr>(cards: &[String], key: &str) -> Result<Option<T>, String> {
    value(cards, key)
        .map(|v| {
            v.replace(['D', 'd'], "E")
                .parse()
                .map_err(|_| format!("invalid {key} = {v}"))
        })
        .transpose()
}

/// Reads, decodes and summarises the data a chunk at a time, on all cores natively.
fn read_data<S: Source + ?Sized>(
    source: &S,
    offset: u64,
    cards: &[String],
    bitpix: i64,
    width: usize,
    len: usize,
) -> Result<(Vec<f32>, Stats, Histogram), String> {
    let bscale: f64 = num(cards, "BSCALE")?.unwrap_or(1.0);
    let bzero: f64 = num(cards, "BZERO")?.unwrap_or(0.0);
    // BLANK only applies to integer data. A real with no fraction (-32768.0) is taken too.
    let blank: Option<i64> = if bitpix > 0 {
        num(cards, "BLANK").or_else(|e| match num::<f64>(cards, "BLANK")? {
            Some(b) if b.fract() == 0.0 => Ok(Some(b as i64)),
            _ => Err(e),
        })?
    } else {
        None
    };
    let int = |raw: i64| {
        if Some(raw) == blank {
            f32::NAN
        } else {
            (bzero + bscale * raw as f64) as f32
        }
    };
    let float = |raw: f64| (bzero + bscale * raw) as f32;

    // Natively the OS zeroes these pages lazily, so they're first touched by the parallel tasks rather than
    // by one thread up front. wasm32 has no overcommit, so there a failed allocation is reported instead.
    #[cfg(not(target_arch = "wasm32"))]
    let mut data = vec![0.0; len];
    #[cfg(target_arch = "wasm32")]
    let mut data = {
        let mut data = Vec::new();
        data.try_reserve_exact(len)
            .map_err(|_| format!("cannot allocate {len} voxels"))?;
        data.resize(len, 0.0);
        data
    };
    let decode = |chunk: usize, out: &mut [f32]| -> Result<Moments, String> {
        let read = |buf: &mut [u8]| {
            source
                .read_at(offset + (chunk * CHUNK_VOXELS * width) as u64, buf)
                .map_err(|e| format!("reading FITS data: {e}"))
        };
        // 32-bit values are read straight into the output and converted in place.
        if width == 4 {
            read(bytemuck::cast_slice_mut(out))?;
            for v in out.iter_mut() {
                let bytes = v.to_ne_bytes();
                *v = match bitpix {
                    32 => int(i32::from_be_bytes(bytes).into()),
                    _ => float(f32::from_be_bytes(bytes).into()),
                };
            }
            return Ok(Moments::of(out));
        }
        let mut raw = vec![0u8; out.len() * width];
        read(&mut raw)?;
        match bitpix {
            8 => convert(out, &raw, |b: [u8; 1]| int(b[0].into())),
            16 => convert(out, &raw, |b| int(i16::from_be_bytes(b).into())),
            64 => convert(out, &raw, |b| int(i64::from_be_bytes(b))),
            _ => convert(out, &raw, |b| float(f64::from_be_bytes(b))),
        }
        Ok(Moments::of(out))
    };
    // A pool of its own with a thread per core, since Bevy's compute pool only gets some of them.
    #[cfg(not(target_arch = "wasm32"))]
    let pool = bevy_tasks::TaskPoolBuilder::new()
        .num_threads(std::thread::available_parallelism().map_or(4, |n| n.get()))
        .build();
    #[cfg(not(target_arch = "wasm32"))]
    let chunks = data.par_chunk_map_mut(&pool, CHUNK_VOXELS, decode);
    #[cfg(target_arch = "wasm32")]
    let chunks: Vec<_> = data
        .chunks_mut(CHUNK_VOXELS)
        .enumerate()
        .map(|(i, out)| decode(i, out))
        .collect();
    let moments = chunks.into_iter().try_fold(Moments::EMPTY, |all, chunk| {
        Ok::<_, String>(all.merge(chunk?))
    })?;
    let stats = moments.stats(len);

    // The range is only known once every chunk is in, so the histogram takes a second pass.
    let bins = ((moments.n as f64).sqrt().round() as usize).clamp(1, MAX_BINS);
    let count = |values: &[f32]| count(values, stats.min, stats.max, bins);
    // A few parts per thread, so the faster cores pick up the slack.
    #[cfg(not(target_arch = "wasm32"))]
    let parts = data.par_chunk_map(&pool, len.div_ceil(4 * pool.thread_num()), |_, part| {
        count(part)
    });
    #[cfg(target_arch = "wasm32")]
    let parts = [count(&data)];
    let mut counts = vec![0; bins];
    for part in parts {
        counts.iter_mut().zip(part).for_each(|(c, n)| *c += n);
    }
    let cumulative = counts.iter().scan(0, |below, &n| {
        *below += n;
        Some(*below)
    });
    Ok((data, stats, Histogram(cumulative.collect())))
}

/// At most u16::MAX bins, so a u16 slot can't overrun a table and its last slot takes the non-finite voxels.
const MAX_BINS: usize = u16::MAX as usize;

/// Counts `values` in `bins` equal bins from `min` to `max`. Four tables take the voxels in turn, so a run of
/// equal values (zeros, or the non-finite voxels in the last slot) doesn't wait on its own increments.
fn count(values: &[f32], min: f32, max: f32, bins: usize) -> Vec<u64> {
    const LANES: usize = 16;
    let mut storage = vec![0u32; 4 << 16];
    let tables: &mut [[u32; 1 << 16]; 4] = storage.as_chunks_mut().0.try_into().unwrap();
    let range = f64::from(max) - f64::from(min);
    let scale = if range > 0.0 {
        (bins as f64 / range) as f32
    } else {
        0.0
    };
    let top = bins as u32 - 1;
    let slot = |v: f32| match v.is_finite() {
        true => (((v - min) * scale) as u32).min(top) as u16,
        false => u16::MAX,
    };
    let (blocks, rest) = values.as_chunks::<LANES>();
    for block in blocks {
        // Worked out for a whole block first, so LLVM can vectorise it.
        let mut slots = [0u16; LANES];
        for i in 0..LANES {
            slots[i] = slot(block[i]);
        }
        for (i, s) in slots.into_iter().enumerate() {
            tables[i % 4][usize::from(s)] += 1;
        }
    }
    for &v in rest {
        tables[0][usize::from(slot(v))] += 1;
    }
    (0..bins)
        .map(|b| tables.iter().map(|t| u64::from(t[b])).sum())
        .collect()
}

/// CARTA's clip presets, for `Histogram::clip`.
pub const PERCENTILES: [f64; 8] = [90.0, 95.0, 99.0, 99.5, 99.9, 99.95, 99.99, 100.0];

/// Cumulative counts of the finite voxels in equal bins from `Stats::min` to `Stats::max`.
#[derive(Clone, Debug)]
pub struct Histogram(Vec<u64>);

impl Histogram {
    /// The fraction of the range below which `rank` (0..=1) of the voxels lie, interpolating linearly within the
    /// bin, as CARTA does.
    fn fraction_below(&self, rank: f64) -> f64 {
        let target = rank * self.0[self.0.len() - 1] as f64;
        let i = self.0.partition_point(|&c| (c as f64) < target);
        let below = if i == 0 { 0 } else { self.0[i - 1] };
        // Only a cube with no finite voxels has an empty bin here.
        let n = (self.0[i] - below).max(1);
        (i as f64 + (target - below as f64) / n as f64) / self.0.len() as f64
    }

    /// Thresholds, as fractions of the range, that keep the middle `percent` of the voxels.
    pub fn clip(&self, percent: f64) -> (f32, f32) {
        let tail = (100.0 - percent) / 200.0;
        let (min, max) = (self.fraction_below(tail), self.fraction_below(1.0 - tail));
        (min as f32, max as f32)
    }
}

fn convert<const N: usize>(out: &mut [f32], raw: &[u8], f: impl Fn([u8; N]) -> f32) {
    for (out, &bytes) in out.iter_mut().zip(raw.as_chunks::<N>().0) {
        *out = f(bytes);
    }
}

/// Count, mean, sum of squared deviations from the mean, and range of the finite values in a chunk.
#[derive(Clone, Copy, Debug)]
struct Moments {
    n: usize,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl Moments {
    const EMPTY: Self = Self {
        n: 0,
        mean: f64::NAN,
        m2: 0.0,
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
    };

    /// Two passes over a chunk that's still in cache, the mean and then the squared deviations from it, since
    /// a running sum of squares loses precision on a large pedestal. Sixteen lanes, filled without branches,
    /// let LLVM vectorise both passes, and they're combined in a fixed order so every run gives the same result.
    fn of(values: &[f32]) -> Self {
        const LANES: usize = 16;
        let (blocks, rest) = values.as_chunks::<LANES>();
        let tail: [f32; LANES] = std::array::from_fn(|i| rest.get(i).copied().unwrap_or(f32::NAN));
        let blocks = || blocks.iter().chain([&tail]);
        // A lane counts a sixteenth of a chunk of CHUNK_VOXELS, so u32 has room to spare.
        let (mut n, mut sum) = ([0u32; LANES], [0f64; LANES]);
        // In f32, which is exact, since comparisons never round.
        let (mut min, mut max) = ([f32::INFINITY; LANES], [f32::NEG_INFINITY; LANES]);
        for block in blocks() {
            for i in 0..LANES {
                let (v, finite) = (block[i], block[i].is_finite());
                n[i] += u32::from(finite);
                sum[i] += if finite { f64::from(v) } else { 0.0 };
                min[i] = if finite { min[i].min(v) } else { min[i] };
                max[i] = if finite { max[i].max(v) } else { max[i] };
            }
        }
        let n = n.iter().map(|&n| n as usize).sum::<usize>();
        let mean = sum.iter().sum::<f64>() / n as f64;
        let mut m2 = [0f64; LANES];
        for block in blocks() {
            for i in 0..LANES {
                let d = f64::from(block[i]) - mean;
                m2[i] += if block[i].is_finite() { d * d } else { 0.0 };
            }
        }
        Self {
            n,
            mean,
            m2: m2.iter().sum(),
            min: f64::from(min.into_iter().fold(f32::INFINITY, f32::min)),
            max: f64::from(max.into_iter().fold(f32::NEG_INFINITY, f32::max)),
        }
    }

    /// Chan et al.'s pairwise update, which only ever adds squared deviations.
    fn merge(self, other: Self) -> Self {
        if self.n == 0 || other.n == 0 {
            return if self.n == 0 { other } else { self };
        }
        let n = self.n + other.n;
        let (na, nb) = (self.n as f64, other.n as f64);
        let delta = other.mean - self.mean;
        Self {
            n,
            mean: self.mean + delta * nb / n as f64,
            m2: self.m2 + other.m2 + delta * delta * na * nb / n as f64,
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    fn stats(self, len: usize) -> Stats {
        Stats {
            min: self.min as f32,
            max: self.max as f32,
            mean: self.mean,
            std: (self.m2 / (self.n as f64 - 1.0)).sqrt(),
            non_finite: len - self.n,
        }
    }
}

pub fn find_downsample_factors(gpu_memory_limit_mb: u64, extent: [usize; 3]) -> [usize; 3] {
    let [x, y, z] = extent.map(|d| d as u64);
    let (mut fx, mut fy, mut fz) = (1, 1, 1);
    // Rounded up, as crop_and_downsample sizes the texture; iDaVIE rounds down here and can ask for 2049.
    while x.div_ceil(fx) > 2048 || y.div_ceil(fy) > 2048 {
        fx += 1;
        fy += 1;
    }
    while z.div_ceil(fz) > 2048 {
        fz += 1;
    }
    // At least one, so that a budget of zero still ends.
    let max_elements = (gpu_memory_limit_mb.saturating_mul(1_000_000) / 4).max(1);
    while x.div_ceil(fx) * y.div_ceil(fy) * z.div_ceil(fz) > max_elements {
        if z / fz > x / fx || z / fz > y / fy {
            fz += 1;
        } else {
            fx += 1;
            fy += 1;
        }
    }
    [fx as usize, fy as usize, fz as usize]
}

/// The dims of `region` downsampled by `factors`, where a block cut short at the far edge still counts.
pub fn downsampled_dims(region: Region, factors: [usize; 3]) -> [usize; 3] {
    let extent = region.extent();
    std::array::from_fn(|i| extent[i].div_ceil(factors[i]))
}

/// Each block of `factors` in `data`, a cube of `dims`, becomes its maximum, ignoring NaN, or its mean, ignoring
/// every non-finite voxel, and NaN when there's nothing else. Factors of one copy the voxels as they are. Returns
/// the new dims and the R32Float texture data, x fastest.
pub fn crop_and_downsample(
    data: &[f32],
    dims: [usize; 3],
    region: Region,
    factors: [usize; 3],
    max_mode: bool,
) -> ([usize; 3], Vec<f32>) {
    let (dims, data, _) = downsample(data, dims, None, region, factors, max_mode, false);
    (dims, data)
}

/// `crop_and_downsample` over any level held in memory. A mean weighs each voxel by `weights`, when given, and
/// with `keep_counts` also returns the weight behind each mean, so that the result can be downsampled again exactly.
fn downsample(
    data: &[f32],
    dims: [usize; 3],
    weights: Option<&[u32]>,
    region: Region,
    factors: [usize; 3],
    max_mode: bool,
    keep_counts: bool,
) -> ([usize; 3], Vec<f32>, Vec<u32>) {
    let Region { min, max } = region;
    assert!(
        (0..3).all(|i| min[i] <= max[i] && max[i] < dims[i] && factors[i] > 0),
        "invalid crop or factors"
    );
    assert!(!(max_mode && keep_counts), "a maximum has no counts");
    let out_dims = downsampled_dims(region, factors);
    let block = |i: usize, o: usize| {
        min[i] + o * factors[i]..(min[i] + (o + 1) * factors[i]).min(max[i] + 1)
    };
    let row = |z: usize, y: usize| {
        let start = (z * dims[1] + y) * dims[0];
        start + min[0]..=start + max[0]
    };
    // One output plane per task, as iDaVIE splits over z, built a row of blocks at a time. On wasm the pool
    // runs them in turn.
    let fill_plane = |oz: usize, plane: &mut [f32], plane_counts: &mut [u32]| {
        let (mut acc, mut count) = (vec![0f32; out_dims[0]], vec![0u32; out_dims[0]]);
        // The weights of an unweighted mean.
        let ones = vec![1u32; factors[0]];
        for (oy, out) in plane.chunks_mut(out_dims[0]).enumerate() {
            if factors == [1; 3] && !keep_counts {
                out.copy_from_slice(&data[row(min[2] + oz, min[1] + oy)]);
                continue;
            }
            // f32::max ignores a NaN operand, so starting from NaN leaves an all-NaN block NaN.
            acc.fill(if max_mode { f32::NAN } else { 0.0 });
            count.fill(0);
            for z in block(2, oz) {
                for y in block(1, oy) {
                    let blocks = data[row(z, y)].chunks(factors[0]);
                    if max_mode {
                        for (a, values) in acc.iter_mut().zip(blocks) {
                            *a = values.iter().fold(*a, |a, &v| a.max(v));
                        }
                    } else {
                        let weights = weights.map(|w| &w[row(z, y)]);
                        for (b, ((a, n), values)) in
                            acc.iter_mut().zip(&mut count).zip(blocks).enumerate()
                        {
                            let weights = weights.map_or(&ones[..], |w| &w[b * factors[0]..]);
                            for (&v, &w) in
                                values.iter().zip(weights).filter(|(v, _)| v.is_finite())
                            {
                                *a += v * w as f32;
                                *n += w;
                            }
                        }
                    }
                }
            }
            if !max_mode {
                for (a, &n) in acc.iter_mut().zip(&count) {
                    *a = if n == 0 { f32::NAN } else { *a / n as f32 };
                }
            }
            out.copy_from_slice(&acc);
            if keep_counts {
                plane_counts[oy * out_dims[0]..][..out_dims[0]].copy_from_slice(&count);
            }
        }
    };
    let fill_plane = &fill_plane;
    let plane_len = out_dims[0] * out_dims[1];
    let mut out = vec![0f32; plane_len * out_dims[2]];
    let mut counts = vec![0u32; if keep_counts { out.len() } else { 0 }];
    let mut count_planes = counts.chunks_mut(plane_len);
    ComputeTaskPool::get_or_init(TaskPool::default).scope(|s| {
        for (oz, plane) in out.chunks_mut(plane_len).enumerate() {
            let plane_counts = count_planes.next().unwrap_or_default();
            s.spawn(async move { fill_plane(oz, plane, plane_counts) });
        }
    });
    (out_dims, out, counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cube of `data` with no header and no statistics.
    pub(crate) fn cube(dims: [usize; 3], data: Vec<f32>) -> Cube {
        Cube {
            info: FileInfo {
                dims,
                axes: [0, 1, 2],
                cards: vec![],
                hdu: 0,
                stats: Moments::EMPTY.stats(0),
                histogram: Histogram(vec![0]),
            },
            data,
        }
    }

    fn pad(mut bytes: Vec<u8>, fill: u8) -> Vec<u8> {
        bytes.resize(bytes.len().div_ceil(BLOCK) * BLOCK, fill);
        bytes
    }

    fn header(cards: &[&str]) -> Vec<u8> {
        let text: String = cards
            .iter()
            .chain(&["END"])
            .map(|c| format!("{c:<80}"))
            .collect();
        pad(text.into_bytes(), b' ')
    }

    #[test]
    fn reads_scaled_4d_extension_after_empty_primary() {
        for bitpix in [16, 32] {
            let bitpix_card = format!("BITPIX  = {bitpix:>20}");
            let mut file = header(&[
                "SIMPLE  =                    T",
                "BITPIX  =                    8",
                "NAXIS   =                    0",
                "EXTEND  =                    T",
            ]);
            let image = [
                "XTENSION= 'IMAGE   '           / image extension",
                &bitpix_card,
                "NAXIS   =                    4",
                "NAXIS1  =                    3",
                "NAXIS2  =                    2",
                "NAXIS3  =                    2",
                "NAXIS4  =                    2",
                "PCOUNT  =                    0",
                "GCOUNT  =                    1",
                "BSCALE  =                  0.5",
                "BZERO   =               1.0D01",
                "BLANK   =               -32768",
                "BUNIT   = 'Jy/beam '           / Brightness (pixel) unit",
            ];
            file.extend(header(&image));
            let raw = (0..24i32).map(|i| if i == 5 { -32768 } else { i });
            let bytes = match bitpix {
                16 => raw.flat_map(|v| (v as i16).to_be_bytes()).collect(),
                _ => raw.flat_map(|v| v.to_be_bytes()).collect(),
            };
            file.extend(pad(bytes, 0));

            let cube = read_fits(&file[..], None).unwrap();
            assert_eq!(
                (cube.info.hdu, cube.info.dims, cube.data.len()),
                (1, [3, 2, 2], 12)
            );
            for (i, v) in cube.data.iter().enumerate() {
                if i == 5 {
                    assert!(v.is_nan())
                } else {
                    assert_eq!(*v, 10.0 + 0.5 * i as f32)
                }
            }
            let info = &cube.info;
            assert_eq!(info.cards.len(), image.len());
            assert_eq!(info.cards[0], format!("{:<80}", image[0]));
            assert_eq!(info.unit().as_deref(), Some("Jy/beam"));
            assert_eq!(
                (info.stats.min, info.stats.max, info.stats.non_finite),
                (10.0, 15.5, 1)
            );
            assert!(read_fits(&file[..], Some(0)).is_err());
            assert_eq!(read_fits(&file[..], Some(1)).unwrap().info.dims, [3, 2, 2]);
        }
    }

    #[test]
    fn crop_and_downsample_clamps_edge_blocks_to_crop() {
        let dims = [5, 3, 2];
        let mut data: Vec<f32> = (0..30)
            .map(|i| (i % 5 + 10 * (i / 5 % 3) + 100 * (i / 15)) as f32)
            .collect();
        for i in [1, 14, 29] {
            data[i] = f32::NAN;
        }

        let crop = Region {
            min: [1, 0, 0],
            max: [4, 2, 1],
        };
        let (new_dims, max) = crop_and_downsample(&data, dims, crop, [3, 2, 2], true);
        assert_eq!(new_dims, [2, 2, 1]);
        assert_eq!(max[..3], [113.0, 114.0, 123.0]);
        assert!(max[3].is_nan());

        let (_, mean) = crop_and_downsample(&data, dims, crop, [3, 2, 2], false);
        assert_eq!(mean[..3], [683.0 / 11.0, 59.0, 72.0]);
        assert!(mean[3].is_nan());

        let crop = Region {
            min: [1, 1, 0],
            max: [3, 2, 1],
        };
        let (new_dims, full) = crop_and_downsample(&data, dims, crop, [1; 3], true);
        assert_eq!(new_dims, [3, 2, 2]);
        assert_eq!(
            full,
            [
                11., 12., 13., 21., 22., 23., 111., 112., 113., 121., 122., 123.
            ]
        );
    }

    #[test]
    fn downsampling_matches_a_voxel_at_a_time_reference() {
        let dims = [37, 29, 11];
        let mut seed = 12345u32;
        let data: Vec<f32> = (0..dims.iter().product::<usize>())
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                if seed.is_multiple_of(7) {
                    f32::NAN
                } else {
                    (seed >> 8) as f32 / 1e3 - 5e3
                }
            })
            .collect();
        // The plain version that the row-at-a-time one replaced: each output voxel's block in z, y, x order.
        let reference = |min: [usize; 3], max: [usize; 3], f: [usize; 3], max_mode: bool| {
            let mut out = vec![];
            for z0 in (min[2]..=max[2]).step_by(f[2]) {
                for y0 in (min[1]..=max[1]).step_by(f[1]) {
                    for x0 in (min[0]..=max[0]).step_by(f[0]) {
                        let (mut acc, mut n) = (if max_mode { f32::MIN } else { 0.0 }, 0);
                        for z in z0..(z0 + f[2]).min(max[2] + 1) {
                            for y in y0..(y0 + f[1]).min(max[1] + 1) {
                                for x in x0..(x0 + f[0]).min(max[0] + 1) {
                                    let v = data[(z * dims[1] + y) * dims[0] + x];
                                    if if max_mode { !v.is_nan() } else { v.is_finite() } {
                                        acc = if max_mode { acc.max(v) } else { acc + v };
                                        n += 1;
                                    }
                                }
                            }
                        }
                        out.push(match n {
                            0 => f32::NAN,
                            _ if max_mode => acc,
                            _ => acc / n as f32,
                        });
                    }
                }
            }
            out
        };
        for factors in [
            [1, 1, 1],
            [2, 2, 1],
            [3, 2, 2],
            [1, 1, 3],
            [5, 4, 3],
            [40, 40, 20],
        ] {
            for (min, max) in [([0; 3], [36, 28, 10]), ([3, 5, 1], [30, 20, 9])] {
                for max_mode in [true, false] {
                    let (_, got) =
                        crop_and_downsample(&data, dims, Region { min, max }, factors, max_mode);
                    let want = reference(min, max, factors, max_mode);
                    assert!(
                        got.len() == want.len()
                            && got.iter().zip(&want).all(
                                |(a, b)| a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan()
                            ),
                        "{factors:?} {min:?} {max_mode}"
                    );
                }
            }
        }
    }

    #[test]
    fn merged_chunk_moments_match_a_direct_calculation() {
        // A large pedestal is what a running sum of squares gets wrong, and 500..515 holds an all-NaN chunk.
        let pedestal: Vec<f32> = (0..1000)
            .map(|i| match i {
                500..515 => f32::NAN,
                _ if i % 97 == 0 => f32::INFINITY,
                _ => 1e4 + (i % 5) as f32 * 0.25,
            })
            .collect();
        // And a noise pixel far above everything else.
        let mut noisy = pedestal.clone();
        noisy[70] = 1e30;
        for data in [pedestal, noisy] {
            let finite: Vec<f64> = (data.iter().filter(|v| v.is_finite()))
                .map(|&v| v as f64)
                .collect();
            let mean = finite.iter().sum::<f64>() / finite.len() as f64;
            let m2: f64 = finite.iter().map(|v| (v - mean) * (v - mean)).sum();
            let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs();
            let merged = data
                .chunks(7)
                .map(Moments::of)
                .fold(Moments::EMPTY, Moments::merge);
            for m in [Moments::of(&data), merged] {
                assert_eq!(m.n, finite.len());
                assert!(
                    close(m.mean, mean) && close(m.m2, m2),
                    "{m:?} against {mean}, {m2}"
                );
            }
        }

        let constant = [3.0f32; 10]
            .chunks(3)
            .map(Moments::of)
            .fold(Moments::EMPTY, Moments::merge)
            .stats(10);
        assert_eq!((constant.mean, constant.std), (3.0, 0.0));
    }

    #[test]
    fn reads_data_spanning_several_chunks() {
        let len = 2 * CHUNK_VOXELS + 123;
        let mut file = header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                   16",
            "NAXIS   =                    3",
            &format!("NAXIS1  = {len:>20}"),
            "NAXIS2  =                    1",
            "NAXIS3  =                    1",
        ]);
        let value = |i: usize| (i % 1000) as i16;
        file.extend(pad(
            (0..len).flat_map(|i| value(i).to_be_bytes()).collect(),
            0,
        ));
        let cube = read_fits(&file[..], None).unwrap();
        assert!(
            (cube.data.iter().enumerate()).all(|(i, &v)| v == value(i) as f32),
            "voxels out of place across a chunk boundary"
        );
        let whole = Moments::of(&cube.data).stats(len);
        let stats = cube.info.stats;
        assert_eq!((stats.min, stats.max), (0.0, 999.0));
        assert!((stats.mean - whole.mean).abs() < 1e-9);
        assert!((stats.std - whole.std).abs() < 1e-9);
    }

    #[test]
    fn percentile_thresholds_match_an_exact_sort() {
        // Noise across several chunks, with a strong negative artefact and a few bright outliers that percentiles
        // should clip, as in THINGS' NGC 5236 cube, and NaN and infinite voxels to leave out.
        let len = 2 * CHUNK_VOXELS + 123;
        let mut seed = 12345u32;
        let mut uniform = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        let values: Vec<f32> = (0..len)
            .map(|i| {
                let noise = (0..12).map(|_| uniform()).sum::<f32>() - 6.0;
                match i {
                    _ if i % 1000 == 999 => f32::NAN,
                    _ if i % 300_000 == 5 => f32::NEG_INFINITY,
                    _ if i % 100_000 == 7 => noise - 80.0,
                    _ if i % 200_000 == 11 => noise + 20.0,
                    _ => noise,
                }
            })
            .collect();
        let mut file = header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                  -32",
            "NAXIS   =                    3",
            &format!("NAXIS1  = {len:>20}"),
            "NAXIS2  =                    1",
            "NAXIS3  =                    1",
        ]);
        file.extend(pad(
            values.iter().flat_map(|v| v.to_be_bytes()).collect(),
            0,
        ));
        let histogram = read_fits(&file[..], None).unwrap().info.histogram;

        let mut finite: Vec<f64> = (values.iter().filter(|v| v.is_finite()))
            .map(|&v| v.into())
            .collect();
        finite.sort_by(f64::total_cmp);
        assert_eq!(histogram.0.last(), Some(&(finite.len() as u64)));
        let (min, max) = (finite[0], finite[finite.len() - 1]);
        let exact = |rank: f64| {
            let at = rank * (finite.len() - 1) as f64;
            let (i, t) = (at as usize, at.fract());
            let v = finite[i] + t * (finite[(i + 1).min(finite.len() - 1)] - finite[i]);
            (v - min) / (max - min)
        };
        // Interpolating within a bin keeps every rank within about a twentieth of a bin here; without it, they'd be
        // off by up to a whole bin.
        let bin = 1.0 / histogram.0.len() as f64;
        for percent in PERCENTILES {
            let tail = (100.0 - percent) / 200.0;
            let (lo, hi) = histogram.clip(percent);
            for (got, want) in [(lo, exact(tail)), (hi, exact(1.0 - tail))] {
                assert!(
                    (f64::from(got) - want).abs() < 0.15 * bin,
                    "{percent}%: {got} against {want}"
                );
            }
        }
    }

    #[test]
    fn sigma_comes_from_the_channels_without_emission() {
        // σ = 0.5 on a pedestal, with emission over most of the middle channels, one channel all NaN and the last
        // eight zero-filled, more than the quietest quarter's middle.
        let dims = [64, 64, 40];
        let mut seed = 12345u32;
        let mut uniform = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        let data = (0..dims.iter().product::<usize>())
            .map(|i| {
                let (z, x) = (i / 4096, i % 64);
                let noise = 0.5 * ((0..12).map(|_| uniform()).sum::<f32>() - 6.0);
                match z {
                    3 => f32::NAN,
                    32.. => 0.0,
                    8..32 if x < 40 => 100.0 + noise + 5.0,
                    _ => 100.0 + noise,
                }
            })
            .collect();
        let sigma = cube(dims, data).sigma().unwrap();
        assert!((sigma - 0.5).abs() < 0.025, "{sigma}");
        assert_eq!(cube([2, 2, 2], vec![f32::NAN; 8]).sigma(), None);
    }

    #[test]
    fn rejects_headers_that_claim_more_than_the_file_holds() {
        let cube = |n: u64| {
            header(&[
                "SIMPLE  =                    T",
                "BITPIX  =                  -32",
                "NAXIS   =                    3",
                &format!("NAXIS1  = {n:>20}"),
                &format!("NAXIS2  = {n:>20}"),
                &format!("NAXIS3  = {:>20}", 100),
            ])
        };
        // 400 MB claimed with no data behind it fails before anything is allocated.
        let err = read_fits(&cube(1000)[..], None).err().unwrap();
        assert!(err.contains("HDU 0: data truncated"), "{err}");
        // A size that overflows in a skipped HDU stops there, rather than wrapping the offset around.
        let mut file = cube(1 << 32);
        file.extend(cube(1));
        let err = read_fits(&file[..], Some(1)).err().unwrap();
        assert!(err.contains("HDU 0: data truncated"), "{err}");
    }

    #[test]
    fn downsample_factors_match_idavie() {
        assert_eq!(find_downsample_factors(384, [4096, 4096, 100]), [5, 5, 1]);
        assert_eq!(find_downsample_factors(64, [512, 512, 2000]), [3, 3, 8]);
        // 4097 / 2 rounds up to 2049, over the 3D texture limit.
        assert_eq!(find_downsample_factors(384, [4097, 10, 10]), [3, 3, 1]);
        // A budget of zero still ends, at one voxel.
        assert_eq!(find_downsample_factors(0, [3, 2, 2]), [3, 3, 2]);
    }

    #[test]
    fn skips_a_random_groups_primary() {
        let mut file = header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                  -32",
            "NAXIS   =                    4",
            "NAXIS1  =                    0",
            "NAXIS2  =                    3",
            "NAXIS3  =                    2",
            "NAXIS4  =                    1",
            "GROUPS  =                    T",
            "PCOUNT  =                    2",
            "GCOUNT  =                    5",
        ]);
        // 5 groups of 2 parameters and 3 x 2 x 1 values, so 160 bytes.
        file.extend(pad(vec![0; 160], 0));
        file.extend(header(&[
            "XTENSION= 'IMAGE   '",
            "BITPIX  =                  -32",
            "NAXIS   =                    3",
            "NAXIS1  =                    2",
            "NAXIS2  =                    1",
            "NAXIS3  =                    1",
            "BLANK   =                  0.5",
        ]));
        file.extend(pad(
            [1.0f32, f32::INFINITY]
                .iter()
                .flat_map(|v| v.to_be_bytes())
                .collect(),
            0,
        ));

        let info = read_fits(&file[..], None).unwrap().info;
        assert_eq!((info.hdu, info.dims), (1, [2, 1, 1]));
        assert!(read_fits(&file[..], Some(0)).is_err());
        // An infinite voxel stays out of the range, which would otherwise scale everything to zero.
        assert_eq!((info.stats.max, info.stats.non_finite), (1.0, 1));
    }

    #[test]
    fn picks_the_first_three_axes_longer_than_one() {
        let image = |lengths: &[u64]| {
            let mut cards = vec![
                "SIMPLE  =                    T".to_string(),
                "BITPIX  =                  -32".to_string(),
                format!("NAXIS   = {:>20}", lengths.len()),
                "CTYPE4  = 'STOKES  '".to_string(),
            ];
            cards.extend(
                (1..)
                    .zip(lengths)
                    .map(|(i, n)| format!("NAXIS{i:<3}= {n:>20}")),
            );
            let mut file = header(&cards.iter().map(String::as_str).collect::<Vec<_>>());
            let values: Vec<f32> = (0..lengths.iter().product::<u64>())
                .map(|v| v as f32)
                .collect();
            file.extend(pad(
                values.iter().flat_map(|v| v.to_be_bytes()).collect(),
                0,
            ));
            read_fits(&file[..], None).unwrap()
        };
        // CASA's RA, DEC, STOKES(1), FREQ order.
        let casa = image(&[2, 2, 1, 4]);
        assert_eq!(
            (casa.info.dims, casa.info.axes, casa.info.other_planes()),
            ([2, 2, 4], [0, 1, 3], None)
        );
        assert_eq!(casa.data, (0..16).map(|v| v as f32).collect::<Vec<_>>());
        let flat = image(&[3, 2]);
        assert_eq!(
            (flat.info.dims, flat.info.axes, flat.data.len()),
            ([3, 2, 1], [0, 1, 2], 6)
        );
        let stokes = image(&[2, 2, 2, 3]);
        assert_eq!((stokes.info.dims, stokes.data.len()), ([2, 2, 2], 8));
        assert_eq!(
            stokes.info.other_planes().as_deref(),
            Some("1 of 3 along STOKES")
        );
    }

    #[test]
    fn reads_quoted_and_continued_strings() {
        let info = FileInfo {
            cards: [
                "BUNIT   = '  K ''per'' &'      / doubled quotes, continued",
                "CONTINUE  'beam'               / unit",
                "OBJECT  = 'AT&T'",
                "BSCALE  =                      / undefined",
            ]
            .map(|c| format!("{c:<80}"))
            .into(),
            ..cube([1; 3], vec![]).info
        };
        assert_eq!(info.unit().as_deref(), Some("K 'per' beam"));
        assert_eq!(value(&info.cards, "OBJECT").as_deref(), Some("AT&T"));
        assert_eq!(num::<f64>(&info.cards, "BSCALE"), Ok(None));
    }

    #[test]
    fn an_inheriting_extension_takes_the_primary_cards_it_lacks() {
        let primary = [
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "EXTEND  =                    T",
            "BUNIT   = 'K       '",
            "SPECSYS = 'BARYCENT'",
            "COMMENT   from the primary",
        ];
        let extension = |inherit: &str| {
            let mut file = header(&primary);
            file.extend(header(&[
                "XTENSION= 'IMAGE   '",
                "BITPIX  =                  -32",
                "NAXIS   =                    2",
                "NAXIS1  =                    2",
                "NAXIS2  =                    1",
                "PCOUNT  =                    0",
                "GCOUNT  =                    1",
                inherit,
                "BUNIT   = 'Jy/beam '",
            ]));
            file.extend(pad(vec![0; 8], 0));
            read_fits(&file[..], None).unwrap().info
        };
        let info = extension("INHERIT =                    T");
        assert_eq!(info.unit().as_deref(), Some("Jy/beam"));
        assert_eq!(value(&info.cards, "SPECSYS").as_deref(), Some("BARYCENT"));
        assert_eq!(value(&info.cards, "BITPIX").as_deref(), Some("-32"));
        assert_eq!(info.cards.len(), 9 + 2);
        assert_eq!(extension("INHERIT =                    F").cards.len(), 9);
    }
}
