use std::io::{self, ErrorKind};
use std::str::FromStr;

use bevy::tasks::{ComputeTaskPool, ParallelSliceMut, TaskPool};

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

pub struct Cube {
    pub dims: [usize; 3],
    pub data: Vec<f32>,
    pub cards: Vec<String>,
    pub hdu: usize,
    pub stats: Stats,
}

impl Cube {
    /// The data unit from BUNIT, e.g. "Jy/beam" (or CASA's "beam-1 Jy").
    pub fn unit(&self) -> Option<&str> {
        value(&self.cards, "BUNIT").filter(|u| !u.is_empty())
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
    loop {
        let Some(cards) = read_header(source, &mut offset)? else {
            return Err(match hdu {
                Some(n) => format!("HDU {n} not found, the file has {index}"),
                None => "no image HDU with NAXIS >= 3 found".into(),
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
        let groups = value(&cards, "GROUPS") == Some("T");
        let is_cube = naxis >= 3
            && axes[..3].iter().all(|&n| n > 0)
            && !groups
            && (index == 0 || value(&cards, "XTENSION") == Some("IMAGE"));

        if hdu == Some(index) || (hdu.is_none() && is_cube) {
            if !is_cube {
                return Err(format!(
                    "HDU {index} is not an image with three non-empty axes"
                ));
            }
            let len = checked_product(&axes[..3])
                .and_then(|n| usize::try_from(n).ok())
                .ok_or(format!("HDU {index}: image too large"))?;
            // Each axis is at most `len`, which fits.
            let dims = [0, 1, 2].map(|i| axes[i] as usize);
            // Checked before allocating, so a header can't claim more memory than its file backs.
            if (len as u64)
                .checked_mul(width)
                .and_then(|bytes| bytes.checked_add(offset))
                .is_none_or(|end| end > size)
            {
                return Err(format!("HDU {index}: data truncated"));
            }
            let (data, stats) = read_data(source, offset, &cards, bitpix, width as usize, len)?;
            return Ok(Cube {
                dims,
                data,
                cards,
                hdu: index,
                stats,
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
        index += 1;
    }
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

/// A card's value: a quoted string without its quotes (and trailing spaces), or the text before any comment.
fn value<'a>(cards: &'a [String], key: &str) -> Option<&'a str> {
    let card = cards
        .iter()
        .find(|c| c[..8].trim_end() == key && &c[8..10] == "= ")?;
    let field = card[10..].trim_start();
    // ponytail: a quote doubled inside the string ('O''Brien') ends it early; fine for the keys read here.
    match field.strip_prefix('\'') {
        Some(quoted) => Some(quoted[..quoted.find('\'')?].trim_end()),
        None => Some(field.split('/').next()?.trim()),
    }
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
) -> Result<(Vec<f32>, Stats), String> {
    let bscale: f64 = num(cards, "BSCALE")?.unwrap_or(1.0);
    let bzero: f64 = num(cards, "BZERO")?.unwrap_or(0.0);
    // BLANK only applies to integer data.
    let blank: Option<i64> = if bitpix > 0 {
        num(cards, "BLANK")?
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

    let mut data = Vec::new();
    data.try_reserve_exact(len)
        .map_err(|_| format!("cannot allocate {len} voxels"))?;
    data.resize(len, 0.0);
    let decode = |chunk: usize, out: &mut [f32]| -> Result<Moments, String> {
        let mut raw = vec![0u8; out.len() * width];
        source
            .read_at(offset + (chunk * CHUNK_VOXELS * width) as u64, &mut raw)
            .map_err(|e| format!("reading FITS data: {e}"))?;
        match bitpix {
            8 => convert(out, &raw, |b: [u8; 1]| int(b[0] as i64)),
            16 => convert(out, &raw, |b| int(i16::from_be_bytes(b) as i64)),
            32 => convert(out, &raw, |b| int(i32::from_be_bytes(b) as i64)),
            64 => convert(out, &raw, |b| int(i64::from_be_bytes(b))),
            -32 => convert(out, &raw, |b| float(f32::from_be_bytes(b) as f64)),
            _ => convert(out, &raw, |b| float(f64::from_be_bytes(b))),
        }
        Ok(Moments::of(out))
    };
    #[cfg(not(target_arch = "wasm32"))]
    let chunks = data.par_chunk_map_mut(
        ComputeTaskPool::get_or_init(TaskPool::default),
        CHUNK_VOXELS,
        decode,
    );
    #[cfg(target_arch = "wasm32")]
    let chunks: Vec<_> = data
        .chunks_mut(CHUNK_VOXELS)
        .enumerate()
        .map(|(i, out)| decode(i, out))
        .collect();
    let moments = chunks.into_iter().try_fold(Moments::EMPTY, |all, chunk| {
        Ok::<_, String>(all.merge(chunk?))
    })?;
    Ok((data, moments.stats(len)))
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

    /// Two passes over a chunk that's still in cache: a single running sum of squares loses precision on
    /// data with a large pedestal.
    fn of(values: &[f32]) -> Self {
        let finite = || values.iter().filter(|v| v.is_finite()).map(|&v| v as f64);
        let (mut n, mut sum, mut min, mut max) = (0, 0.0, f64::INFINITY, f64::NEG_INFINITY);
        for v in finite() {
            n += 1;
            sum += v;
            min = min.min(v);
            max = max.max(v);
        }
        let mean = sum / n as f64;
        let m2 = finite().map(|v| (v - mean) * (v - mean)).sum();
        Self {
            n,
            mean,
            m2,
            min,
            max,
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

pub fn find_downsample_factors(max_cube_size_mb: u64, region_dims: [usize; 3]) -> [usize; 3] {
    let [x, y, z] = region_dims.map(|d| d as u64);
    let (mut fx, mut fy, mut fz) = (1, 1, 1);
    // Rounded up, as crop_and_downsample sizes the texture; iDaVIE rounds down here and can ask for 2049.
    while x.div_ceil(fx) > 2048 || y.div_ceil(fy) > 2048 {
        fx += 1;
        fy += 1;
    }
    while z.div_ceil(fz) > 2048 {
        fz += 1;
    }
    let max_elements = max_cube_size_mb * 1_000_000 / 4;
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

/// min/max are 0-based inclusive voxel corners. Returns (new_dims, data) with data x-fastest.
pub fn crop_and_downsample(
    cube: &Cube,
    min: [usize; 3],
    max: [usize; 3],
    factors: [usize; 3],
    max_mode: bool,
) -> ([usize; 3], Vec<f32>) {
    assert!(
        (0..3).all(|i| min[i] <= max[i] && max[i] < cube.dims[i] && factors[i] > 0),
        "invalid crop or factors"
    );
    let dims: [usize; 3] = std::array::from_fn(|i| (max[i] - min[i] + 1).div_ceil(factors[i]));
    let block = |i: usize, o: usize| {
        min[i] + o * factors[i]..(min[i] + (o + 1) * factors[i]).min(max[i] + 1)
    };
    let [width, height, _] = cube.dims;
    let row = |z: usize, y: usize| &cube.data[(z * height + y) * width..];
    // One output plane per task, as iDaVIE splits over z. On wasm the pool runs them in turn.
    let fill_plane = |oz: usize, plane: &mut [f32]| {
        if factors == [1; 3] {
            for (out, y) in plane.chunks_mut(dims[0]).zip(min[1]..) {
                out.copy_from_slice(&row(min[2] + oz, y)[min[0]..=max[0]]);
            }
            return;
        }
        for (oy, out) in plane.chunks_mut(dims[0]).enumerate() {
            for (ox, out) in out.iter_mut().enumerate() {
                let (mut acc, mut count) = (if max_mode { f32::MIN } else { 0.0 }, 0);
                for z in block(2, oz) {
                    for y in block(1, oy) {
                        for &v in row(z, y)[block(0, ox)].iter().filter(|v| !v.is_nan()) {
                            acc = if max_mode { acc.max(v) } else { acc + v };
                            count += 1;
                        }
                    }
                }
                *out = match count {
                    0 => f32::NAN,
                    _ if max_mode => acc,
                    _ => acc / count as f32,
                };
            }
        }
    };
    let mut out = vec![0.0; dims.iter().product()];
    out.par_chunk_map_mut(
        ComputeTaskPool::get_or_init(TaskPool::default),
        dims[0] * dims[1],
        fill_plane,
    );
    (dims, out)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut file = header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "EXTEND  =                    T",
        ]);
        let image = [
            "XTENSION= 'IMAGE   '           / image extension",
            "BITPIX  =                   16",
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
        let raw: Vec<i16> = (0..24).map(|i| if i == 5 { i16::MIN } else { i }).collect();
        file.extend(pad(raw.iter().flat_map(|v| v.to_be_bytes()).collect(), 0));

        let cube = read_fits(&file[..], None).unwrap();
        assert_eq!((cube.hdu, cube.dims, cube.data.len()), (1, [3, 2, 2], 12));
        for (i, v) in cube.data.iter().enumerate() {
            if i == 5 {
                assert!(v.is_nan())
            } else {
                assert_eq!(*v, 10.0 + 0.5 * i as f32)
            }
        }
        assert_eq!(cube.cards.len(), image.len());
        assert_eq!(cube.cards[0], format!("{:<80}", image[0]));
        assert_eq!(cube.unit(), Some("Jy/beam"));
        assert_eq!(
            (cube.stats.min, cube.stats.max, cube.stats.non_finite),
            (10.0, 15.5, 1)
        );
        assert!(read_fits(&file[..], Some(0)).is_err());
        assert_eq!(read_fits(&file[..], Some(1)).unwrap().dims, [3, 2, 2]);
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
        let cube = Cube {
            dims,
            data,
            cards: vec![],
            hdu: 0,
            stats: Moments::EMPTY.stats(0),
        };

        let (new_dims, max) = crop_and_downsample(&cube, [1, 0, 0], [4, 2, 1], [3, 2, 2], true);
        assert_eq!(new_dims, [2, 2, 1]);
        assert_eq!(max[..3], [113.0, 114.0, 123.0]);
        assert!(max[3].is_nan());

        let (_, mean) = crop_and_downsample(&cube, [1, 0, 0], [4, 2, 1], [3, 2, 2], false);
        assert_eq!(mean[..3], [683.0 / 11.0, 59.0, 72.0]);
        assert!(mean[3].is_nan());

        let (new_dims, full) = crop_and_downsample(&cube, [1, 1, 0], [3, 2, 1], [1; 3], true);
        assert_eq!(new_dims, [3, 2, 2]);
        assert_eq!(
            full,
            [
                11., 12., 13., 21., 22., 23., 111., 112., 113., 121., 122., 123.
            ]
        );
    }

    #[test]
    fn merged_chunk_moments_match_the_whole() {
        // A large pedestal is what the running sum of squares got wrong, and 500..515 holds an all-NaN chunk.
        let data: Vec<f32> = (0..1000)
            .map(|i| match i {
                500..515 => f32::NAN,
                _ if i % 97 == 0 => f32::INFINITY,
                _ => 1e4 + (i % 5) as f32 * 0.25,
            })
            .collect();
        let whole = Moments::of(&data);
        let merged = data
            .chunks(7)
            .map(Moments::of)
            .fold(Moments::EMPTY, Moments::merge);
        assert_eq!(
            (merged.n, merged.min, merged.max),
            (whole.n, whole.min, whole.max)
        );
        assert!((merged.mean - whole.mean).abs() < 1e-9);
        assert!((merged.m2 - whole.m2).abs() < 1e-9 * whole.m2);

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
        assert_eq!((cube.stats.min, cube.stats.max), (0.0, 999.0));
        assert!((cube.stats.mean - whole.mean).abs() < 1e-9);
        assert!((cube.stats.std - whole.std).abs() < 1e-9);
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

        let cube = read_fits(&file[..], None).unwrap();
        assert_eq!((cube.hdu, cube.dims), (1, [2, 1, 1]));
        assert!(read_fits(&file[..], Some(0)).is_err());
        // An infinite voxel stays out of the range, which would otherwise scale everything to zero.
        assert_eq!((cube.stats.max, cube.stats.non_finite), (1.0, 1));
    }
}
