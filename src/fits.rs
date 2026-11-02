use std::io::{ErrorKind, Read};
use std::str::FromStr;

const BLOCK: usize = 2880;
const CHUNK: usize = 4 << 20;

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
        let card = self.cards.iter().find(|c| c.starts_with("BUNIT   ="))?;
        let quoted = card[9..].trim_start().strip_prefix('\'')?;
        let unit = quoted[..quoted.find('\'')?].trim_end();
        (!unit.is_empty()).then_some(unit)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub min: f32,
    pub max: f32,
    pub mean: f64,
    pub std: f64,
    pub nan_count: usize,
}

pub fn read_fits(mut reader: impl Read, hdu: Option<usize>) -> Result<Cube, String> {
    let mut index = 0;
    loop {
        let Some(cards) = read_header(&mut reader)? else {
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
        let is_cube = naxis >= 3 && (index == 0 || value(&cards, "XTENSION") == Some("IMAGE"));

        if hdu == Some(index) || (hdu.is_none() && is_cube) {
            if !is_cube {
                return Err(format!("HDU {index} is not an image with NAXIS >= 3"));
            }
            let dims = [axes[0] as usize, axes[1] as usize, axes[2] as usize];
            let data = read_data(
                &mut reader,
                &cards,
                bitpix,
                width as usize,
                dims.iter().product(),
            )?;
            let stats = stats(&data);
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
        let bytes = if naxis == 0 {
            0
        } else {
            width * gcount * (pcount + axes.iter().product::<u64>())
        };
        let padded = bytes.div_ceil(BLOCK as u64) * BLOCK as u64;
        let skipped = std::io::copy(&mut reader.by_ref().take(padded), &mut std::io::sink())
            .map_err(|e| e.to_string())?;
        if skipped != padded {
            return Err(format!("HDU {index}: data truncated"));
        }
        index += 1;
    }
}

/// Returns None at a clean end of file.
fn read_header(reader: &mut impl Read) -> Result<Option<Vec<String>>, String> {
    let mut cards = Vec::new();
    let mut block = [0u8; BLOCK];
    loop {
        match reader.read_exact(&mut block) {
            Err(e) if e.kind() == ErrorKind::UnexpectedEof && cards.is_empty() => return Ok(None),
            r => r.map_err(|e| format!("reading FITS header: {e}"))?,
        }
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

fn value<'a>(cards: &'a [String], key: &str) -> Option<&'a str> {
    let card = cards
        .iter()
        .find(|c| c[..8].trim_end() == key && &c[8..10] == "= ")?;
    Some(
        card[10..]
            .split('/')
            .next()?
            .trim()
            .trim_matches('\'')
            .trim(),
    )
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

fn read_data(
    reader: &mut impl Read,
    cards: &[String],
    bitpix: i64,
    width: usize,
    len: usize,
) -> Result<Vec<f32>, String> {
    let bscale: f64 = num(cards, "BSCALE")?.unwrap_or(1.0);
    let bzero: f64 = num(cards, "BZERO")?.unwrap_or(0.0);
    let blank: Option<i64> = num(cards, "BLANK")?;
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
    let mut buf = vec![0u8; CHUNK.min(len * width)];
    while data.len() < len {
        let bytes = &mut buf[..(len - data.len()).min(CHUNK / width) * width];
        reader
            .read_exact(bytes)
            .map_err(|e| format!("reading FITS data: {e}"))?;
        match bitpix {
            8 => extend(&mut data, bytes, |b: [u8; 1]| int(b[0] as i64)),
            16 => extend(&mut data, bytes, |b| int(i16::from_be_bytes(b) as i64)),
            32 => extend(&mut data, bytes, |b| int(i32::from_be_bytes(b) as i64)),
            64 => extend(&mut data, bytes, |b| int(i64::from_be_bytes(b))),
            -32 => extend(&mut data, bytes, |b| float(f32::from_be_bytes(b) as f64)),
            _ => extend(&mut data, bytes, |b| float(f64::from_be_bytes(b))),
        }
    }
    Ok(data)
}

fn extend<const N: usize>(data: &mut Vec<f32>, bytes: &[u8], f: impl Fn([u8; N]) -> f32) {
    data.extend(bytes.as_chunks::<N>().0.iter().map(|&b| f(b)));
}

fn stats(data: &[f32]) -> Stats {
    let (mut min, mut max, mut sum, mut squares, mut n) =
        (f32::INFINITY, f32::NEG_INFINITY, 0.0, 0.0, 0usize);
    for &v in data.iter().filter(|v| !v.is_nan()) {
        min = min.min(v);
        max = max.max(v);
        sum += v as f64;
        squares += v as f64 * v as f64;
        n += 1;
    }
    let count = n as f64;
    let std = ((count * squares - sum * sum) / (count * (count - 1.0))).sqrt();
    Stats {
        min,
        max,
        mean: sum / count,
        std,
        nan_count: data.len() - n,
    }
}

pub fn find_downsample_factors(max_cube_size_mb: u64, region_dims: [usize; 3]) -> [usize; 3] {
    let [x, y, z] = region_dims.map(|d| d as u64);
    let (mut fx, mut fy, mut fz) = (1, 1, 1);
    while x / fx > 2048 || y / fy > 2048 {
        fx += 1;
        fy += 1;
    }
    while z / fz > 2048 {
        fz += 1;
    }
    let max_elements = max_cube_size_mb * 1_000_000 / 4;
    while x * y * z / (fx * fy * fz) > max_elements {
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
    let mut out = Vec::with_capacity(dims.iter().product());
    for oz in 0..dims[2] {
        for oy in 0..dims[1] {
            for ox in 0..dims[0] {
                let (mut acc, mut count) = (if max_mode { f32::MIN } else { 0.0 }, 0);
                for z in block(2, oz) {
                    for y in block(1, oy) {
                        let row = &cube.data[(z * height + y) * width..];
                        for &v in row[block(0, ox)].iter().filter(|v| !v.is_nan()) {
                            acc = if max_mode { acc.max(v) } else { acc + v };
                            count += 1;
                        }
                    }
                }
                out.push(match count {
                    0 => f32::NAN,
                    _ if max_mode => acc,
                    _ => acc / count as f32,
                });
            }
        }
    }
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
            (cube.stats.min, cube.stats.max, cube.stats.nan_count),
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
            stats: stats(&[]),
        };

        let (new_dims, max) = crop_and_downsample(&cube, [1, 0, 0], [4, 2, 1], [3, 2, 2], true);
        assert_eq!(new_dims, [2, 2, 1]);
        assert_eq!(max[..3], [113.0, 114.0, 123.0]);
        assert!(max[3].is_nan());

        let (_, mean) = crop_and_downsample(&cube, [1, 0, 0], [4, 2, 1], [3, 2, 2], false);
        assert_eq!(mean[..3], [683.0 / 11.0, 59.0, 72.0]);
        assert!(mean[3].is_nan());
    }

    #[test]
    fn downsample_factors_match_idavie() {
        assert_eq!(find_downsample_factors(384, [4096, 4096, 100]), [5, 5, 1]);
        assert_eq!(find_downsample_factors(64, [512, 512, 2000]), [3, 3, 8]);
    }
}
