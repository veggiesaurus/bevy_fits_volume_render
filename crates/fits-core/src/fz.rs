//! Tile-compressed images, as fpack writes them to `.fits.fz`: a binary table with a row for each tile of the image,
//! compressed on its own. RICE_1, GZIP_1, GZIP_2 and NOCOMPRESS are read, with floats quantised or kept as they were;
//! PLIO_1 and HCOMPRESS_1 are refused.

use std::{io::Read, sync::OnceLock};

use crate::{Source, num, value};

/// The most a value's bytes shrink by, DEFLATE's 1032 to 1, which no compression here beats; RICE_1's is about 200.
/// A tile or an image that claims more than its data could hold is refused before anything is allocated for it.
const MOST_SHRUNK: u64 = 1032;
/// The most voxels a tile may have, since one is decoded whole: a 4096 x 4096 plane. fpack's own are a row, or about
/// 100 x 100 when asked for.
const MOST_TILE: u64 = 1 << 24;

/// The image's own cards: the table's, without those that describe the table, and with the Z keywords that describe the
/// image under their own names.
pub(crate) fn image_cards(cards: &[String]) -> Vec<String> {
    const TABLE: [&str; 18] = [
        "XTENSION", "BITPIX", "NAXIS", "PCOUNT", "GCOUNT", "TFIELDS", "THEAP", "ZIMAGE",
        "ZCMPTYPE", "ZQUANTIZ", "ZDITHER0", "ZBLANK", "ZSCALE", "ZZERO", "ZBLOCKED", "ZHECKSUM",
        "ZDATASUM", "ZMASKCMP",
    ];
    const NUMBERED: [&str; 12] = [
        "NAXIS", "TTYPE", "TFORM", "TUNIT", "TDIM", "TSCAL", "TZERO", "TNULL", "TDISP", "ZTILE",
        "ZNAME", "ZVAL",
    ];
    const RENAMED: [&str; 7] = [
        "ZSIMPLE", "ZTENSION", "ZBITPIX", "ZNAXIS", "ZPCOUNT", "ZGCOUNT", "ZEXTEND",
    ];
    let numbered = |key: &str, prefix: &str| {
        (key.strip_prefix(prefix))
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
    };
    (cards.iter())
        .filter_map(|card| {
            let key = card[..8].trim_end();
            if RENAMED.contains(&key) || numbered(key, "ZNAXIS") {
                let own = if key == "ZTENSION" {
                    "XTENSION"
                } else {
                    &key[1..]
                };
                return Some(format!("{own:<8}{}", &card[8..]));
            }
            // The table's checksums don't hold for the image, and fpack names an image that had no name.
            let placeholder =
                key == "EXTNAME" && value(cards, "EXTNAME").as_deref() == Some("COMPRESSED_IMAGE");
            let table =
                TABLE.contains(&key) || ["CHECKSUM", "DATASUM"].contains(&key) || placeholder;
            (!table && !NUMBERED.iter().any(|prefix| numbered(key, prefix))).then(|| card.clone())
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq)]
enum Compression {
    Rice,
    Gzip,
    /// GZIP_2, with each value's bytes shuffled into planes first.
    Shuffled,
    None,
}

/// How floats were made integers: not at all, or scaled with or without fpack's dither.
#[derive(Clone, Copy, PartialEq)]
enum Quantise {
    Lossless,
    Plain,
    /// `seed` is ZDITHER0 less 1, the first tile's place in the random numbers.
    Dither {
        seed: u64,
        zeros: bool,
    },
}

/// A field of each row, or a value for every tile.
#[derive(Clone, Copy)]
enum Field {
    Column { at: u64, kind: u8 },
    Value(f64),
}

/// A column of variable-length arrays: where each row's descriptor is, whether it's a 64-bit Q one, and the width of
/// the array's elements.
#[derive(Clone, Copy)]
struct Array {
    at: u64,
    wide: bool,
    kind: u8,
}

pub(crate) struct Tiled {
    row_bytes: u64,
    rows: u64,
    /// The heap's place after the table's start, and its length.
    heap: u64,
    heap_len: u64,
    /// A tile's extent along each axis, whose product `new` has checked.
    tile: Vec<u64>,
    compression: Compression,
    quantise: Quantise,
    block: usize,
    bytepix: usize,
    data: Array,
    /// Where fpack keeps tiles it couldn't quantise, losslessly: gzipped, or as they were.
    gzipped: Option<Array>,
    whole: Option<Array>,
    scale: Option<Field>,
    zero: Option<Field>,
    blank: Option<Field>,
}

fn width(kind: u8) -> Option<u64> {
    Some(match kind {
        b'L' | b'B' | b'A' => 1,
        b'I' => 2,
        b'J' | b'E' => 4,
        b'K' | b'D' | b'C' => 8,
        b'M' => 16,
        b'P' => 8,
        b'Q' => 16,
        _ => return None,
    })
}

impl Tiled {
    /// The layout of the table of `cards`, a tile-compressed image of `lengths` with `bitpix`, or why it can't be read.
    pub(crate) fn new(cards: &[String], lengths: &[u64], bitpix: i64) -> Result<Self, String> {
        let get = |key: &str| num::<u64>(cards, key);
        let algorithm = value(cards, "ZCMPTYPE").ok_or("no ZCMPTYPE")?;
        let compression = match algorithm.as_str() {
            "RICE_1" | "RICE_ONE" => Compression::Rice,
            "GZIP_1" => Compression::Gzip,
            "GZIP_2" => Compression::Shuffled,
            "NOCOMPRESS" => Compression::None,
            other => {
                return Err(format!(
                    "the image is compressed with {other}, which isn't supported; decompress it with funpack"
                ));
            }
        };
        let (rows, row_bytes) = (get("NAXIS2")?.unwrap_or(0), get("NAXIS1")?.unwrap_or(0));
        let (table, pcount) = (rows.saturating_mul(row_bytes), get("PCOUNT")?.unwrap_or(0));
        // PCOUNT counts any gap between the table and the heap too.
        let heap = get("THEAP")?.unwrap_or(table).max(table);
        let heap_len = (table.checked_add(pcount))
            .and_then(|end| end.checked_sub(heap))
            .ok_or("a table or THEAP past the data")?;
        let fields: u64 = get("TFIELDS")?.unwrap_or(0);
        let (mut at, mut columns) = (0u64, Vec::new());
        for n in 1..=fields.min(999) {
            let form = value(cards, &format!("TFORM{n}")).ok_or(format!("no TFORM{n}"))?;
            let digits = form.bytes().take_while(u8::is_ascii_digit).count();
            let repeat: u64 = form[..digits].parse().unwrap_or(1);
            let kind = *form
                .as_bytes()
                .get(digits)
                .ok_or(format!("TFORM{n} = {form}"))?;
            let element = form.as_bytes().get(digits + 1).copied();
            let bytes = match kind {
                b'X' => Some(repeat.div_ceil(8)),
                _ => width(kind).and_then(|w| repeat.checked_mul(w)),
            };
            let bytes = bytes.ok_or(format!("TFORM{n} = {form}"))?;
            columns.push((
                value(cards, &format!("TTYPE{n}")).unwrap_or_default(),
                at,
                repeat,
                kind,
                element,
            ));
            at = at.checked_add(bytes).ok_or(format!("TFORM{n} = {form}"))?;
        }
        if at != row_bytes {
            return Err(format!(
                "a table of {row_bytes}-byte rows whose columns take {at}"
            ));
        }
        let column = |name: &str| columns.iter().find(|c| c.0 == name);
        let array = |name: &str| match column(name) {
            Some(&(_, at, 1, kind @ (b'P' | b'Q'), Some(element))) if width(element).is_some() => {
                Ok(Some(Array {
                    at,
                    wide: kind == b'Q',
                    kind: element,
                }))
            }
            Some(_) => Err(format!("a {name} column that isn't an array")),
            None => Ok(None),
        };
        let field = |name: &str| -> Result<Option<Field>, String> {
            match column(name) {
                Some(&(_, at, 1, kind @ (b'I' | b'J' | b'K' | b'E' | b'D'), _)) => {
                    Ok(Some(Field::Column { at, kind }))
                }
                Some(_) => Err(format!("a {name} column of more than one number")),
                None => Ok(num::<f64>(cards, name)?.map(Field::Value)),
            }
        };
        let data = array("COMPRESSED_DATA")?.ok_or("no COMPRESSED_DATA column")?;
        let scale = field("ZSCALE")?;
        let quantised = match value(cards, "ZQUANTIZ").as_deref() {
            _ if bitpix > 0 || scale.is_none() => None,
            None | Some("NO_DITHER") => Some(Quantise::Plain),
            Some("NONE") => None,
            Some(dither @ ("SUBTRACTIVE_DITHER_1" | "SUBTRACTIVE_DITHER_2")) => {
                Some(Quantise::Dither {
                    seed: (get("ZDITHER0")?.unwrap_or(1).max(1) - 1) % 10_000,
                    zeros: dither.ends_with('2'),
                })
            }
            Some(other) => {
                return Err(format!(
                    "floats quantised by {other}, which isn't supported"
                ));
            }
        };
        let quantise = quantised.unwrap_or(Quantise::Lossless);
        // RICE_1's own parameters, in fpack's ZNAMEn and ZVALn.
        let parameter = |name: &str| -> Result<Option<u64>, String> {
            for n in 1..=999 {
                match value(cards, &format!("ZNAME{n}")) {
                    Some(found) if found == name => return num(cards, &format!("ZVAL{n}")),
                    Some(_) => {}
                    None => return Ok(None),
                }
            }
            Ok(None)
        };
        // Only RICE_1 has it.
        let bytepix = match compression {
            Compression::Rice => parameter("BYTEPIX")?.unwrap_or(4) as usize,
            _ => 0,
        };
        if compression == Compression::Rice
            && (quantise == Quantise::Lossless && bitpix < 0 || ![1, 2, 4].contains(&bytepix))
        {
            return Err(format!(
                "RICE_1 of {bitpix}-bit values in {bytepix} bytes, which isn't supported"
            ));
        }
        let mut tile = Vec::new();
        for (i, &length) in lengths.iter().enumerate() {
            let default = if i == 0 { length } else { 1 };
            let n = get(&format!("ZTILE{}", i + 1))?.unwrap_or(default);
            if !(1..=length).contains(&n) {
                return Err(format!("ZTILE{} = {n} for an axis of {length}", i + 1));
            }
            tile.push(n);
        }
        let pixels = crate::checked_product(&tile).filter(|&n| n <= MOST_TILE);
        pixels.ok_or(format!(
            "tiles of {tile:?}, more than the {MOST_TILE} voxels read at a time; funpack it first"
        ))?;
        let tiles =
            (lengths.iter().zip(&tile)).try_fold(1u64, |n, (&l, &t)| n.checked_mul(l.div_ceil(t)));
        if tiles != Some(rows) {
            return Err(format!("{rows} rows for {tiles:?} tiles"));
        }
        Ok(Self {
            row_bytes,
            rows,
            heap,
            heap_len,
            tile,
            compression,
            quantise,
            block: parameter("BLOCKSIZE")?.unwrap_or(32).max(1) as usize,
            bytepix,
            data,
            gzipped: array("GZIP_COMPRESSED_DATA")?,
            whole: array("UNCOMPRESSED_DATA")?,
            scale,
            zero: field("ZZERO")?,
            blank: field("ZBLANK")?,
        })
    }

    /// Whether the table, which starts at `start`, is in the file's `size` bytes, and its heap could hold `bytes` of
    /// image.
    pub(crate) fn holds(&self, start: u64, size: u64, bytes: u64) -> bool {
        let in_file = (self.heap.checked_add(self.heap_len))
            .and_then(|end| end.checked_add(start))
            .is_some_and(|end| end <= size);
        in_file && bytes <= self.heap_len.saturating_mul(MOST_SHRUNK)
    }

    /// Tile `k`'s first voxel and its extent along each axis of an image of `lengths`.
    fn place(&self, k: u64, lengths: &[u64]) -> (Vec<u64>, Vec<u64>) {
        let mut rest = k;
        let mut origin = Vec::with_capacity(lengths.len());
        for (&length, &tile) in lengths.iter().zip(&self.tile) {
            let across = length.div_ceil(tile);
            origin.push(rest % across * tile);
            rest /= across;
        }
        let extent = (origin.iter().zip(lengths).zip(&self.tile))
            .map(|((&o, &l), &t)| t.min(l - o))
            .collect();
        (origin, extent)
    }

    fn read_at<S: Source + ?Sized>(
        &self,
        source: &S,
        at: u64,
        buf: &mut [u8],
    ) -> Result<(), String> {
        source
            .read_at(at, buf)
            .map_err(|e| format!("reading a compressed tile: {e}"))
    }

    /// Row `k`'s bytes at `at`, `len` of them, from `table`, the table's rows.
    fn cell<'a>(&self, table: &'a [u8], k: u64, at: u64, len: u64) -> &'a [u8] {
        let at = (k * self.row_bytes + at) as usize;
        &table[at..at + len as usize]
    }

    /// Where row `k`'s array in `column` is in the heap, from `table`, the table's rows, and its length in bytes, once
    /// it's checked. No more than `pixels` values of `wide` bytes take, compressed or not, which an eighth more and a
    /// header always cover.
    fn located(
        &self,
        table: &[u8],
        k: u64,
        column: Array,
        (pixels, wide): (usize, usize),
    ) -> Result<(u64, u64), String> {
        let (count, offset) = match column.wide {
            true => {
                let d = self.cell(table, k, column.at, 16);
                (integer(&d[..8]) as u64, integer(&d[8..]) as u64)
            }
            false => {
                let d = self.cell(table, k, column.at, 8);
                (
                    integer(&d[..4]) as u32 as u64,
                    integer(&d[4..]) as u32 as u64,
                )
            }
        };
        let bytes = count.checked_mul(width(column.kind).unwrap_or(1));
        let fits = bytes
            .and_then(|b| b.checked_add(offset))
            .is_some_and(|end| end <= self.heap_len);
        let bytes = bytes
            .filter(|_| fits)
            .ok_or(format!("tile {k}'s data is past the heap"))?;
        let values = (pixels as u64).saturating_mul(wide as u64);
        if bytes > values.saturating_add(values / 8).saturating_add(64) {
            return Err(format!("tile {k} has {bytes} bytes for {pixels} values"));
        }
        Ok((offset, bytes))
    }

    /// The bytes of row `k`'s array in `column`, from the heap of the table at `start`, whose rows are `table`.
    fn array<S: Source + ?Sized>(
        &self,
        source: &S,
        (start, table): (u64, &[u8]),
        k: u64,
        column: Array,
        values: (usize, usize),
    ) -> Result<Vec<u8>, String> {
        let (offset, bytes) = self.located(table, k, column, values)?;
        let mut buf = gathered(std::iter::repeat_n(0, bytes as usize), bytes as usize)?;
        self.read_at(source, start + self.heap + offset, &mut buf)?;
        Ok(buf)
    }

    /// The width of the values in COMPRESSED_DATA, and in the columns of tiles kept whole, for an image of `bitpix`.
    fn widths(&self, bitpix: i64) -> (usize, usize) {
        let lossless = bitpix.unsigned_abs() as usize / 8;
        let data = match (self.compression, self.quantise) {
            (Compression::Rice, _) => self.bytepix,
            (_, Quantise::Lossless) => lossless,
            _ => 4,
        };
        (data, lossless)
    }

    /// Whether tile `k`'s arrays are all where they can be, before anything is allocated for the image.
    fn check(&self, table: &[u8], k: u64, pixels: usize, bitpix: i64) -> Result<(), String> {
        let (data, lossless) = self.widths(bitpix);
        if self.located(table, k, self.data, (pixels, data))?.1 > 0 {
            return Ok(());
        }
        let gzipped =
            (self.gzipped).map(|gzipped| self.located(table, k, gzipped, (pixels, lossless)));
        let whole = (self.whole).map(|whole| {
            let wide = width(whole.kind).unwrap_or(1) as usize;
            self.located(table, k, whole, (pixels, wide))
        });
        let mut found = false;
        for located in gzipped.into_iter().chain(whole) {
            found |= located?.1 > 0;
        }
        found.then_some(()).ok_or(format!("tile {k} has no data"))
    }

    fn field(&self, table: &[u8], k: u64, field: Option<Field>) -> Option<f64> {
        match field? {
            Field::Value(v) => Some(v),
            Field::Column { at, kind } => Some(number(
                kind,
                self.cell(table, k, at, width(kind).unwrap_or(8)),
            )),
        }
    }

    /// Tile `k` of the table at `start`, whose rows are `table`, in the image's units: `bitpix`'s values, scaled by
    /// `bscale` and `bzero`, with `blank` and quantised nulls NaN.
    fn tile<S: Source + ?Sized>(
        &self,
        source: &S,
        start: (u64, &[u8]),
        k: u64,
        pixels: usize,
        bitpix: i64,
        (bscale, bzero, blank): (f64, f64, Option<i64>),
    ) -> Result<Vec<f32>, String> {
        let (data_wide, lossless) = self.widths(bitpix);
        let mut bytes = self.array(source, start, k, self.data, (pixels, data_wide))?;
        let (mut compression, mut quantise, mut kind) = (self.compression, self.quantise, None);
        if bytes.is_empty() {
            // A tile fpack couldn't quantise, kept whole in whichever column has it.
            let gzipped = (self.gzipped)
                .map(|gzipped| self.array(source, start, k, gzipped, (pixels, lossless)))
                .transpose()?
                .filter(|bytes| !bytes.is_empty());
            (bytes, compression, kind) = match (gzipped, self.whole) {
                (Some(gzipped), _) => (gzipped, Compression::Gzip, None),
                (None, Some(whole)) => {
                    let wide = width(whole.kind).unwrap_or(1) as usize;
                    let bytes = self.array(source, start, k, whole, (pixels, wide))?;
                    (bytes, Compression::None, Some(whole.kind))
                }
                (None, None) => return Err(format!("tile {k} has no data")),
            };
            quantise = Quantise::Lossless;
        }
        // The values' width, and whether they're floats.
        let (wide, float) = match (kind, quantise) {
            (Some(kind), _) => (
                width(kind).unwrap_or(1) as usize,
                matches!(kind, b'E' | b'D'),
            ),
            (None, Quantise::Lossless) => (bitpix.unsigned_abs() as usize / 8, bitpix < 0),
            (None, _) => (4, false),
        };
        let wide = if compression == Compression::Rice {
            self.bytepix
        } else {
            wide
        };
        // Its gzip header and RICE_1's first value and block codes take a few bytes more.
        let room = (bytes.len() as u64)
            .saturating_mul(MOST_SHRUNK)
            .saturating_add(64);
        let claimed = ((pixels as u64).checked_mul(wide as u64))
            .filter(|&claimed| claimed <= room)
            .and_then(|claimed| usize::try_from(claimed).ok());
        let claimed = claimed.ok_or(format!(
            "tile {k} claims more than its {} bytes could hold",
            bytes.len()
        ))?;
        let raw = match compression {
            Compression::Rice => {
                let ints =
                    rice(&bytes, pixels, self.block, wide).map_err(|e| format!("tile {k}: {e}"))?;
                Raw::Int(ints)
            }
            _ => {
                if matches!(compression, Compression::Gzip | Compression::Shuffled) {
                    bytes = gunzip(&bytes, claimed).map_err(|e| format!("tile {k}: {e}"))?;
                }
                if bytes.len() != claimed {
                    return Err(format!(
                        "tile {k} has {} bytes for {pixels} values",
                        bytes.len()
                    ));
                }
                if compression == Compression::Shuffled {
                    bytes = unshuffle(&bytes, wide);
                }
                let values = bytes.chunks_exact(wide);
                match float {
                    true => {
                        let kind = if wide == 4 { b'E' } else { b'D' };
                        Raw::Float(gathered(values.map(|b| number(kind, b)), pixels)?)
                    }
                    false => Raw::Int(gathered(values.map(integer), pixels)?),
                }
            }
        };
        // Each buffer goes once the next is made, so a tile holds two at most.
        drop(bytes);
        let scaled = |v: f64| (bzero + bscale * v) as f32;
        match (raw, quantise) {
            (Raw::Float(values), _) => gathered(values.into_iter().map(scaled), pixels),
            (Raw::Int(ints), Quantise::Lossless) => {
                let blank = self
                    .field(start.1, k, self.blank)
                    .map(|b| b as i64)
                    .or(blank);
                let values = ints.into_iter().map(|v| match Some(v) == blank {
                    true => f32::NAN,
                    false => scaled(v as f64),
                });
                gathered(values, pixels)
            }
            (Raw::Int(ints), quantise) => {
                let field = |field| self.field(start.1, k, field);
                let (scale, zero) = (
                    field(self.scale).unwrap_or(1.0),
                    field(self.zero).unwrap_or(0.0),
                );
                let blank = field(self.blank).map(|b| b as i64);
                let values = dequantise(&ints, k, quantise, scale, zero, blank)?;
                gathered(values.into_iter().map(|v| scaled(f64::from(v))), pixels)
            }
        }
    }

    /// `len` voxels of the image of `lengths`, whose table starts at `start`: those where each of `fixed`'s axes is at
    /// its place, each at the sum of its coordinates times `strides`. Tiles that miss them are skipped, and every tile
    /// is checked before the voxels are allocated.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read<S: Source + ?Sized>(
        &self,
        source: &S,
        start: u64,
        lengths: &[u64],
        fixed: &[(usize, u64)],
        strides: &[u64],
        bitpix: i64,
        scaling: (f64, f64, Option<i64>),
        len: usize,
    ) -> Result<Vec<f32>, String> {
        // A tile that spans planes that aren't shown would be decoded for one of them.
        if let Some(&(axis, _)) = fixed.iter().find(|&&(axis, _)| self.tile[axis] > 1) {
            return Err(format!(
                "tiles span {} planes of axis {}, which isn't shown; funpack it first",
                self.tile[axis],
                axis + 1
            ));
        }
        let wanted: Vec<u64> = (0..self.rows)
            .filter(|&k| {
                let (origin, extent) = self.place(k, lengths);
                fixed
                    .iter()
                    .all(|&(axis, at)| (origin[axis]..origin[axis] + extent[axis]).contains(&at))
            })
            .collect();
        // The rows, all of them, which `holds` has checked the file has.
        let mut table = vec![0u8; (self.rows * self.row_bytes) as usize];
        self.read_at(source, start, &mut table)?;
        let pixels = |k| self.place(k, lengths).1.iter().product::<u64>() as usize;
        for &k in &wanted {
            self.check(&table, k, pixels(k), bitpix)?;
        }
        let mut out = crate::allocate(len)?;
        let decode = |k: u64| {
            let (origin, extent) = self.place(k, lengths);
            let pixels = extent.iter().product::<u64>() as usize;
            self.tile(source, (start, &table), k, pixels, bitpix, scaling)
                .map(|values| (origin, extent, values))
        };
        let mut place =
            |(origin, extent, values): (Vec<u64>, Vec<u64>, Vec<f32>)| -> Result<(), String> {
                let row = extent[0] as usize;
                let mut at = vec![0u64; lengths.len()];
                for values in values.chunks_exact(row) {
                    let coordinate = |axis: usize| origin[axis] + at[axis];
                    if fixed.iter().all(|&(axis, place)| coordinate(axis) == place) {
                        let first: u64 = (0..lengths.len())
                            .map(|axis| coordinate(axis) * strides[axis])
                            .sum();
                        let to = out
                            .get_mut(first as usize..)
                            .and_then(|out| out.get_mut(..row));
                        to.ok_or("a tile outside the image")?
                            .copy_from_slice(values);
                    }
                    // The next row along the axes after the first.
                    for axis in 1..lengths.len() {
                        at[axis] += 1;
                        if at[axis] < extent[axis] {
                            break;
                        }
                        at[axis] = 0;
                    }
                }
                Ok(())
            };
        // Natively, a batch of tiles decoded on every core at a time, then put in place.
        #[cfg(not(target_arch = "wasm32"))]
        {
            use bevy_tasks::ParallelSlice;
            let pool = crate::all_cores();
            let pixels = self.tile.iter().product::<u64>() as usize;
            let per_task = (crate::CHUNK_VOXELS / pixels).max(1);
            // As many tiles as make a chunk for each thread, and one at a time when a tile is bigger than that.
            let batch = (crate::CHUNK_VOXELS * pool.thread_num() / pixels).max(1);
            for batch in wanted.chunks(batch) {
                let decoded = batch.par_chunk_map(&pool, per_task, |_, tiles| {
                    tiles
                        .iter()
                        .map(|&k| decode(k))
                        .collect::<Result<Vec<_>, _>>()
                });
                for tiles in decoded {
                    for tile in tiles? {
                        place(tile)?;
                    }
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        for k in wanted {
            place(decode(k)?)?;
        }
        Ok(out)
    }
}

enum Raw {
    Int(Vec<i64>),
    Float(Vec<f64>),
}

/// The `len` values, or an error where there's no memory for them, as there may not be in wasm32's.
fn gathered<T>(values: impl Iterator<Item = T>, len: usize) -> Result<Vec<T>, String> {
    let mut out = Vec::new();
    (out.try_reserve_exact(len)).map_err(|_| format!("no memory for a tile of {len} values"))?;
    out.extend(values.take(len));
    Ok(out)
}

/// A big-endian number of `kind`, a FITS column type.
fn number(kind: u8, b: &[u8]) -> f64 {
    match kind {
        b'E' => f64::from(f32::from_be_bytes(b[..4].try_into().unwrap())),
        b'D' => f64::from_be_bytes(b[..8].try_into().unwrap()),
        _ => integer(b) as f64,
    }
}

/// A big-endian integer of 1 (unsigned, as FITS has it), 2, 4 or 8 bytes.
fn integer(b: &[u8]) -> i64 {
    match b.len() {
        1 => i64::from(b[0]),
        2 => i64::from(i16::from_be_bytes(b.try_into().unwrap())),
        4 => i64::from(i32::from_be_bytes(b.try_into().unwrap())),
        _ => i64::from_be_bytes(b[..8].try_into().unwrap()),
    }
}

/// At most a byte more than the `len` expected from gzip's format, so that a gzip bomb stops there.
fn gunzip(bytes: &[u8], len: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    out.try_reserve_exact(len + 1)
        .map_err(|_| format!("no memory for a tile of {len} bytes"))?;
    let decoder = flate2::read::GzDecoder::new(bytes);
    (decoder.take(len as u64 + 1).read_to_end(&mut out)).map_err(|e| format!("gzip: {e}"))?;
    Ok(out)
}

/// GZIP_2's planes of bytes, the first byte of every value then the second and so on, back into values.
fn unshuffle(bytes: &[u8], wide: usize) -> Vec<u8> {
    let n = bytes.len() / wide;
    let mut out = vec![0u8; bytes.len()];
    for (b, plane) in bytes.chunks_exact(n.max(1)).enumerate().take(wide) {
        for (i, &byte) in plane.iter().enumerate() {
            out[i * wide + b] = byte;
        }
    }
    out
}

/// `pixels` values of `bytepix` bytes from RICE_1, as cfitsio's fits_rdecomp writes them: the first value as it is,
/// then blocks of `block` differences, each with the number of low bits it sends as they are.
fn rice(bytes: &[u8], pixels: usize, block: usize, bytepix: usize) -> Result<Vec<i64>, String> {
    let (fsbits, fsmax, bbits) = match bytepix {
        1 => (3, 6, 8),
        2 => (4, 14, 16),
        _ => (5, 25, 32),
    };
    let mask = if bbits == 32 {
        u32::MAX
    } else {
        (1u32 << bbits) - 1
    };
    let mut at = bytepix.min(bytes.len());
    // Past the end reads zeros, and is an error once the block is done.
    let next = |at: &mut usize| {
        let byte = bytes.get(*at).copied().unwrap_or(0);
        *at += 1;
        u64::from(byte)
    };
    let mut last = (bytes[..at].iter()).fold(0u32, |v, &b| v << 8 | u32::from(b));
    let (mut b, mut nbits) = (next(&mut at), 8i64);
    let mut out = Vec::new();
    (out.try_reserve_exact(pixels))
        .map_err(|_| format!("no memory for a tile of {pixels} values"))?;
    let sign = |diff: u32| {
        if diff & 1 == 0 {
            diff >> 1
        } else {
            !(diff >> 1)
        }
    };
    while out.len() < pixels {
        nbits -= fsbits;
        while nbits < 0 {
            b = b << 8 | next(&mut at);
            nbits += 8;
        }
        let fs = (b >> nbits) as i64 - 1;
        b &= (1 << nbits) - 1;
        let end = (out.len() + block).min(pixels);
        if fs < 0 {
            out.resize(end, last);
        } else if fs == fsmax {
            for _ in out.len()..end {
                let mut k = bbits - nbits;
                let mut diff = b << k;
                k -= 8;
                while k >= 0 {
                    b = next(&mut at);
                    diff |= b << k;
                    k -= 8;
                }
                if nbits > 0 {
                    b = next(&mut at);
                    diff |= b >> -k;
                    b &= (1 << nbits) - 1;
                } else {
                    b = 0;
                }
                last = last.wrapping_add(sign(diff as u32)) & mask;
                out.push(last);
            }
        } else {
            for _ in out.len()..end {
                while b == 0 {
                    if at > bytes.len() {
                        return Err("RICE_1 data cut short".into());
                    }
                    nbits += 8;
                    b = next(&mut at);
                }
                let top = 63 - b.leading_zeros() as i64;
                let zeros = nbits - top - 1;
                nbits = top;
                b ^= 1 << nbits;
                nbits -= fs;
                while nbits < 0 {
                    b = b << 8 | next(&mut at);
                    nbits += 8;
                }
                let diff = (zeros as u64) << fs | b >> nbits;
                b &= (1 << nbits) - 1;
                last = last.wrapping_add(sign(diff as u32)) & mask;
                out.push(last);
            }
        }
        if at > bytes.len() {
            return Err("RICE_1 data cut short".into());
        }
    }
    let signed = |v: u32| match bytepix {
        1 => i64::from(v as u8),
        2 => i64::from(v as u16 as i16),
        _ => i64::from(v as i32),
    };
    Ok(out.into_iter().map(signed).collect())
}

/// fpack's 10,000 random numbers for dithering, the FITS standard's generator in float as cfitsio has them.
fn randoms() -> &'static [f32] {
    static RANDOMS: OnceLock<Vec<f32>> = OnceLock::new();
    RANDOMS.get_or_init(|| {
        let (a, m) = (16807.0f64, 2147483647.0f64);
        let mut seed = 1.0f64;
        (0..10_000)
            .map(|_| {
                let temp = a * seed;
                seed = temp - m * (temp / m).trunc();
                (seed / m) as f32
            })
            .collect()
    })
}

/// Tile `k`'s quantised values as floats, NaN at `blank`, undoing fpack's dither as cfitsio's unquantize_i4r4 does.
/// fpack puts the integers near -2^31 and the zero point as far the other way, so the multiply and add are fused, as
/// cfitsio's are when its compiler fuses them, rather than lose the low bits to that cancellation.
fn dequantise(
    ints: &[i64],
    k: u64,
    quantise: Quantise,
    scale: f64,
    zero: f64,
    blank: Option<i64>,
) -> Result<Vec<f32>, String> {
    const ZERO_VALUE: i64 = -2147483646;
    let Quantise::Dither { seed, zeros } = quantise else {
        let values = (ints.iter()).map(|&v| match Some(v) == blank {
            true => f32::NAN,
            false => (v as f64).mul_add(scale, zero) as f32,
        });
        return gathered(values, ints.len());
    };
    let randoms = randoms();
    let n = randoms.len();
    let mut iseed = ((k % n as u64 + seed) % n as u64) as usize;
    let mut next = (randoms[iseed] * 500.0) as usize;
    let values = (ints.iter()).map(|&v| {
        let value = match v {
            _ if Some(v) == blank => f32::NAN,
            ZERO_VALUE if zeros => 0.0,
            _ => (v as f64 - f64::from(randoms[next]) + 0.5).mul_add(scale, zero) as f32,
        };
        next += 1;
        if next == n {
            iseed = (iseed + 1) % n;
            next = (randoms[iseed] * 500.0) as usize;
        }
        value
    });
    gathered(values, ints.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read_fits;

    #[test]
    fn fpacked_tiles_read_as_funpack_reads_them() {
        // From cfitsio 4.7's fpack -qz 4 -t 16,16: a noisy cube in 16 x 16 tiles quantised with SUBTRACTIVE_DITHER_2,
        // with two tiles of NaN that it kept gzipped instead and zeros that stay zero.
        let cube = read_fits(&include_bytes!("../testdata/cube.fits.fz")[..], None).unwrap();
        let funpacked = include_bytes!("../testdata/cube.funpacked.fits");
        let funpacked = read_fits(&funpacked[..], None).unwrap();
        assert_eq!((cube.info.dims, cube.info.hdu), ([40, 30, 4], 1));
        assert_eq!(cube.info.unit().as_deref(), Some("Jy/beam"));
        assert!(
            !cube
                .info
                .cards
                .iter()
                .any(|card| card.contains("COMPRESSED_IMAGE"))
        );
        // funpack reads the gzipped tiles back with only their first seven rows NaN and the rest zero, so they're
        // checked against the cube fpack was given, which was NaN throughout them.
        for (i, (got, want)) in cube.data.iter().zip(&funpacked.data).enumerate() {
            let [x, y, z] = [i % 40, i / 40 % 30, i / 1200];
            let same = match x < 16 && y < 16 && z < 2 {
                true => got.is_nan(),
                false => got.is_nan() && want.is_nan() || got == want,
            };
            assert!(same, "{i}: {got} against {want}");
        }
        assert_eq!(cube.info.stats.non_finite, 2 * 16 * 16);
        assert!(
            cube.data[3 * 1200 + 16 * 40 + 16..][..16]
                .iter()
                .all(|&v| v == 0.0)
        );
        assert_eq!(randoms()[9999], (1043618065.0 / 2147483647.0) as f32);

        // fpack -r of 16-bit integers with BLANK, BSCALE and BZERO, which RICE_1 keeps exactly.
        let image = include_bytes!("../testdata/image16.fits.fz");
        let read = read_fits(&image[..], None).unwrap();
        for (i, &v) in read.data.iter().enumerate() {
            let want = 10.0 + 0.5 * ((i * 37 % 2000) as f32 - 1000.0);
            assert!(if i % 97 == 0 { v.is_nan() } else { v == want }, "{i}: {v}");
        }
        // Compressions that aren't read are refused with what to do instead, and so are headers that claim more than
        // the file could hold, before anything is allocated for them.
        let edited = |cards: &[(&str, &str)]| {
            let mut bytes = image.to_vec();
            for (key, value) in cards {
                let key = format!("{key:<8}=");
                let at = (bytes.chunks(80)).position(|card| card.starts_with(key.as_bytes()));
                let card = format!("{key} {value:>20}");
                bytes[at.unwrap() * 80..][..card.len()].copy_from_slice(card.as_bytes());
            }
            bytes
        };
        let with = |cards: &[(&str, &str)]| read_fits(&edited(cards)[..], None).err().unwrap();
        let refused = with(&[("ZCMPTYPE", "'PLIO_1'")]);
        assert!(
            refused.contains("PLIO_1") && refused.contains("funpack"),
            "{refused}"
        );
        // Unless there's an image after it that can be.
        let mut then = edited(&[("ZCMPTYPE", "'PLIO_1'")]);
        then.extend(crate::tests::header(&[
            "XTENSION= 'IMAGE   '",
            "BITPIX  =                  -32",
            "NAXIS   =                    2",
            "NAXIS1  =                    3",
            "NAXIS2  =                    2",
        ]));
        then.extend(crate::tests::pad(vec![0; 24], 0));
        assert_eq!(read_fits(&then[..], None).unwrap().info.hdu, 2);
        let refused = with(&[("ZNAXIS1", "4000000"), ("ZTILE1", "4000000")]);
        assert!(
            refused.contains("larger than its data could hold"),
            "{refused}"
        );
        let refused = with(&[
            ("ZNAXIS1", "5000"),
            ("ZNAXIS2", "4000"),
            ("ZTILE1", "5000"),
            ("ZTILE2", "4000"),
        ]);
        assert!(refused.contains("voxels read at a time"), "{refused}");
        let refused = with(&[("ZTILE1", "41")]);
        assert!(refused.contains("ZTILE1 = 41"), "{refused}");
        // A tile's data pointed at the whole heap, more than its 40 values could take, which every tile could do.
        let mut bytes = image.to_vec();
        let ends: Vec<usize> = (0..bytes.len() / 80)
            .filter(|&i| bytes[i * 80..].starts_with(b"END     "))
            .collect();
        let table = ((ends[1] + 1) * 80).next_multiple_of(2880);
        let cards: Vec<String> = (bytes[2880..table].chunks(80))
            .map(|card| String::from_utf8_lossy(card).into_owned())
            .collect();
        let heap: u32 = crate::num(&cards, "PCOUNT").unwrap().unwrap();
        bytes[table..table + 4].copy_from_slice(&heap.to_be_bytes());
        let refused = read_fits(&bytes[..], None).err().unwrap();
        assert!(refused.contains("bytes for 40 values"), "{refused}");
    }
}
