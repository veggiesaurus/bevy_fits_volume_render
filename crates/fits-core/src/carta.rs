//! CARTA's protocol, the part of it fits-core speaks: its messages, with the ones added for volumes, the 8-byte header
//! before each, and what a cube's header, statistics and histogram become in them and back.

use prost::Message;

use crate::{FileInfo, Histogram, ImageHeader, Stats, axis_lengths, engine::Mode, num, shown_axes};

// Generated from carta-protobuf, whose doc comments have tabs.
#[allow(clippy::all)]
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/carta.rs"));
}

use proto::{EntryType, EventType, HeaderEntry, SmoothingMode, StatisticsValue, StatsType};

/// The ICD version in every header, which carta_backend checks in REGISTER_VIEWER.
pub const ICD_VERSION: u16 = 30;
/// carta_backend's region and channel for the whole cube.
pub const CUBE: i32 = -2;
/// ERROR_DATA's tag when the server has closed a file without being asked, its file ID in the data: a failed open
/// after the one before was dropped to make room.
pub const FILE_CLOSED: &str = "file_closed";

impl From<Mode> for SmoothingMode {
    fn from(mode: Mode) -> Self {
        match mode {
            Mode::Max => SmoothingMode::BlockMax,
            Mode::Mean => SmoothingMode::BlockAverage,
        }
    }
}

/// A cubelet message's downsampling mode. NoSmoothing, the default, is taken as BlockMax, since only level 0 can go
/// without downsampling and it's the same in either.
pub fn mode(smoothing_mode: i32) -> Result<Mode, String> {
    match SmoothingMode::try_from(smoothing_mode) {
        Ok(SmoothingMode::BlockMax | SmoothingMode::NoSmoothing) => Ok(Mode::Max),
        Ok(SmoothingMode::BlockAverage) => Ok(Mode::Mean),
        _ => Err(format!("cubelets downsampled by mode {smoothing_mode}")),
    }
}

/// `message` after its header: the event type, the ICD version and the request ID, little-endian.
pub fn frame(event: EventType, request_id: u32, message: &impl Message) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(8 + message.encoded_len());
    bytes.extend_from_slice(&(event as u16).to_le_bytes());
    bytes.extend_from_slice(&ICD_VERSION.to_le_bytes());
    bytes.extend_from_slice(&request_id.to_le_bytes());
    message.encode_raw(&mut bytes);
    bytes
}

/// A message's header, and what follows it.
pub struct Frame<'a> {
    /// None for an event type this side doesn't know.
    pub event: Option<EventType>,
    pub icd_version: u16,
    pub request_id: u32,
    pub payload: &'a [u8],
}

pub fn unframe(bytes: &[u8]) -> Result<Frame<'_>, String> {
    let (header, payload) = bytes
        .split_first_chunk::<8>()
        .ok_or("a message shorter than its header")?;
    let event = u16::from_le_bytes([header[0], header[1]]);
    Ok(Frame {
        event: EventType::try_from(i32::from(event)).ok(),
        icd_version: u16::from_le_bytes([header[2], header[3]]),
        request_id: u32::from_le_bytes([header[4], header[5], header[6], header[7]]),
        payload,
    })
}

impl Frame<'_> {
    pub fn decode<M: Message + Default>(&self) -> Result<M, String> {
        M::decode(self.payload).map_err(|e| format!("{:?}: {e}", self.event))
    }
}

/// A cubelet's encoded coordinate in AddRequiredCubelets and RemoveRequiredCubelets: its level and its place in the
/// level's grid of cubelets, 16 bits each, as `level << 48 | z << 32 | y << 16 | x`.
pub fn cubelet_key(level: u16, [x, y, z]: [u16; 3]) -> u64 {
    u64::from(level) << 48 | u64::from(z) << 32 | u64::from(y) << 16 | u64::from(x)
}

pub fn cubelet_at(key: u64) -> (u16, [u16; 3]) {
    let at = |shift: u32| (key >> shift) as u16;
    (at(48), [at(0), at(16), at(32)])
}

/// Keywords whose cards hold text rather than a value, as far as the entries go. A HIERARCH card's long keyword
/// and value stay in its text.
const COMMENTARY: [&str; 5] = ["COMMENT", "HISTORY", "CONTINUE", "HIERARCH", ""];

/// A card per entry, as carta_backend sends them: the keyword, the value without its quotes, and the comment. An
/// undefined value is a FLOAT entry with no value and a NaN, so that it isn't taken for an empty string, and so is a
/// keyword with no value indicator, with its text as the comment.
pub fn header_entries(cards: &[String]) -> Vec<HeaderEntry> {
    cards.iter().map(|card| header_entry(card)).collect()
}

fn header_entry(card: &str) -> HeaderEntry {
    let name = card.get(..8).unwrap_or(card).trim_end().to_string();
    let field = card.get(8..).unwrap_or("");
    let Some(field) = field
        .strip_prefix("= ")
        .filter(|_| !COMMENTARY.contains(&&*name))
    else {
        if COMMENTARY.contains(&&*name) {
            return HeaderEntry {
                name,
                value: field.trim_end().to_string(),
                ..Default::default()
            };
        }
        return HeaderEntry {
            name,
            entry_type: EntryType::Float.into(),
            numeric_value: f64::NAN,
            comment: field.trim().to_string(),
            ..Default::default()
        };
    };
    let field = field.trim_start();
    let (value, comment, entry_type, numeric) = if let Some(quoted) = field.strip_prefix('\'') {
        // Up to the closing quote, with '' read as a quote.
        let (mut text, mut rest, mut chars) = (String::new(), "", quoted.char_indices());
        while let Some((i, c)) = chars.next() {
            if c == '\'' {
                if quoted[i + 1..].starts_with('\'') {
                    chars.next();
                } else {
                    rest = &quoted[i + 1..];
                    break;
                }
            }
            text.push(c);
        }
        let comment = rest.split_once('/').map_or("", |(_, c)| c);
        (text.trim_end().to_string(), comment, EntryType::String, 0.0)
    } else {
        let (value, comment) = field.split_once('/').unwrap_or((field, ""));
        let value = value.trim().to_string();
        let (entry_type, numeric) = match value.as_str() {
            "" => (EntryType::Float, f64::NAN),
            "T" => (EntryType::Int, 1.0),
            "F" => (EntryType::Int, 0.0),
            v => match (v.parse::<i64>(), v.replace(['D', 'd'], "E").parse::<f64>()) {
                (Ok(i), _) => (EntryType::Int, i as f64),
                (_, Ok(f)) => (EntryType::Float, f),
                _ => (EntryType::String, 0.0),
            },
        };
        (value, comment, entry_type, numeric)
    };
    HeaderEntry {
        name,
        value,
        entry_type: entry_type.into(),
        numeric_value: numeric,
        comment: comment.trim().to_string(),
    }
}

/// 80-character cards again, as AST and fits-core read them. Anything that isn't ASCII becomes '?', as when a file is
/// read, so that a card can be sliced anywhere.
pub fn cards(entries: &[HeaderEntry]) -> Vec<String> {
    entries
        .iter()
        .map(|entry| {
            let name: String = entry.name.chars().take(8).collect();
            let name = name.trim_end();
            // carta_backend sends a commentary or HIERARCH card whole as its name.
            let mut card = if COMMENTARY.contains(&name) {
                format!("{:<8}{}", entry.name, entry.value)
            } else {
                let value = if entry.entry_type == EntryType::String as i32 {
                    format!("{:<8}", entry.value.replace('\'', "''"))
                } else {
                    format!("{:>20}", entry.value)
                };
                let quote = if entry.entry_type == EntryType::String as i32 {
                    "'"
                } else {
                    ""
                };
                let mut card = format!("{name:<8}= {quote}{value}{quote}");
                if !entry.comment.is_empty() {
                    card.push_str(" / ");
                    card.push_str(&entry.comment);
                }
                card
            };
            card = card
                .chars()
                .map(|c| if c.is_ascii() { c } else { '?' })
                .collect();
            card.truncate(80);
            format!("{card:<80}")
        })
        .collect()
}

/// The FileInfoExtended of an image header, its axes as carta_backend numbers them, from 1.
pub fn file_info_extended(header: &ImageHeader) -> proto::FileInfoExtended {
    let naxis = num(&header.cards, "NAXIS").ok().flatten().unwrap_or(0);
    let [width, height, depth] = header.dims.map(|d| d.min(i32::MAX as usize) as i32);
    let [x, y, z] = header.axes.map(|a| a as i32 + 1);
    proto::FileInfoExtended {
        dimensions: naxis,
        width,
        height,
        depth,
        stokes: 1,
        header_entries: header_entries(&header.cards),
        axes_numbers: Some(proto::AxesNumbers {
            spatial_x: x,
            spatial_y: y,
            depth: z,
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// The whole cube's statistics, as REGION_STATS_DATA carries them.
pub fn statistics(stats: &Stats, voxels: usize) -> Vec<StatisticsValue> {
    [
        (StatsType::NumPixels, (voxels - stats.non_finite) as f64),
        (StatsType::NanCount, stats.non_finite as f64),
        (StatsType::Min, f64::from(stats.min)),
        (StatsType::Max, f64::from(stats.max)),
        (StatsType::Mean, stats.mean),
        (StatsType::Sigma, stats.std),
    ]
    .map(|(stats_type, value)| StatisticsValue {
        stats_type: stats_type.into(),
        value,
    })
    .to_vec()
}

fn stats_from(values: &[StatisticsValue]) -> Result<Stats, String> {
    let get = |wanted: StatsType| {
        values
            .iter()
            .find(|v| v.stats_type == wanted as i32)
            .map(|v| v.value)
            .ok_or(format!("no {wanted:?} in the cube's statistics"))
    };
    Ok(Stats {
        min: get(StatsType::Min)? as f32,
        max: get(StatsType::Max)? as f32,
        mean: get(StatsType::Mean)?,
        std: get(StatsType::Sigma)?,
        non_finite: get(StatsType::NanCount)? as usize,
    })
}

/// The cube's histogram, its bins from the range's minimum to its maximum, and whether a count past int32's range was
/// clamped to it.
pub fn histogram(histogram: &Histogram, stats: &Stats) -> (proto::Histogram, bool) {
    let bins = histogram.0.len();
    let width = (f64::from(stats.max) - f64::from(stats.min)) / bins as f64;
    let mut clamped = false;
    let counts = histogram.0.iter().scan(0, |below, &upto| {
        let n = upto - *below;
        *below = upto;
        clamped |= n > i32::MAX as u64;
        Some(n.min(i32::MAX as u64) as i32)
    });
    let message = proto::Histogram {
        num_bins: bins as i32,
        bin_width: width,
        first_bin_center: f64::from(stats.min) + width / 2.0,
        bins: counts.collect(),
        mean: stats.mean,
        std_dev: stats.std,
    };
    (message, clamped)
}

fn histogram_from(message: &proto::Histogram) -> Result<Histogram, String> {
    if message.bins.is_empty() || message.bins.iter().any(|&n| n < 0) {
        return Err("a histogram with no bins or a negative count".into());
    }
    let counts: Vec<u64> = message.bins.iter().map(|&n| n as u64).collect();
    Ok(Histogram::of_counts(&counts))
}

/// The dims and axes `read_fits` would read an image of `cards` with.
pub fn image_axes(cards: &[String]) -> Result<([usize; 3], [usize; 3]), String> {
    let lengths = axis_lengths(cards)?;
    let (dims, axes, _) = shown_axes(&lengths)?;
    if dims.contains(&0) {
        return Err(format!("an image of {dims:?}"));
    }
    Ok((dims, axes))
}

/// What a viewer knows of a cube it opens on a server, from OPEN_FILE_ACK and the whole cube's histogram and
/// statistics. Its dims and axes come from its header, as when a file is read.
pub fn file_info(
    ack: &proto::OpenFileAck,
    histogram: &proto::Histogram,
    statistics: &[StatisticsValue],
) -> Result<FileInfo, String> {
    let extended = ack.file_info_extended.as_ref().ok_or("no header")?;
    let cards = cards(&extended.header_entries);
    let (dims, axes) = image_axes(&cards)?;
    let hdu = (ack.file_info.as_ref())
        .and_then(|f| f.hdu_list.first()?.parse().ok())
        .unwrap_or(0);
    let sigma = (ack.volume_info.as_ref())
        .map(|v| v.noise_sigma)
        .filter(|s| s.is_finite() && *s > 0.0);
    Ok(FileInfo {
        dims,
        axes,
        cards,
        hdu,
        stats: stats_from(statistics)?,
        histogram: histogram_from(histogram)?,
        sigma,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value;

    #[test]
    fn header_cards_survive_the_trip_through_header_entries() {
        let cards: Vec<String> = [
            "SIMPLE  =                    T / conforms to FITS standard",
            "NAXIS   =                    3",
            "CRVAL3  =   2.305380000000E+11 / Hz",
            "CDELT1  =  -1.388888888889D-04",
            "OBJECT  = 'NGC 5236''s CO'     / the object's name, with a quote",
            "LONGSTR = 'a string that goes on long enough to need a continuation card, as &'",
            "CONTINUE  'here it is'",
            "HISTORY Made by hand / with a slash",
            "",
            "BUNIT   = 'K       '",
            "BLANKVAL=                      / undefined",
            "EMPTY   = ''",
            "HIERARCH ESO DET CHIP NAME = 'CCD-44' / the chip",
            "NOVALUE   with only text after it",
        ]
        .map(|card| format!("{card:<80}"))
        .to_vec();
        let back = super::cards(&header_entries(&cards));
        assert!(back.iter().all(|card| card.len() == 80));
        for key in ["SIMPLE", "NAXIS", "CRVAL3", "OBJECT", "LONGSTR", "BUNIT"] {
            assert_eq!(value(&back, key), value(&cards, key), "{key}");
        }
        assert_eq!(value(&back, "OBJECT").unwrap(), "NGC 5236's CO");
        assert!(value(&back, "LONGSTR").unwrap().ends_with("as here it is"));
        assert_eq!(
            num::<f64>(&back, "CDELT1").unwrap(),
            Some(-1.388888888889e-4)
        );
        assert_eq!(back[7].trim_end(), cards[7].trim_end());
        // An undefined value isn't an empty string, and a HIERARCH card comes back as it was.
        assert_eq!(
            (value(&back, "BLANKVAL"), value(&cards, "BLANKVAL")),
            (None, None)
        );
        assert_eq!(back[10], cards[10]);
        assert_eq!(value(&back, "EMPTY").as_deref(), Some(""));
        assert_eq!(back[12], cards[12]);
        // Nor is a keyword without a value indicator, whose text stays.
        assert_eq!(
            (value(&back, "NOVALUE"), value(&cards, "NOVALUE")),
            (None, None)
        );
        assert!(back[13].contains("with only text after it"));
        let entry = header_entry(&cards[2]);
        assert_eq!(
            (entry.entry_type, entry.numeric_value, &*entry.comment),
            (EntryType::Float as i32, 2.30538e11, "Hz")
        );
        // A server's entries can't make a card that isn't 80 ASCII characters.
        let odd = HeaderEntry {
            name: "ÉTÉ".into(),
            value: "ñ".repeat(100),
            ..Default::default()
        };
        assert!(
            super::cards(&[odd])
                .iter()
                .all(|card| card.len() == 80 && card.is_ascii())
        );
    }
}
