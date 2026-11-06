//! The data source's messages, and those of a connection to a server, as bytes for postMessage between the viewer
//! and its worker. A message is a header of little-endian fields, its length first and padded to a whole f32, then a
//! cubelet's voxels if it carries some, so that the receiver can copy those straight into a vector of their own.

use crate::{
    Cell, Cubelet, FileInfo, Histogram, Noise, Processing, Region, Smoothing, Stats, Voxels,
    engine::{CubeletId, Event, Mode, Request},
    remote::{Budget, Command, Listing, Opened, Received},
};

pub trait Wire: Sized {
    fn write(&self, w: &mut Writer);
    fn read(r: &mut Reader) -> Option<Self>;

    /// What goes after the header.
    fn voxels(&self) -> &[f32] {
        &[]
    }
}

pub struct Writer(Vec<u8>);

pub struct Reader<'a> {
    header: &'a [u8],
    voxels: Option<TakeVoxels<'a>>,
}

/// The voxels after the header, given their count.
type TakeVoxels<'a> = Box<dyn FnOnce(usize) -> Option<Vec<f32>> + 'a>;

impl Writer {
    fn bytes(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }
}

impl Reader<'_> {
    fn bytes<const N: usize>(&mut self) -> Option<[u8; N]> {
        let (bytes, rest) = self.header.split_first_chunk()?;
        self.header = rest;
        Some(*bytes)
    }
}

/// What goes before `message.voxels()`.
pub fn header<M: Wire>(message: &M) -> Vec<u8> {
    let mut w = Writer(vec![0; 4]);
    message.write(&mut w);
    let mut bytes = w.0;
    bytes.resize(bytes.len().next_multiple_of(4), 0);
    let len = bytes.len() as u32;
    bytes[..4].copy_from_slice(&len.to_le_bytes());
    bytes
}

pub fn encode<M: Wire>(message: &M) -> Vec<u8> {
    let mut bytes = header(message);
    bytes.extend_from_slice(bytemuck::cast_slice(message.voxels()));
    bytes
}

/// The header's length, from a message's first four bytes.
pub fn header_len(message: [u8; 4]) -> usize {
    u32::from_le_bytes(message) as usize
}

/// A message from its header, with `voxels` asked for the voxels after it once their count is known.
pub fn decode<'a, M: Wire>(
    header: &'a [u8],
    voxels: impl FnOnce(usize) -> Option<Vec<f32>> + 'a,
) -> Result<M, String> {
    let mut r = Reader {
        header: header.get(4..).unwrap_or_default(),
        voxels: Some(Box::new(voxels)),
    };
    M::read(&mut r).ok_or_else(|| "a truncated or unknown message".to_string())
}

macro_rules! numbers {
    ($($t:ty),*) => {$(
        impl Wire for $t {
            fn write(&self, w: &mut Writer) {
                w.bytes(&self.to_le_bytes());
            }

            fn read(r: &mut Reader) -> Option<Self> {
                r.bytes().map(Self::from_le_bytes)
            }
        }
    )*};
}
numbers!(u8, u16, u32, u64, i32, f32, f64);

impl Wire for usize {
    fn write(&self, w: &mut Writer) {
        (*self as u64).write(w);
    }

    fn read(r: &mut Reader) -> Option<Self> {
        u64::read(r)?.try_into().ok()
    }
}

impl Wire for bool {
    fn write(&self, w: &mut Writer) {
        u8::from(*self).write(w);
    }

    fn read(r: &mut Reader) -> Option<Self> {
        Some(u8::read(r)? != 0)
    }
}

impl<T: Wire> Wire for [T; 3] {
    fn write(&self, w: &mut Writer) {
        self.iter().for_each(|v| v.write(w));
    }

    fn read(r: &mut Reader) -> Option<Self> {
        Some([T::read(r)?, T::read(r)?, T::read(r)?])
    }
}

impl<T: Wire> Wire for Vec<T> {
    fn write(&self, w: &mut Writer) {
        (self.len() as u32).write(w);
        self.iter().for_each(|v| v.write(w));
    }

    fn read(r: &mut Reader) -> Option<Self> {
        let len = u32::read(r)? as usize;
        // A corrupt length can't ask for more than there is: each item takes at least a byte, and the first few
        // thousand are all that's set aside, as an item can take far more memory than its bytes.
        let mut items = Vec::with_capacity(len.min(r.header.len()).min(1 << 12));
        for _ in 0..len {
            items.push(T::read(r)?);
        }
        Some(items)
    }
}

impl<T: Wire> Wire for Option<T> {
    fn write(&self, w: &mut Writer) {
        self.is_some().write(w);
        if let Some(v) = self {
            v.write(w);
        }
    }

    fn read(r: &mut Reader) -> Option<Self> {
        Some(if bool::read(r)? {
            Some(T::read(r)?)
        } else {
            None
        })
    }
}

impl Wire for String {
    fn write(&self, w: &mut Writer) {
        write_bytes(self.as_bytes(), w);
    }

    fn read(r: &mut Reader) -> Option<Self> {
        String::from_utf8(read_bytes(r)?).ok()
    }
}

/// Raw bytes after their length, rather than a byte at a time as a `Vec<u8>`.
fn write_bytes(bytes: &[u8], w: &mut Writer) {
    (bytes.len() as u32).write(w);
    w.bytes(bytes);
}

fn read_bytes(r: &mut Reader) -> Option<Vec<u8>> {
    let len = u32::read(r)? as usize;
    let bytes = r.header.get(..len)?;
    r.header = &r.header[len..];
    Some(bytes.to_vec())
}

impl Wire for Region {
    fn write(&self, w: &mut Writer) {
        self.min.write(w);
        self.max.write(w);
    }

    fn read(r: &mut Reader) -> Option<Self> {
        Some(Self {
            min: Wire::read(r)?,
            max: Wire::read(r)?,
        })
    }
}

impl Wire for Noise {
    fn write(&self, w: &mut Writer) {
        self.median.write(w);
        self.sigma.write(w);
    }

    fn read(r: &mut Reader) -> Option<Self> {
        Some(Self {
            median: Wire::read(r)?,
            sigma: Wire::read(r)?,
        })
    }
}

/// Normalise in the lowest bit, the smoothing in the two above it.
impl Wire for Processing {
    fn write(&self, w: &mut Writer) {
        let smoothing = match self.smoothing {
            Smoothing::Off => 0,
            Smoothing::Spectral => 1,
            Smoothing::Spatial => 2,
        };
        (u8::from(self.normalise) | smoothing << 1).write(w);
    }

    fn read(r: &mut Reader) -> Option<Self> {
        let bits = u8::read(r)?;
        let smoothing = match bits >> 1 {
            0 => Smoothing::Off,
            1 => Smoothing::Spectral,
            2 => Smoothing::Spatial,
            _ => return None,
        };
        Some(Self {
            normalise: bits & 1 != 0,
            smoothing,
        })
    }
}

/// A request or event with the id of the page's load whose file it's about.
impl<M: Wire> Wire for (u32, M) {
    fn write(&self, w: &mut Writer) {
        self.0.write(w);
        self.1.write(w);
    }

    fn read(r: &mut Reader) -> Option<Self> {
        Some((u32::read(r)?, M::read(r)?))
    }

    fn voxels(&self) -> &[f32] {
        self.1.voxels()
    }
}

impl Wire for Mode {
    fn write(&self, w: &mut Writer) {
        u8::from(*self == Mode::Mean).write(w);
    }

    fn read(r: &mut Reader) -> Option<Self> {
        Some(if bool::read(r)? {
            Mode::Mean
        } else {
            Mode::Max
        })
    }
}

impl Wire for CubeletId {
    fn write(&self, w: &mut Writer) {
        self.mode.write(w);
        self.processing.write(w);
        self.level.write(w);
        [self.x, self.y, self.z].write(w);
    }

    fn read(r: &mut Reader) -> Option<Self> {
        let (mode, processing) = (Mode::read(r)?, Processing::read(r)?);
        let (level, [x, y, z]) = (u8::read(r)?, <[u16; 3]>::read(r)?);
        Some(Self {
            mode,
            processing,
            level,
            x,
            y,
            z,
        })
    }
}

impl Wire for Request {
    fn write(&self, w: &mut Writer) {
        match self {
            Request::Levels { factors, chain } => {
                0u8.write(w);
                factors.write(w);
                chain.write(w);
            }
            Request::AddCubelets { ids } => {
                1u8.write(w);
                ids.write(w);
            }
            Request::RemoveCubelets { ids } => {
                2u8.write(w);
                ids.write(w);
            }
            Request::PixelValue { id, voxel } => {
                3u8.write(w);
                id.write(w);
                voxel.write(w);
            }
            Request::Measure {
                id,
                mode,
                processing,
                level,
                texels,
            } => {
                4u8.write(w);
                id.write(w);
                mode.write(w);
                processing.write(w);
                level.write(w);
                texels.write(w);
            }
            Request::Mask {
                id,
                mode,
                processing,
                level,
                texels,
                threshold,
            } => {
                5u8.write(w);
                id.write(w);
                mode.write(w);
                processing.write(w);
                level.write(w);
                texels.write(w);
                threshold.write(w);
            }
        }
    }

    fn read(r: &mut Reader) -> Option<Self> {
        Some(match u8::read(r)? {
            0 => Request::Levels {
                factors: Wire::read(r)?,
                chain: Wire::read(r)?,
            },
            1 => Request::AddCubelets {
                ids: Wire::read(r)?,
            },
            2 => Request::RemoveCubelets {
                ids: Wire::read(r)?,
            },
            3 => Request::PixelValue {
                id: Wire::read(r)?,
                voxel: Wire::read(r)?,
            },
            4 => Request::Measure {
                id: Wire::read(r)?,
                mode: Wire::read(r)?,
                processing: Wire::read(r)?,
                level: Wire::read(r)?,
                texels: Wire::read(r)?,
            },
            5 => Request::Mask {
                id: Wire::read(r)?,
                mode: Wire::read(r)?,
                processing: Wire::read(r)?,
                level: Wire::read(r)?,
                texels: Wire::read(r)?,
                threshold: Wire::read(r)?,
            },
            _ => return None,
        })
    }
}

impl Wire for Cubelet {
    fn write(&self, w: &mut Writer) {
        self.dims.write(w);
        self.cell.max.write(w);
        self.cell.has_nan.write(w);
        match &self.voxels {
            Voxels::AllNan => 0u8.write(w),
            Voxels::Constant(v) => {
                1u8.write(w);
                v.write(w);
            }
            Voxels::Data(_) => 2u8.write(w),
        }
    }

    fn read(r: &mut Reader) -> Option<Self> {
        let dims: [usize; 3] = Wire::read(r)?;
        let cell = Cell {
            max: Wire::read(r)?,
            has_nan: Wire::read(r)?,
        };
        let voxels = match u8::read(r)? {
            0 => Voxels::AllNan,
            1 => Voxels::Constant(Wire::read(r)?),
            2 => Voxels::Data(r.voxels.take()?(dims.iter().product())?),
            _ => return None,
        };
        Some(Self { dims, voxels, cell })
    }
}

impl Wire for Event {
    fn write(&self, w: &mut Writer) {
        match self {
            Event::Cubelet { id, cubelet } => {
                0u8.write(w);
                id.write(w);
                cubelet.write(w);
            }
            Event::PixelValue { id, value } => {
                1u8.write(w);
                id.write(w);
                value.write(w);
            }
            Event::Measured { id, noise } => {
                2u8.write(w);
                id.write(w);
                noise.write(w);
            }
            Event::Mask { id, voxels } => {
                3u8.write(w);
                id.write(w);
                voxels.write(w);
            }
            Event::MaskPart {
                id,
                origin,
                dims,
                mask,
            } => {
                4u8.write(w);
                id.write(w);
                origin.write(w);
                dims.write(w);
                write_bytes(mask, w);
            }
        }
    }

    fn voxels(&self) -> &[f32] {
        match self {
            Event::Cubelet { cubelet, .. } => match &cubelet.voxels {
                Voxels::Data(data) => data,
                _ => &[],
            },
            _ => &[],
        }
    }

    fn read(r: &mut Reader) -> Option<Self> {
        Some(match u8::read(r)? {
            0 => Event::Cubelet {
                id: Wire::read(r)?,
                cubelet: Wire::read(r)?,
            },
            1 => Event::PixelValue {
                id: Wire::read(r)?,
                value: Wire::read(r)?,
            },
            2 => Event::Measured {
                id: Wire::read(r)?,
                noise: Wire::read(r)?,
            },
            3 => Event::Mask {
                id: Wire::read(r)?,
                voxels: Wire::read(r)?,
            },
            4 => Event::MaskPart {
                id: Wire::read(r)?,
                origin: Wire::read(r)?,
                dims: Wire::read(r)?,
                mask: read_bytes(r)?,
            },
            _ => return None,
        })
    }
}

impl Wire for Stats {
    fn write(&self, w: &mut Writer) {
        self.min.write(w);
        self.max.write(w);
        self.mean.write(w);
        self.std.write(w);
        self.non_finite.write(w);
    }

    fn read(r: &mut Reader) -> Option<Self> {
        Some(Self {
            min: Wire::read(r)?,
            max: Wire::read(r)?,
            mean: Wire::read(r)?,
            std: Wire::read(r)?,
            non_finite: Wire::read(r)?,
        })
    }
}

impl Wire for FileInfo {
    fn write(&self, w: &mut Writer) {
        self.dims.write(w);
        self.axes.write(w);
        self.cards.write(w);
        self.hdu.write(w);
        self.stats.write(w);
        self.histogram.0.write(w);
        self.sigma.write(w);
    }

    fn read(r: &mut Reader) -> Option<Self> {
        Some(Self {
            dims: Wire::read(r)?,
            axes: Wire::read(r)?,
            cards: Wire::read(r)?,
            hdu: Wire::read(r)?,
            stats: Wire::read(r)?,
            histogram: Histogram(Wire::read(r)?),
            sigma: Wire::read(r)?,
        })
    }
}

impl Wire for Budget {
    fn write(&self, w: &mut Writer) {
        self.gpu_memory_limit_mb.write(w);
        self.max_depth.write(w);
    }

    fn read(r: &mut Reader) -> Option<Self> {
        Some(Self {
            gpu_memory_limit_mb: Wire::read(r)?,
            max_depth: Wire::read(r)?,
        })
    }
}

impl Wire for Command {
    fn write(&self, w: &mut Writer) {
        match self {
            Command::List(directory) => {
                0u8.write(w);
                directory.write(w);
            }
            Command::Info(path) => {
                1u8.write(w);
                path.write(w);
            }
            Command::Open {
                path,
                hdu,
                budget,
                sigmas,
            } => {
                2u8.write(w);
                path.write(w);
                hdu.write(w);
                budget.write(w);
                sigmas.write(w);
            }
            Command::Request { file_id, request } => {
                3u8.write(w);
                file_id.write(w);
                request.write(w);
            }
            Command::Close(file_id) => {
                4u8.write(w);
                file_id.write(w);
            }
            Command::ClearCache => 5u8.write(w),
        }
    }

    fn read(r: &mut Reader) -> Option<Self> {
        Some(match u8::read(r)? {
            0 => Command::List(Wire::read(r)?),
            1 => Command::Info(Wire::read(r)?),
            2 => Command::Open {
                path: Wire::read(r)?,
                hdu: Wire::read(r)?,
                budget: Wire::read(r)?,
                sigmas: Wire::read(r)?,
            },
            3 => Command::Request {
                file_id: Wire::read(r)?,
                request: Wire::read(r)?,
            },
            4 => Command::Close(Wire::read(r)?),
            5 => Command::ClearCache,
            _ => return None,
        })
    }
}

impl Wire for Listing {
    fn write(&self, w: &mut Writer) {
        self.directory.write(w);
        self.parent.write(w);
        self.directories.write(w);
        (self.files.len() as u32).write(w);
        for (name, bytes) in &self.files {
            name.write(w);
            bytes.write(w);
        }
    }

    fn read(r: &mut Reader) -> Option<Self> {
        let (directory, parent, directories) = (Wire::read(r)?, Wire::read(r)?, Wire::read(r)?);
        let len = u32::read(r)? as usize;
        let mut files = Vec::with_capacity(len.min(r.header.len()).min(1 << 12));
        for _ in 0..len {
            files.push((Wire::read(r)?, Wire::read(r)?));
        }
        Some(Self {
            directory,
            parent,
            directories,
            files,
        })
    }
}

impl Wire for Opened {
    fn write(&self, w: &mut Writer) {
        self.file_id.write(w);
        self.path.write(w);
        self.info.write(w);
        self.levels.write(w);
        self.tolerance.write(w);
    }

    fn read(r: &mut Reader) -> Option<Self> {
        Some(Self {
            file_id: Wire::read(r)?,
            path: Wire::read(r)?,
            info: Wire::read(r)?,
            levels: Wire::read(r)?,
            tolerance: Wire::read(r)?,
        })
    }
}

impl Wire for Received {
    fn write(&self, w: &mut Writer) {
        match self {
            Received::Registered { server } => {
                0u8.write(w);
                server.write(w);
            }
            Received::Listing(listing) => {
                1u8.write(w);
                listing.write(w);
            }
            Received::Info {
                path,
                dims,
                hdu,
                bytes,
            } => {
                2u8.write(w);
                path.write(w);
                dims.write(w);
                hdu.write(w);
                bytes.write(w);
            }
            Received::Opened(opened) => {
                3u8.write(w);
                opened.write(w);
            }
            Received::Event { file_id, event } => {
                4u8.write(w);
                file_id.write(w);
                event.write(w);
            }
            Received::Failed(message) => {
                5u8.write(w);
                message.write(w);
            }
            Received::OpenFailed(message) => {
                6u8.write(w);
                message.write(w);
            }
            Received::FileClosed { file_id, reason } => {
                7u8.write(w);
                file_id.write(w);
                reason.write(w);
            }
            Received::Closed(reason) => {
                8u8.write(w);
                reason.write(w);
            }
        }
    }

    fn read(r: &mut Reader) -> Option<Self> {
        Some(match u8::read(r)? {
            0 => Received::Registered {
                server: Wire::read(r)?,
            },
            1 => Received::Listing(Wire::read(r)?),
            2 => Received::Info {
                path: Wire::read(r)?,
                dims: Wire::read(r)?,
                hdu: Wire::read(r)?,
                bytes: Wire::read(r)?,
            },
            3 => Received::Opened(Box::new(Wire::read(r)?)),
            4 => Received::Event {
                file_id: Wire::read(r)?,
                event: Wire::read(r)?,
            },
            5 => Received::Failed(Wire::read(r)?),
            6 => Received::OpenFailed(Wire::read(r)?),
            7 => Received::FileClosed {
                file_id: Wire::read(r)?,
                reason: Wire::read(r)?,
            },
            8 => Received::Closed(Wire::read(r)?),
            _ => return None,
        })
    }

    fn voxels(&self) -> &[f32] {
        match self {
            Received::Event { event, .. } => event.voxels(),
            _ => &[],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::cube;

    /// Through bytes and back, read as the viewer reads a message: the header, then the voxels after it.
    fn round_trip<M: Wire + std::fmt::Debug>(message: M) {
        let bytes = encode(&message);
        let len = header_len(bytes[..4].try_into().unwrap());
        assert_eq!((len % 4, bytes.len() % 4), (0, 0));
        let voxels = |n: usize| {
            let bytes = bytes.get(len..len + 4 * n)?;
            Some(
                bytes
                    .chunks(4)
                    .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                    .collect(),
            )
        };
        let back: M = decode(&bytes[..len], voxels).unwrap();
        // Debug, so that NaNs compare equal.
        assert_eq!(format!("{back:?}"), format!("{message:?}"));
        assert!(decode::<M>(&bytes[..len - 4], voxels).is_err());
    }

    #[test]
    fn messages_survive_the_trip_and_a_short_one_is_an_error() {
        let processing = Processing {
            normalise: true,
            smoothing: Smoothing::Spatial,
        };
        let id = CubeletId::new(Mode::Mean, processing, 3, [1, 700, 2]);
        round_trip((
            9,
            Request::Levels {
                factors: vec![[1; 3], [2, 2, 1], [5, 5, 1]],
                chain: 2,
            },
        ));
        round_trip(Request::AddCubelets {
            ids: vec![
                id,
                CubeletId::new(Mode::Max, Processing::default(), 0, [0; 3]),
            ],
        });
        let texels = Region {
            min: [3, 0, 64],
            max: [618, 617, 195],
        };
        round_trip(Request::Measure {
            id: 4,
            mode: Mode::Mean,
            processing,
            level: 1,
            texels,
        });
        round_trip((
            2,
            Request::Mask {
                id: 5,
                mode: Mode::Max,
                processing: Processing::default(),
                level: 0,
                texels,
                threshold: 5.0,
            },
        ));
        round_trip(Event::Measured {
            id: 4,
            noise: Some(Noise {
                median: -2e-4,
                sigma: 6.4e-3,
            }),
        });
        round_trip(Event::Measured { id: 4, noise: None });
        round_trip(Event::Mask {
            id: 5,
            voxels: 696_888,
        });
        round_trip((
            7,
            Event::MaskPart {
                id: 5,
                origin: [61, 0, 0],
                dims: [3, 2, 1],
                mask: vec![0, 1, 1, 0, 0, 1],
            },
        ));
        round_trip(Request::RemoveCubelets { ids: vec![] });
        round_trip(Request::PixelValue {
            id: u32::MAX,
            voxel: [1237, 0, 195],
        });
        round_trip(Event::PixelValue {
            id: 3,
            value: f32::NAN,
        });
        let cell = Cell {
            max: f32::NEG_INFINITY,
            has_nan: true,
        };
        for voxels in [
            Voxels::AllNan,
            Voxels::Constant(-2.5),
            Voxels::Data(vec![1.0, f32::NAN, f32::INFINITY, -0.0, 5e-39, 7.0]),
        ] {
            let cubelet = Cubelet {
                dims: [3, 2, 1],
                voxels,
                cell,
            };
            round_trip((u32::MAX, Event::Cubelet { id, cubelet }));
        }
        let mut info = cube([4, 3, 2], vec![]).info;
        info.cards = vec![format!("{:<80}", "OBJECT  = 'M83 ñ'")];
        info.histogram = Histogram(vec![1, 5, 24]);
        info.sigma = Some(0.25);
        round_trip(info.clone());

        // A server connection's messages, between the viewer and the worker that holds the socket.
        round_trip((
            3,
            Command::Open {
                path: "folder/cube.fits".into(),
                hdu: Some(1),
                budget: Budget {
                    gpu_memory_limit_mb: 384,
                    max_depth: 2048,
                },
                sigmas: 0.5,
            },
        ));
        round_trip(Command::Request {
            file_id: -1,
            request: Request::PixelValue {
                id: 2,
                voxel: [1, 2, 3],
            },
        });
        round_trip(Command::Close(7));
        round_trip(Command::ClearCache);
        round_trip(Received::Listing(Listing {
            directory: "a/b".into(),
            parent: Some("a".into()),
            directories: vec!["c".into()],
            files: vec![("cube.fits".into(), 1 << 40)],
        }));
        round_trip(Received::Opened(Box::new(Opened {
            file_id: 2,
            path: "cube.fits".into(),
            info,
            levels: vec![[1; 3], [2, 2, 1]],
            tolerance: None,
        })));
        let cubelet = Cubelet {
            dims: [3, 2, 1],
            voxels: Voxels::Data(vec![1.0, f32::NAN, 3.0, -0.0, 5e-39, 7.0]),
            cell,
        };
        round_trip((
            1,
            Received::Event {
                file_id: 2,
                event: Event::Cubelet { id, cubelet },
            },
        ));
        round_trip(Received::FileClosed {
            file_id: 2,
            reason: "gone".into(),
        });
    }
}
