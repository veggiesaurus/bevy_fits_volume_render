//! The remote data source: the viewer's requests as CARTA messages to a server, and what comes back as the data
//! source's events. `Client` only turns one into the other, so the viewer's socket thread natively and the browser's
//! worker can share it, each moving the bytes its own way. It keeps the cubelets it's sent, compressed, and answers
//! from those what it can.

use std::{
    collections::{HashSet, VecDeque},
    sync::Arc,
};

use prost::Message as _;

use crate::{
    CUBELET, FileInfo,
    cache::{self, Lru, Pack, Store},
    carta::{self, CUBE, FILE_CLOSED, Frame, cubelet_key, frame, proto},
    codec,
    engine::{CubeletId, Event, Request},
};
use proto::{CompressionType, EventType, SmoothingMode};

#[cfg(not(target_arch = "wasm32"))]
pub use native::Connection;

/// What the viewer can hold, which picks the level the whole cube is shown at.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub gpu_memory_limit_mb: u64,
    pub max_depth: usize,
}

#[derive(Debug)]
pub enum Command {
    /// A directory under the server's root, "" for the root itself.
    List(String),
    Info(String),
    /// A file under the server's root, in place of the one open, and the tolerance its cubelets come at in units of
    /// its noise σ.
    Open {
        path: String,
        hdu: Option<usize>,
        budget: Budget,
        sigmas: f64,
    },
    /// A data source request for the file opened as `file_id`.
    Request {
        file_id: i32,
        request: Request,
    },
    /// The file opened as `file_id` is no longer wanted.
    Close(i32),
    /// Empties the cache, in memory and on disk.
    ClearCache,
}

#[derive(Debug)]
pub enum Received {
    Registered {
        server: String,
    },
    Listing(Listing),
    Info {
        path: String,
        dims: [usize; 3],
        hdu: usize,
        bytes: u64,
    },
    Opened(Box<Opened>),
    Event {
        file_id: i32,
        event: Event,
    },
    /// A request that failed, or an error from the server.
    Failed(String),
    /// The open asked for failed.
    OpenFailed(String),
    /// A file opened before is closed, so nothing more comes for it: the server dropped it for an open that then
    /// failed, or the viewer's levels didn't match the server's.
    FileClosed {
        file_id: i32,
        reason: String,
    },
    /// The connection is gone, and why.
    Closed(String),
}

#[derive(Debug)]
pub struct Listing {
    pub directory: String,
    /// None at the root.
    pub parent: Option<String>,
    pub directories: Vec<String>,
    /// With their sizes in bytes.
    pub files: Vec<(String, u64)>,
}

#[derive(Debug)]
pub struct Opened {
    pub file_id: i32,
    pub path: String,
    pub info: FileInfo,
    /// The level list the server cuts by, which the viewer's has to match.
    pub levels: Vec<[usize; 3]>,
    /// The bound on a level 0 voxel's error, in the cube's units, and on other levels' by `Mode::noise_ratio`;
    /// None when they come exact.
    pub tolerance: Option<f64>,
}

/// An open on its way: the ack, then the whole cube's histogram and statistics, in either order.
struct Opening {
    file_id: i32,
    path: String,
    sigmas: f64,
    ack: Option<proto::OpenFileAck>,
    histogram: Option<proto::Histogram>,
    statistics: Option<Vec<proto::StatisticsValue>>,
}

/// A cubelet's CUBELET_DATA as the server sent it, which is what the cache keeps.
type Payload = Arc<[u8]>;

/// Cubelets by what they're of and where, with the tolerance they're within.
type Memory = Lru<(Arc<str>, CubeletId), (f64, Payload)>;

struct Open {
    file_id: i32,
    levels: Vec<[usize; 3]>,
    tolerance: Option<f64>,
    /// What the file's cubelets are kept as, and the name of its pack, when the server says what the file is.
    identity: Option<(Arc<str>, String)>,
    pack: Option<Pack>,
    /// The cache was cleared before the pack was attached, so it's emptied then.
    cleared: bool,
    /// Cubelets asked for that are in the cache, for `cached`.
    hits: VecDeque<CubeletId>,
}

pub struct Client {
    next_file_id: i32,
    opening: Option<Opening>,
    open: Option<Open>,
    /// The paths of FILE_INFO requests, by request ID.
    info: Vec<(u32, String)>,
    next_request_id: u32,
    /// Messages for the server that `receive` has made, for `replies`.
    replies: Vec<Vec<u8>>,
    memory: Memory,
    /// The open file, which the viewer is told of once its pack is attached, and the pack's name until it's asked
    /// for.
    held: Option<Received>,
    pack_to_open: Option<String>,
}

/// CARTA's directories and files are separate fields.
fn split(path: &str) -> (String, String) {
    match path.rsplit_once('/') {
        Some((directory, file)) => (directory.into(), file.into()),
        None => (String::new(), path.into()),
    }
}

/// Runs of `ids` in the same mode, in order, each as its keys.
fn by_mode(ids: &[CubeletId]) -> Result<Vec<(SmoothingMode, Vec<u64>)>, String> {
    let mut runs: Vec<(SmoothingMode, Vec<u64>)> = Vec::new();
    for id in ids {
        if !id.processing.is_none() {
            return Err("processed cubelets aren't served remotely".into());
        }
        let key = cubelet_key(id.level.into(), [id.x, id.y, id.z]);
        let mode = SmoothingMode::from(id.mode);
        match runs.last_mut() {
            Some((m, keys)) if *m == mode => keys.push(key),
            _ => runs.push((mode, vec![key])),
        }
    }
    Ok(runs)
}

impl Open {
    /// Whether `id` is in the cache within the tolerance asked for, in which case it's queued for `cached`.
    fn hit(&mut self, memory: &mut Memory, id: CubeletId) -> bool {
        let Some((identity, _)) = &self.identity else {
            return false;
        };
        let tolerance = self.tolerance.unwrap_or(0.0);
        let in_memory = (memory.get(&(identity.clone(), id))).is_some_and(|e| e.0 <= tolerance);
        let hit = in_memory || self.pack.as_ref().is_some_and(|p| p.has(id, tolerance));
        if hit {
            self.hits.push_back(id);
        }
        hit
    }

    /// A cubelet from the cache, from memory or else from the pack, and then kept in both: the pack may have been made
    /// since the cubelet came.
    fn find(&mut self, memory: &mut Memory, id: CubeletId) -> Option<Payload> {
        let (identity, _) = self.identity.as_ref()?;
        let tolerance = self.tolerance.unwrap_or(0.0);
        let key = (identity.clone(), id);
        if let Some(&(within, ref payload)) = memory.get(&key).filter(|e| e.0 <= tolerance) {
            if let Some(pack) = &mut self.pack {
                pack.put(id, within, payload);
            }
            return Some(payload.clone());
        }
        let (within, payload) = self.pack.as_mut()?.get(id, tolerance)?;
        let payload: Payload = payload.into();
        memory.insert(key, (within, payload.clone()), payload.len());
        Some(payload)
    }

    /// Keeps a cubelet the server sent, `payload` being its CUBELET_DATA.
    fn keep(&mut self, memory: &mut Memory, id: CubeletId, payload: &[u8]) {
        let Some((identity, _)) = &self.identity else {
            return;
        };
        let tolerance = self.tolerance.unwrap_or(0.0);
        if let Some(pack) = &mut self.pack {
            pack.put(id, tolerance, payload);
        }
        let key = (identity.clone(), id);
        memory.insert(key, (tolerance, payload.into()), payload.len());
    }

    fn add(&self, ids: &[CubeletId]) -> Result<Vec<Vec<u8>>, String> {
        let (compression_type, tolerance) = match self.tolerance {
            Some(tolerance) => (CompressionType::ZfpExplicit, tolerance),
            None => (CompressionType::None, 0.0),
        };
        let messages = (by_mode(ids)?.into_iter()).map(|(mode, cubelets)| {
            let add = proto::AddRequiredCubelets {
                file_id: self.file_id,
                mode: mode.into(),
                cubelets,
                compression_type: compression_type.into(),
                tolerance,
            };
            frame(EventType::AddRequiredCubelets, 0, &add)
        });
        Ok(messages.collect())
    }
}

impl Client {
    /// With the REGISTER_VIEWER to send first, which carries `api_key` as the token for a server that wasn't given one
    /// when the socket connected. Up to `memory` bytes of the cubelets the server sends are kept in memory.
    pub fn new(api_key: String, memory: usize) -> (Self, Vec<u8>) {
        let client = Self {
            next_file_id: 0,
            opening: None,
            open: None,
            info: Vec::new(),
            next_request_id: 1,
            replies: Vec::new(),
            memory: Lru::new(memory),
            held: None,
            pack_to_open: None,
        };
        let register = proto::RegisterViewer {
            api_key,
            ..Default::default()
        };
        (client, frame(EventType::RegisterViewer, 0, &register))
    }

    fn request_id(&mut self) -> u32 {
        self.next_request_id += 1;
        self.next_request_id
    }

    /// The messages that carry `command`, and what the viewer is to be told of it at once.
    pub fn command(&mut self, command: Command) -> (Vec<Vec<u8>>, Option<Received>) {
        let id = self.request_id();
        let messages = match command {
            Command::List(directory) => {
                let request = proto::FileListRequest {
                    directory,
                    ..Default::default()
                };
                vec![frame(EventType::FileListRequest, id, &request)]
            }
            Command::Info(path) => {
                let (directory, file) = split(&path);
                self.info.push((id, path));
                let request = proto::FileInfoRequest {
                    directory,
                    file,
                    ..Default::default()
                };
                vec![frame(EventType::FileInfoRequest, id, &request)]
            }
            Command::Open {
                path,
                hdu,
                budget,
                sigmas,
            } => self.open(id, path, hdu, budget, sigmas),
            Command::Request { file_id, request } => return self.request(file_id, request),
            Command::Close(file_id) => {
                if self
                    .open
                    .as_ref()
                    .is_some_and(|open| open.file_id == file_id)
                {
                    self.open = None;
                }
                vec![frame(
                    EventType::CloseFile,
                    id,
                    &proto::CloseFile { file_id },
                )]
            }
            Command::ClearCache => {
                self.memory.clear();
                if let Some(open) = &mut self.open {
                    match &mut open.pack {
                        Some(pack) => pack.clear(),
                        None => open.cleared = true,
                    }
                }
                Vec::new()
            }
        };
        (messages, None)
    }

    /// OPEN_FILE, then what the viewer needs to know of the whole cube, sent together since the server answers in
    /// turn.
    fn open(
        &mut self,
        id: u32,
        path: String,
        hdu: Option<usize>,
        budget: Budget,
        sigmas: f64,
    ) -> Vec<Vec<u8>> {
        self.next_file_id += 1;
        let file_id = self.next_file_id;
        let (directory, file) = split(&path);
        let open = proto::OpenFile {
            directory,
            file,
            hdu: hdu.map_or(String::new(), |h| h.to_string()),
            file_id,
            volume_budget: Some(proto::VolumeBudget {
                gpu_memory_limit_mb: budget.gpu_memory_limit_mb.min(i64::MAX as u64) as i64,
                max_depth: budget.max_depth.min(i32::MAX as usize) as i32,
            }),
            ..Default::default()
        };
        let histogram = proto::SetHistogramRequirements {
            file_id,
            region_id: CUBE,
            histograms: vec![proto::HistogramConfig {
                channel: CUBE,
                num_bins: -1,
                ..Default::default()
            }],
        };
        use proto::StatsType::*;
        let stats = proto::SetStatsRequirements {
            file_id,
            region_id: CUBE,
            stats_configs: vec![proto::set_stats_requirements::StatsConfig {
                coordinate: "z".into(),
                stats_types: [NumPixels, NanCount, Min, Max, Mean, Sigma]
                    .map(i32::from)
                    .to_vec(),
            }],
        };
        self.opening = Some(Opening {
            file_id,
            path,
            sigmas,
            ack: None,
            histogram: None,
            statistics: None,
        });
        vec![
            frame(EventType::OpenFile, id, &open),
            frame(EventType::SetHistogramRequirements, id, &histogram),
            frame(EventType::SetStatsRequirements, id, &stats),
        ]
    }

    fn request(&mut self, file_id: i32, request: Request) -> (Vec<Vec<u8>>, Option<Received>) {
        // Cut at other factors, the server's cubelets would land in the wrong places, so the file goes.
        if let Request::Levels { factors, .. } = &request
            && let Some(open) =
                (self.open.as_ref()).filter(|o| o.file_id == file_id && o.levels != *factors)
        {
            let reason = format!(
                "the server cuts by levels {:?}, not the viewer's {factors:?}",
                open.levels
            );
            self.open = None;
            let close = frame(EventType::CloseFile, 0, &proto::CloseFile { file_id });
            return (vec![close], Some(Received::FileClosed { file_id, reason }));
        }
        match self.messages(file_id, request) {
            Ok(messages) => (messages, None),
            Err(e) => (Vec::new(), Some(Received::Failed(e))),
        }
    }

    fn messages(&mut self, file_id: i32, request: Request) -> Result<Vec<Vec<u8>>, String> {
        // A request for a file opened before, from a view that's on its way out.
        let Some(open) = self.open.as_mut().filter(|open| open.file_id == file_id) else {
            return Ok(Vec::new());
        };
        Ok(match request {
            Request::Levels { .. } => Vec::new(),
            Request::AddCubelets { ids } => {
                let ids: Vec<_> = (ids.into_iter())
                    .filter(|&id| !open.hit(&mut self.memory, id))
                    .collect();
                open.add(&ids)?
            }
            Request::RemoveCubelets { ids } => {
                let removed: HashSet<_> = ids.iter().collect();
                open.hits.retain(|id| !removed.contains(id));
                (by_mode(&ids)?.into_iter())
                    .map(|(mode, cubelets)| {
                        let remove = proto::RemoveRequiredCubelets {
                            file_id,
                            mode: mode.into(),
                            cubelets,
                        };
                        frame(EventType::RemoveRequiredCubelets, 0, &remove)
                    })
                    .collect()
            }
            // The request ID pairs the answer with it.
            Request::PixelValue { id, voxel } => {
                let [x, y, z] = voxel.map(|v| v.min(i32::MAX as usize) as i32);
                let request = proto::PixelValueRequest {
                    file_id,
                    x,
                    y,
                    z,
                    stokes: 0,
                };
                vec![frame(EventType::PixelValueRequest, id, &request)]
            }
            Request::Measure { .. } | Request::Mask { .. } => {
                return Err("noise and masks aren't served remotely".into());
            }
        })
    }

    /// The messages `receive` and `cached` have made for the server since this was last called, to be sent after
    /// what was received.
    pub fn replies(&mut self) -> Vec<Vec<u8>> {
        std::mem::take(&mut self.replies)
    }

    /// Whether `cached` has cubelets to decode.
    pub fn has_cached(&self) -> bool {
        self.open.as_ref().is_some_and(|open| !open.hits.is_empty())
    }

    /// The next of the cubelets asked for that are in the cache, read and decoded, which is slow enough to go one at a
    /// time. One that has gone from the cache since, or doesn't decode, is asked of the server instead, in `replies`.
    pub fn cached(&mut self) -> Option<Received> {
        let open = self.open.as_mut()?;
        while let Some(id) = open.hits.pop_front() {
            let payload = open.find(&mut self.memory, id).ok_or_else(String::new);
            let data =
                payload.and_then(|p| proto::CubeletData::decode(&p[..]).map_err(|e| e.to_string()));
            match data.and_then(|data| codec::decode(&data)) {
                Ok(cubelet) => {
                    let event = Event::Cubelet { id, cubelet };
                    let file_id = open.file_id;
                    return Some(Received::Event { file_id, event });
                }
                Err(_) => {
                    if let Some(pack) = &mut open.pack {
                        pack.forget(id);
                    }
                    self.replies.extend(open.add(&[id]).unwrap_or_default());
                }
            }
        }
        None
    }

    /// The name of the open file's pack, if it has one.
    pub fn pack(&self) -> Option<&str> {
        let open = self.open.as_ref()?;
        open.identity.as_ref().map(|(_, name)| name.as_str())
    }

    /// The pack to open for the file just opened, once: the viewer is told of the file when `attach` has it.
    pub fn pack_to_open(&mut self) -> Option<String> {
        self.pack_to_open.take()
    }

    /// The pack `name` for the open file: the store it's in and how long it may grow, or None when it can't be kept.
    /// Returns the open, for the viewer, unless the file has gone since.
    pub fn attach(&mut self, name: &str, store: Option<(Box<dyn Store>, u64)>) -> Option<Received> {
        let open = self.open.as_mut()?;
        let (identity, wanted) = open.identity.as_ref()?;
        if wanted != name || self.held.is_none() {
            return None;
        }
        open.pack = store.map(|(store, room)| Pack::open(store, identity, room));
        if let (Some(pack), true) = (&mut open.pack, open.cleared) {
            pack.clear();
        }
        self.held.take()
    }

    /// What a message from the server means for the viewer. An error ends the connection: a message that doesn't make
    /// sense, after which nothing from that server can be trusted, or a registration it refused.
    pub fn receive(&mut self, bytes: &[u8]) -> Result<Vec<Received>, String> {
        let message = carta::unframe(bytes)?;
        let Some(event) = message.event else {
            return Ok(Vec::new());
        };
        Ok(match event {
            EventType::RegisterViewerAck => {
                let ack: proto::RegisterViewerAck = message.decode()?;
                if !ack.success {
                    return Err(format!("not registered: {}", ack.message));
                }
                if ack.server_feature_flags & proto::ServerFeatureFlags::Volumes as u32 == 0 {
                    return Err("the server doesn't serve volumes".into());
                }
                let server = ack.platform_strings.get("server").cloned();
                vec![Received::Registered {
                    server: server.unwrap_or_default(),
                }]
            }
            EventType::FileListResponse => {
                let list: proto::FileListResponse = message.decode()?;
                if !list.success {
                    return Ok(vec![Received::Failed(list.message)]);
                }
                vec![Received::Listing(Listing {
                    parent: (!list.directory.is_empty()).then_some(list.parent),
                    directory: list.directory,
                    directories: list.subdirectories.into_iter().map(|d| d.name).collect(),
                    files: (list.files.into_iter())
                        .map(|f| (f.name, f.size.max(0) as u64))
                        .collect(),
                })]
            }
            EventType::FileInfoResponse => self.file_info(&message)?,
            EventType::OpenFileAck => {
                let ack: proto::OpenFileAck = message.decode()?;
                match self.opening.as_mut().filter(|o| o.file_id == ack.file_id) {
                    Some(_) if !ack.success => {
                        self.opening = None;
                        vec![Received::OpenFailed(ack.message)]
                    }
                    Some(opening) => {
                        opening.ack = Some(ack);
                        self.opened()?
                    }
                    None => Vec::new(),
                }
            }
            EventType::RegionHistogramData => {
                let data: proto::RegionHistogramData = message.decode()?;
                match self.opening.as_mut().filter(|o| o.file_id == data.file_id) {
                    Some(opening) if data.region_id == CUBE && data.channel == CUBE => {
                        opening.histogram = data.histograms;
                        self.opened()?
                    }
                    _ => Vec::new(),
                }
            }
            EventType::RegionStatsData => {
                let data: proto::RegionStatsData = message.decode()?;
                match self.opening.as_mut().filter(|o| o.file_id == data.file_id) {
                    Some(opening) if data.region_id == CUBE => {
                        opening.statistics = Some(data.statistics);
                        self.opened()?
                    }
                    _ => Vec::new(),
                }
            }
            EventType::CubeletData => {
                let data: proto::CubeletData = message.decode()?;
                let Some(open) = self.open.as_mut().filter(|o| o.file_id == data.file_id) else {
                    return Ok(Vec::new());
                };
                let level = usize::try_from(data.level)
                    .ok()
                    .filter(|&l| l < open.levels.len());
                let at = [data.x, data.y, data.z].map(|v| u16::try_from(v).ok());
                let (Some(level), [Some(x), Some(y), Some(z)]) = (level, at) else {
                    return Err(format!(
                        "a cubelet at {:?}",
                        (data.level, data.x, data.y, data.z)
                    ));
                };
                let mode = carta::mode(data.mode)?;
                let id =
                    CubeletId::new(mode, Default::default(), level, [x, y, z].map(usize::from));
                let cubelet = codec::decode(&data)?;
                open.keep(&mut self.memory, id, message.payload);
                vec![Received::Event {
                    file_id: data.file_id,
                    event: Event::Cubelet { id, cubelet },
                }]
            }
            EventType::PixelValueResponse => {
                let value: proto::PixelValueResponse = message.decode()?;
                vec![Received::Event {
                    file_id: value.file_id,
                    event: Event::PixelValue {
                        id: message.request_id,
                        value: value.value,
                    },
                }]
            }
            EventType::ErrorData => {
                let error: proto::ErrorData = message.decode()?;
                let closed = (error.tags.iter().any(|tag| tag == FILE_CLOSED))
                    .then(|| error.data.parse().ok())
                    .flatten();
                match closed {
                    Some(file_id) => {
                        if self.open.as_ref().is_some_and(|o| o.file_id == file_id) {
                            self.open = None;
                        }
                        vec![Received::FileClosed {
                            file_id,
                            reason: error.message,
                        }]
                    }
                    None => vec![Received::Failed(error.message)],
                }
            }
            _ => Vec::new(),
        })
    }

    fn file_info(&mut self, message: &Frame) -> Result<Vec<Received>, String> {
        let response: proto::FileInfoResponse = message.decode()?;
        let Some(at) = self
            .info
            .iter()
            .position(|(id, _)| *id == message.request_id)
        else {
            return Ok(Vec::new());
        };
        let path = self.info.remove(at).1;
        if !response.success {
            return Ok(vec![Received::Failed(response.message)]);
        }
        let file = response.file_info.unwrap_or_default();
        let hdu = file.hdu_list.first().cloned().unwrap_or_default();
        let extended = response.file_info_extended.get(&hdu).ok_or("no header")?;
        let cards = carta::cards(&extended.header_entries);
        let (dims, _) = carta::image_axes(&cards)?;
        Ok(vec![Received::Info {
            path,
            dims,
            hdu: hdu.parse().unwrap_or(0),
            bytes: file.size.max(0) as u64,
        }])
    }

    /// The open, once all of it is in.
    fn opened(&mut self) -> Result<Vec<Received>, String> {
        let Some(Opening {
            ack: Some(_),
            histogram: Some(_),
            statistics: Some(_),
            ..
        }) = &self.opening
        else {
            return Ok(Vec::new());
        };
        let opening = self.opening.take().unwrap();
        let (ack, histogram, statistics) = (
            opening.ack.unwrap(),
            opening.histogram.unwrap(),
            opening.statistics.unwrap(),
        );
        let info = carta::file_info(&ack, &histogram, &statistics)?;
        let volume = ack.volume_info.ok_or("no volume info")?;
        // The server dropped the file before for this one, which goes as well.
        if volume.cubelet_size != CUBELET as i32 {
            let reason = format!(
                "unsupported cubelet size: the server's cubelets are {}³, the viewer's {CUBELET}³",
                volume.cubelet_size
            );
            let close = proto::CloseFile {
                file_id: opening.file_id,
            };
            self.replies.push(frame(EventType::CloseFile, 0, &close));
            let mut told: Vec<_> = (self.open.take().into_iter())
                .map(|open| Received::FileClosed {
                    file_id: open.file_id,
                    reason: reason.clone(),
                })
                .collect();
            told.push(Received::OpenFailed(reason));
            return Ok(told);
        }
        let levels: Vec<[usize; 3]> = (volume.levels.iter())
            .map(|f| [f.x, f.y, f.z].map(|f| usize::try_from(f).ok().filter(|&f| f > 0)))
            .map(|f| Some([f[0]?, f[1]?, f[2]?]))
            .collect::<Option<_>>()
            .filter(|levels: &Vec<_>| levels.first() == Some(&[1; 3]))
            .ok_or("a level list that doesn't start at level 0")?;
        let tolerance = info.sigma.map(|sigma| opening.sigmas * sigma);
        let identity = (!volume.content_id.is_empty()).then(|| {
            let identity = cache::identity(&volume.content_id, &levels);
            let name = cache::pack_name(&identity);
            (identity.into(), name)
        });
        self.pack_to_open = identity.as_ref().map(|(_, name)| name.clone());
        self.open = Some(Open {
            file_id: opening.file_id,
            levels: levels.clone(),
            tolerance,
            identity,
            pack: None,
            cleared: false,
            hits: VecDeque::new(),
        });
        let opened = Received::Opened(Box::new(Opened {
            file_id: opening.file_id,
            path: opening.path,
            info,
            levels,
            tolerance,
        }));
        if self.pack_to_open.is_none() {
            self.held = None;
            return Ok(vec![opened]);
        }
        self.held = Some(opened);
        Ok(Vec::new())
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::{
        collections::VecDeque,
        io::{self, ErrorKind, Read, Write},
        net::{Shutdown, TcpStream},
        path::PathBuf,
        sync::{
            Arc,
            mpsc::{self, RecvTimeoutError, TryRecvError},
        },
        time::Duration,
    };

    use tungstenite::{
        Message, WebSocket,
        error::UrlError,
        protocol::{Role, WebSocketConfig},
        stream::MaybeTlsStream,
    };

    use super::{Client, Command, Received};
    use crate::cache::{Folder, Store};

    /// A message from the server at most: a cubelet is a MiB and a bit.
    const MAX_MESSAGE: usize = 64 << 20;
    /// How long a write waits for the server to take more, before the thread reads again.
    const POLL: Duration = Duration::from_millis(5);

    /// What the socket's thread wakes for, so that it sleeps while there's nothing to do.
    enum Wake {
        Command(Command),
        /// What the reader thread read, empty at the end of the stream.
        Read(io::Result<Vec<u8>>),
        /// The last `Connection` has gone.
        Quit,
    }

    /// A server's socket on a thread of its own, which connects, registers and then sends commands and hands what
    /// comes back to a callback, until the connection closes.
    #[derive(Clone)]
    pub struct Connection(Arc<Waker>);

    struct Waker(mpsc::Sender<Wake>);

    impl Drop for Waker {
        fn drop(&mut self) {
            let _ = self.0.send(Wake::Quit);
        }
    }

    impl Connection {
        /// `url` is `ws://host:port/`, with `?token=...` or else the token as `api_key`. Up to `memory` bytes of the
        /// cubelets sent are kept in memory, and with `folder` on disk there. `receive` is called on the connection's
        /// thread, last with `Received::Closed`, once.
        pub fn connect(
            url: &str,
            api_key: String,
            memory: usize,
            folder: Option<PathBuf>,
            mut receive: impl FnMut(Received) + Send + 'static,
        ) -> Self {
            let (wake, rx) = mpsc::channel();
            let (url, reader) = (url.to_string(), wake.clone());
            std::thread::spawn(move || {
                let client = Client::new(api_key, memory);
                let folder = folder.map(Folder::new);
                let reason = run(&url, client, folder, reader, &rx, &mut receive).err();
                receive(Received::Closed(reason.unwrap_or_else(|| "closed".into())));
            });
            Self(Arc::new(Waker(wake)))
        }

        pub fn send(&self, command: Command) {
            // Once the thread is gone, the viewer has been told.
            let _ = self.0.0.send(Wake::Command(command));
        }
    }

    /// The socket as tungstenite sees it: reads from what the reader thread has read, and writes that give up after
    /// `POLL` with what they couldn't write still queued.
    struct Stream {
        tcp: TcpStream,
        read: VecDeque<u8>,
        ended: bool,
    }

    impl Read for Stream {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            match self.read.is_empty() && !self.ended {
                true => Err(ErrorKind::WouldBlock.into()),
                false => self.read.read(buf),
            }
        }
    }

    impl Write for Stream {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.tcp.write(buf)
        }

        fn flush(&mut self) -> io::Result<()> {
            self.tcp.flush()
        }
    }

    /// A read or write that ran out of time or data, which leaves what it was writing queued.
    fn waiting(e: &tungstenite::Error) -> bool {
        matches!(e, tungstenite::Error::Io(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut))
    }

    fn written(result: tungstenite::Result<()>) -> Result<(), String> {
        match result {
            Err(e) if !waiting(&e) => Err(e.to_string()),
            _ => Ok(()),
        }
    }

    fn run(
        url: &str,
        client: (Client, Vec<u8>),
        folder: Option<Folder>,
        reader: mpsc::Sender<Wake>,
        wakes: &mpsc::Receiver<Wake>,
        receive: &mut impl FnMut(Received),
    ) -> Result<(), String> {
        let config = WebSocketConfig::default()
            .max_message_size(Some(MAX_MESSAGE))
            .max_frame_size(Some(MAX_MESSAGE));
        // Errors leave the URL out, since it has the token in it.
        let (socket, _) = tungstenite::client::connect_with_config(url, Some(config), 0).map_err(
            |e| match e {
                tungstenite::Error::Url(UrlError::UnableToConnect(_)) => {
                    "couldn't connect to the server".into()
                }
                e => format!("connecting: {e}"),
            },
        )?;
        let MaybeTlsStream::Plain(tcp) = socket.into_inner() else {
            return Err("only ws:// server URLs are supported".into());
        };
        tcp.set_write_timeout(Some(POLL))
            .map_err(|e| e.to_string())?;
        let mut reading = tcp.try_clone().map_err(|e| e.to_string())?;
        std::thread::spawn(move || {
            let mut buf = vec![0; 1 << 16];
            loop {
                let read = match reading.read(&mut buf) {
                    Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                    read => read.map(|n| buf[..n].to_vec()),
                };
                let more = matches!(&read, Ok(bytes) if !bytes.is_empty());
                if reader.send(Wake::Read(read)).is_err() || !more {
                    return;
                }
            }
        });
        // The server sends nothing before REGISTER_VIEWER, so the handshake left nothing of its unread.
        let stream = Stream {
            tcp,
            read: VecDeque::new(),
            ended: false,
        };
        let mut socket = WebSocket::from_raw_socket(stream, Role::Client, Some(config));
        let result = serve(&mut socket, client, folder.as_ref(), wakes, receive);
        let _ = socket.close(None);
        let _ = socket.flush();
        // Which ends the reader thread too.
        let _ = socket.get_ref().tcp.shutdown(Shutdown::Both);
        result
    }

    /// Asleep until there's a command to send or something the server sent, or while a write waits on the server,
    /// until it might take more. Cubelets from the cache are decoded while there's nothing else to do.
    fn serve(
        socket: &mut WebSocket<Stream>,
        (mut client, register): (Client, Vec<u8>),
        folder: Option<&Folder>,
        wakes: &mpsc::Receiver<Wake>,
        receive: &mut impl FnMut(Received),
    ) -> Result<(), String> {
        written(socket.write(Message::binary(register)))?;
        loop {
            let unwritten = match socket.flush() {
                Err(e) if waiting(&e) => true,
                result => result.map(|()| false).map_err(|e| e.to_string())?,
            };
            let wake = if client.has_cached() {
                match wakes.try_recv() {
                    Ok(wake) => wake,
                    Err(TryRecvError::Empty) => {
                        client.cached().into_iter().for_each(&mut *receive);
                        for message in client.replies() {
                            written(socket.write(Message::binary(message)))?;
                        }
                        continue;
                    }
                    Err(TryRecvError::Disconnected) => return Ok(()),
                }
            } else if unwritten {
                match wakes.recv_timeout(POLL) {
                    Ok(wake) => wake,
                    Err(RecvTimeoutError::Timeout) => continue,
                    Err(RecvTimeoutError::Disconnected) => return Ok(()),
                }
            } else {
                wakes.recv().map_err(|_| "closed".to_string())?
            };
            let mut messages = Vec::new();
            match wake {
                Wake::Command(command) => {
                    if let (Command::ClearCache, Some(folder)) = (&command, folder) {
                        folder.clear(client.pack());
                    }
                    let (sent, told) = client.command(command);
                    told.into_iter().for_each(&mut *receive);
                    messages = sent;
                }
                Wake::Read(Ok(bytes)) => {
                    let stream = socket.get_mut();
                    stream.ended |= bytes.is_empty();
                    stream.read.extend(bytes);
                    loop {
                        match socket.read() {
                            Ok(Message::Binary(bytes)) => {
                                client.receive(&bytes)?.into_iter().for_each(&mut *receive);
                                if let Some(name) = client.pack_to_open() {
                                    let store = (folder.and_then(|folder| folder.open(&name))).map(
                                        |(file, room)| (Box::new(file) as Box<dyn Store>, room),
                                    );
                                    client
                                        .attach(&name, store)
                                        .into_iter()
                                        .for_each(&mut *receive);
                                }
                                messages.extend(client.replies());
                            }
                            Ok(Message::Close(_)) => {
                                return Err("the server closed the connection".into());
                            }
                            Ok(_) => {}
                            Err(e) if waiting(&e) => break,
                            Err(e) => return Err(e.to_string()),
                        }
                    }
                }
                Wake::Read(Err(e)) => return Err(e.to_string()),
                Wake::Quit => return Ok(()),
            }
            for message in messages {
                written(socket.write(Message::binary(message)))?;
            }
        }
    }
}
