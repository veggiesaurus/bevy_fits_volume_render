//! FITS cubes under a root folder, served as cubelets over a subset of CARTA's protocol with the volume messages added
//! to it. A session has one cube open at a time and holds all of it, cutting cubelets through fits-core's engine on
//! blocking threads, and the cubelets it compresses are kept for every session in one cache.

use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    fs::{File, Metadata},
    panic::AssertUnwindSafe,
    path::{Component, Path, PathBuf},
    sync::{
        Arc, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    time::{Duration, Instant, UNIX_EPOCH},
};

use axum::{
    Router,
    body::Bytes,
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade, rejection::WebSocketUpgradeRejection},
    },
    http::{HeaderMap, StatusCode, Uri},
    response::{IntoResponse, Response},
    routing::get,
};
use fits_core::{
    self as fits, CUBELET, FileInfo,
    carta::{self, CUBE, FILE_CLOSED, Frame, ICD_VERSION, cubelet_at, frame, proto},
    codec,
    engine::{CubeletId, Engine, Event, Mode, Request},
};
use futures_util::{SinkExt, Stream, StreamExt};
use prost::Message as _;
use proto::{CompressionType, EventType, SmoothingMode};
use tokio::{
    net::TcpListener,
    sync::{Semaphore, mpsc},
    task::{JoinHandle, spawn_blocking},
};

/// The largest message a client may send: a level's worth of cubelets for a big cube, many times over.
const MAX_REQUEST: usize = 4 << 20;
/// How long a client has to register, since one without a token in the request can connect.
const REGISTER_WITHIN: Duration = Duration::from_secs(10);
/// Messages that arrive while a file is opening are held until it's open. Once this many are held, or `MAX_HELD_BYTES`
/// of them, the session stops reading the socket until the open is done, and the client's further messages wait there.
const MAX_HELD: usize = 1024;
const MAX_HELD_BYTES: usize = 2 * MAX_REQUEST;
/// Compressed cubelets kept for every session.
const CACHE_BYTES: usize = 1 << 30;
/// A client that doesn't say what it can hold gets the viewer's defaults.
const DEFAULT_BUDGET: proto::VolumeBudget = proto::VolumeBudget {
    gpu_memory_limit_mb: 384,
    max_depth: 2048,
};

struct Server {
    /// Canonical, so that a resolved path can be checked against it.
    root: PathBuf,
    token: String,
    cache: Mutex<Cache>,
    /// read_fits reads on every core, so one read at a time.
    opens: Arc<Semaphore>,
    sessions: AtomicU32,
    /// Jobs a session has out at a time.
    workers: usize,
}

/// Serves the cubes under `root` until the listener fails. A client connects to `/` with `token` as `?token=`, as
/// `Authorization: Bearer`, or in the `carta-auth-token` header.
pub async fn serve(listener: TcpListener, root: &Path, token: String) -> std::io::Result<()> {
    if token.is_empty() {
        return Err(std::io::Error::other("an empty token would let anyone in"));
    }
    let server = Arc::new(Server {
        root: root.canonicalize()?,
        token,
        cache: Mutex::new(Cache::default()),
        opens: Arc::new(Semaphore::new(1)),
        sessions: AtomicU32::new(0),
        workers: std::thread::available_parallelism().map_or(4, |n| n.get()),
    });
    let app = Router::new().route("/", get(upgrade)).with_state(server);
    axum::serve(listener, app).await
}

/// A client gives the token in the request, or else as REGISTER_VIEWER's api_key once it's connected, since a browser
/// can't set headers and a URL can end up in its console. A wrong token in the request is refused at once.
async fn upgrade(
    State(server): State<Arc<Server>>,
    headers: HeaderMap,
    uri: Uri,
    upgrade: Result<WebSocketUpgrade, WebSocketUpgradeRejection>,
) -> Response {
    let given = given_tokens(&uri, &headers);
    let authorised = (given.iter()).any(|given| same(given.as_bytes(), server.token.as_bytes()));
    if !given.is_empty() && !authorised {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    match upgrade {
        Ok(upgrade) => upgrade
            .max_message_size(MAX_REQUEST)
            .max_frame_size(MAX_REQUEST)
            .on_upgrade(move |socket| run(server, socket, authorised)),
        Err(rejection) => rejection.into_response(),
    }
}

/// The tokens in the request, as `?token=`, `Authorization: Bearer` or `carta-auth-token`, any of which may match.
fn given_tokens<'a>(uri: &'a Uri, headers: &'a HeaderMap) -> Vec<&'a str> {
    let query = (uri.query().into_iter())
        .flat_map(|query| query.split('&'))
        .find_map(|pair| pair.strip_prefix("token="));
    let header = |name| headers.get(name).and_then(|value| value.to_str().ok());
    let bearer = header("authorization").and_then(|value| value.strip_prefix("Bearer "));
    [query, bearer, header("carta-auth-token")]
        .into_iter()
        .flatten()
        .collect()
}

/// In time that depends only on the lengths.
fn same(a: &[u8], b: &[u8]) -> bool {
    let differ = a.iter().zip(b).fold(0, |differ, (x, y)| differ | (x ^ y));
    a.len() == b.len() && std::hint::black_box(differ) == 0
}

impl Server {
    /// The cache, even after a job panicked holding it: the worst a panic leaves is an entry missing.
    fn cache(&self) -> MutexGuard<'_, Cache> {
        self.cache.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `path`, relative to the root, as a path under the root once any symlinks are followed.
    fn resolve(&self, path: &str) -> Result<PathBuf, String> {
        let relative = Path::new(path);
        let plain =
            (relative.components()).all(|c| matches!(c, Component::Normal(_) | Component::CurDir));
        if !plain || path.contains('\0') {
            return Err("not a path under the root".into());
        }
        let full =
            (self.root.join(relative).canonicalize()).map_err(|_| "no such file or directory")?;
        if !full.starts_with(&self.root) {
            return Err("not a path under the root".into());
        }
        Ok(full)
    }

    fn list(&self, directory: &str) -> proto::FileListResponse {
        let mut response = proto::FileListResponse {
            directory: directory.to_string(),
            parent: directory
                .rsplit_once('/')
                .map_or("", |(parent, _)| parent)
                .into(),
            ..Default::default()
        };
        let entries = match self.resolve(directory).map(std::fs::read_dir) {
            Ok(Ok(entries)) => entries,
            _ => {
                response.message = "no such directory".into();
                return response;
            }
        };
        for entry in entries.flatten() {
            let Some(name) = entry.file_name().to_str().map(String::from) else {
                continue;
            };
            if name.starts_with('.') {
                continue;
            }
            // Through resolve, so that a symlink out of the root isn't listed.
            let Ok(metadata) = (self.resolve(&join(directory, &name)))
                .and_then(|p| p.metadata().map_err(|e| e.to_string()))
            else {
                continue;
            };
            let date = modified(&metadata).as_secs() as i64;
            if metadata.is_dir() {
                response.subdirectories.push(proto::DirectoryInfo {
                    name,
                    item_count: 0,
                    date,
                });
            } else if is_fits(&name) {
                response.files.push(file_info(&name, &metadata, None));
            }
        }
        response.subdirectories.sort_by(|a, b| a.name.cmp(&b.name));
        response.files.sort_by(|a, b| a.name.cmp(&b.name));
        response.success = true;
        response
    }

    fn info(&self, request: &proto::FileInfoRequest) -> Result<proto::FileInfoResponse, String> {
        let path = join(&directory(&request.directory), &request.file);
        let file = File::open(self.resolve(&path)?).map_err(|e| e.to_string())?;
        let header = fits::read_image_header(&file, hdu(&request.hdu)?)?;
        let metadata = file.metadata().map_err(|e| e.to_string())?;
        let hdu = header.hdu.to_string();
        Ok(proto::FileInfoResponse {
            success: true,
            file_info: Some(file_info(&request.file, &metadata, Some(hdu.clone()))),
            file_info_extended: [(hdu, carta::file_info_extended(&header))].into(),
            ..Default::default()
        })
    }
}

/// CARTA's name for the root is $BASE.
fn directory(directory: &str) -> String {
    match directory.trim_matches('/') {
        "$BASE" | "." => String::new(),
        directory => directory.into(),
    }
}

fn join(directory: &str, name: &str) -> String {
    match directory {
        "" => name.into(),
        directory => format!("{directory}/{name}"),
    }
}

fn is_fits(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    [".fits", ".fit", ".fts"]
        .iter()
        .any(|ext| name.ends_with(ext))
}

fn hdu(hdu: &str) -> Result<Option<usize>, String> {
    (!hdu.is_empty()).then(|| fits::parse_hdu(hdu)).transpose()
}

fn modified(metadata: &Metadata) -> std::time::Duration {
    (metadata.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .unwrap_or_default()
}

fn file_info(name: &str, metadata: &Metadata, hdu: Option<String>) -> proto::FileInfo {
    proto::FileInfo {
        name: name.into(),
        r#type: proto::FileType::Fits.into(),
        size: metadata.len() as i64,
        hdu_list: hdu.into_iter().collect(),
        date: modified(metadata).as_secs() as i64,
    }
}

fn error_data(message: impl Into<String>) -> Vec<u8> {
    let error = proto::ErrorData {
        severity: proto::ErrorSeverity::Error.into(),
        message: message.into(),
        ..Default::default()
    };
    frame(EventType::ErrorData, 0, &error)
}

/// Infinities as NaN, as a client takes them, so that the cells, pyramids and values it's sent agree with the
/// cubelets.
fn nan_infinities(data: &mut [f32]) {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|scope| {
        for chunk in data.chunks_mut(data.len().div_ceil(threads).max(1)) {
            scope.spawn(|| {
                for v in chunk {
                    if v.is_infinite() {
                        *v = f32::NAN;
                    }
                }
            });
        }
    });
}

/// Compressed cubelets by the content they're of, its pyramid, level factors, place and tolerance, least recently used
/// first out.
type Key = (Arc<str>, Mode, [usize; 3], [usize; 3], Option<u64>);

/// A cubelet's CUBELET_DATA without its address, encoded, to follow the address of each one it's sent as.
type Payload = Arc<[u8]>;

#[derive(Default)]
struct Cache {
    entries: HashMap<Key, (Payload, u64)>,
    used: BTreeMap<u64, Key>,
    tick: u64,
    bytes: usize,
}

fn size(payload: &Payload) -> usize {
    payload.len() + 64
}

impl Cache {
    fn get(&mut self, key: &Key) -> Option<Payload> {
        self.tick += 1;
        let (payload, used) = self.entries.get_mut(key)?;
        let key = self.used.remove(used).unwrap();
        *used = self.tick;
        self.used.insert(self.tick, key);
        Some(payload.clone())
    }

    fn insert(&mut self, key: Key, payload: Payload) {
        self.tick += 1;
        self.bytes += size(&payload);
        if let Some((old, used)) = self.entries.insert(key.clone(), (payload, self.tick)) {
            self.bytes -= size(&old);
            self.used.remove(&used);
        }
        self.used.insert(self.tick, key);
        while self.bytes > CACHE_BYTES
            && let Some((_, key)) = self.used.pop_first()
        {
            let (old, _) = self.entries.remove(&key).unwrap();
            self.bytes -= size(&old);
        }
    }
}

/// A cubelet's CUBELET_DATA: its address, then its payload, since protobuf merges a message's encodings.
fn cubelet_frame(file_id: i32, id: CubeletId, payload: &[u8]) -> Vec<u8> {
    let [x, y, z] = id.at().map(|v| v as i32);
    let address = proto::CubeletData {
        file_id,
        mode: SmoothingMode::from(id.mode).into(),
        level: id.level.into(),
        x,
        y,
        z,
        ..Default::default()
    };
    let mut bytes = frame(EventType::CubeletData, 0, &address);
    bytes.extend_from_slice(payload);
    bytes
}

struct Session {
    server: Arc<Server>,
    id: u32,
    /// Whether the request carried the right token, so that REGISTER_VIEWER needn't.
    authorised: bool,
    registered: bool,
    file: Option<Open>,
    /// Jobs out, for this file or one before it.
    jobs: Vec<Running>,
    done: mpsc::UnboundedSender<Done>,
    /// To the task writing to the socket, so that the session reads on while the client is slow to.
    // ponytail: unbounded, so a client that stops reading grows it; stage 2's flow control bounds it.
    out: mpsc::UnboundedSender<Message>,
    /// Why the session ends, once it's to end after what it's sending.
    closing: Option<String>,
}

/// A job out, and the flag that drops what it cuts once that's no longer wanted.
struct Running {
    cubelet: Option<CubeletId>,
    cancelled: Arc<AtomicBool>,
    handle: JoinHandle<()>,
}

struct Open {
    file_id: i32,
    path: String,
    info: FileInfo,
    engine: Engine,
    content_id: Arc<str>,
    levels: Vec<[usize; 3]>,
    /// What level 0's cubelets are compressed to, from the request for them last: None for exact.
    tolerance: Option<f64>,
    sent: Sent,
}

#[derive(Default)]
struct Sent {
    messages: usize,
    bytes: usize,
    cubelets: usize,
    cached: usize,
}

impl Open {
    /// The cache's key for `id`, and the tolerance it's compressed to.
    fn key(&self, id: CubeletId) -> (Key, Option<f64>) {
        let factors = self.levels[usize::from(id.level)];
        let tolerance = self.tolerance.map(|t| t * id.mode.noise_ratio(factors));
        let key = (
            self.content_id.clone(),
            id.mode,
            factors,
            id.at(),
            tolerance.map(f64::to_bits),
        );
        (key, tolerance)
    }
}

/// What a job cut, compressed, unless it was cancelled, or why it panicked.
struct Done {
    cancelled: Arc<AtomicBool>,
    cubelets: Result<Vec<(CubeletId, Payload)>, String>,
}

async fn run(server: Arc<Server>, socket: WebSocket, authorised: bool) {
    let id = server.sessions.fetch_add(1, Ordering::Relaxed) + 1;
    let (mut sink, mut stream) = socket.split();
    let (out, mut outgoing) = mpsc::unbounded_channel();
    let writer = tokio::spawn(async move {
        while let Some(message) = outgoing.recv().await {
            if sink.send(message).await.is_err() {
                return;
            }
        }
        let _ = sink.close().await;
    });
    let (done, mut finished) = mpsc::unbounded_channel();
    let mut session = Session {
        server,
        id,
        authorised,
        registered: false,
        file: None,
        jobs: Vec::new(),
        done,
        out,
        closing: None,
    };
    println!("Session {id}: connected");
    // What came in while a file was opening, to be handled after it in turn.
    let mut deferred: VecDeque<Bytes> = VecDeque::new();
    let deadline = tokio::time::sleep(REGISTER_WITHIN);
    tokio::pin!(deadline);
    let reason = loop {
        session.start_jobs();
        let handled = match deferred.pop_front() {
            Some(bytes) => session.handle(&bytes, &mut stream, &mut deferred).await,
            None => tokio::select! {
                message = stream.next() => match message {
                    Some(Ok(Message::Binary(bytes))) => {
                        session.handle(&bytes, &mut stream, &mut deferred).await
                    }
                    Some(Ok(Message::Text(text))) if text.as_str() == "PING" => {
                        let _ = session.out.send(Message::Text("PONG".into()));
                        Ok(())
                    }
                    Some(Ok(Message::Close(_))) | None => break "closed by the client".to_string(),
                    Some(Ok(_)) => Ok(()),
                    Some(Err(e)) => break e.to_string(),
                },
                Some(done) = finished.recv() => {
                    session.finished(done);
                    Ok(())
                }
                () = &mut deadline, if !session.registered => {
                    break "no REGISTER_VIEWER in time".to_string();
                }
            },
        };
        // A message that makes no sense: nothing after it from this client can be trusted.
        if let Err(e) = handled {
            writer.abort();
            break format!("a bad message, {e}");
        }
        if session.out.is_closed() {
            break "the client went".into();
        }
        if let Some(reason) = session.closing.take() {
            break reason;
        }
    };
    session.close_file();
    // What's left to send goes, then the socket closes.
    drop(session);
    let _ = writer.await;
    println!("Session {id}: {reason}");
}

impl Session {
    fn send(&mut self, bytes: Vec<u8>) {
        if let Some(Open { sent, .. }) = &mut self.file {
            let cubelet = (EventType::CubeletData as u16).to_le_bytes();
            sent.messages += 1;
            sent.bytes += bytes.len();
            sent.cubelets += usize::from(bytes.starts_with(&cubelet));
        }
        let _ = self.out.send(Message::Binary(bytes.into()));
    }

    async fn handle(
        &mut self,
        bytes: &[u8],
        stream: &mut (impl Stream<Item = Result<Message, axum::Error>> + Unpin),
        deferred: &mut VecDeque<Bytes>,
    ) -> Result<(), String> {
        let message = carta::unframe(bytes)?;
        let request_id = message.request_id;
        let Some(event) = message.event else {
            self.send(error_data("Bad event type"));
            return Ok(());
        };
        if !self.registered && event != EventType::RegisterViewer {
            return Err(format!("{event:?} before REGISTER_VIEWER"));
        }
        match event {
            EventType::RegisterViewer => self.register(&message)?,
            EventType::FileListRequest => {
                let request: proto::FileListRequest = message.decode()?;
                let server = self.server.clone();
                let list = spawn_blocking(move || server.list(&directory(&request.directory)));
                let list = list.await.map_err(|e| e.to_string())?;
                self.send(frame(EventType::FileListResponse, request_id, &list));
            }
            EventType::FileInfoRequest => {
                let request: proto::FileInfoRequest = message.decode()?;
                let server = self.server.clone();
                let info = spawn_blocking(move || server.info(&request));
                let info = (info.await.map_err(|e| e.to_string())?).unwrap_or_else(|message| {
                    proto::FileInfoResponse {
                        message,
                        ..Default::default()
                    }
                });
                self.send(frame(EventType::FileInfoResponse, request_id, &info));
            }
            EventType::OpenFile => {
                let open = message.decode()?;
                self.open_reading_on(request_id, open, stream, deferred)
                    .await
            }
            EventType::CloseFile => self.close(message.decode()?),
            EventType::SetHistogramRequirements => self.histogram(message.decode()?),
            EventType::SetStatsRequirements => self.stats(message.decode()?),
            EventType::AddRequiredCubelets => self.add(message.decode()?),
            EventType::RemoveRequiredCubelets => self.remove(message.decode()?),
            EventType::PixelValueRequest => self.pixel_value(request_id, message.decode()?),
            _ => self.send(error_data(format!("Bad event type {event:?}"))),
        }
        Ok(())
    }

    fn register(&mut self, message: &Frame) -> Result<(), String> {
        let register: proto::RegisterViewer = message.decode()?;
        let token = self.server.token.as_bytes();
        let authorised = self.authorised
            || (!register.api_key.is_empty() && same(register.api_key.as_bytes(), token));
        let success = authorised && message.icd_version == ICD_VERSION;
        let flags =
            proto::ServerFeatureFlags::ReadOnly as u32 | proto::ServerFeatureFlags::Volumes as u32;
        let ack = proto::RegisterViewerAck {
            session_id: self.id,
            success,
            message: match (authorised, success) {
                (false, _) => "Invalid token".into(),
                (true, true) => String::new(),
                (true, false) => format!(
                    "Invalid ICD version number. Expected {ICD_VERSION}, got {}",
                    message.icd_version
                ),
            },
            session_type: proto::SessionType::New.into(),
            ..Default::default()
        };
        // A client without the token isn't told anything about the server.
        let ack = match authorised {
            true => proto::RegisterViewerAck {
                server_feature_flags: flags,
                platform_strings: [(
                    "server".into(),
                    format!("fits-server {}", env!("CARGO_PKG_VERSION")),
                )]
                .into(),
                ..ack
            },
            false => ack,
        };
        self.registered |= success;
        if !success {
            self.closing = Some(ack.message.clone());
        }
        self.send(frame(
            EventType::RegisterViewerAck,
            message.request_id,
            &ack,
        ));
        Ok(())
    }

    /// The open, while the client's messages are read on and kept for after it, so that if the client goes the open is
    /// given up, along with its place in the queue to read.
    async fn open_reading_on(
        &mut self,
        request_id: u32,
        open: proto::OpenFile,
        stream: &mut (impl Stream<Item = Result<Message, axum::Error>> + Unpin),
        deferred: &mut VecDeque<Bytes>,
    ) {
        let out = self.out.clone();
        let mut held: usize = deferred.iter().map(Bytes::len).sum();
        let gone = {
            let opening = self.open(request_id, open);
            tokio::pin!(opening);
            loop {
                let full = deferred.len() >= MAX_HELD || held >= MAX_HELD_BYTES;
                tokio::select! {
                    biased;
                    () = &mut opening => break None,
                    message = stream.next(), if !full => match message {
                        Some(Ok(Message::Binary(bytes))) => {
                            held += bytes.len();
                            // Copied, since the message is a slice of the socket's read buffer, and holding the
                            // slice would keep the whole buffer in memory.
                            deferred.push_back(Bytes::copy_from_slice(&bytes));
                        }
                        Some(Ok(Message::Text(text))) if text.as_str() == "PING" => {
                            let _ = out.send(Message::Text("PONG".into()));
                        }
                        Some(Ok(Message::Close(_))) | None => break Some("closed by the client".into()),
                        Some(Ok(_)) => {}
                        Some(Err(e)) => break Some(e.to_string()),
                    },
                }
            }
        };
        if let Some(reason) = gone {
            self.closing = Some(format!("{reason}, while a file was opening"));
        }
    }

    /// In place of the file open, which goes once the new one's header has been read and before its data is, so that
    /// a bad path leaves it open and the two are never in memory together.
    async fn open(&mut self, request_id: u32, open: proto::OpenFile) {
        let start = Instant::now();
        let file_id = open.file_id;
        let path = join(&directory(&open.directory), &open.file);
        let ack = match self.read(&path, &open).await {
            Ok((file, metadata)) => {
                let info = &file.info;
                println!(
                    "Session {}: opened {path} (HDU {}, {} x {} x {}) in {:.2} s",
                    self.id,
                    info.hdu,
                    info.dims[0],
                    info.dims[1],
                    info.dims[2],
                    start.elapsed().as_secs_f32()
                );
                let levels = (file.levels.iter())
                    .map(|f| f.map(|f| f as i32))
                    .map(|[x, y, z]| proto::LevelFactors { x, y, z })
                    .collect();
                let ack = proto::OpenFileAck {
                    success: true,
                    file_id,
                    file_info: Some(file_info(&open.file, &metadata, Some(info.hdu.to_string()))),
                    file_info_extended: Some(carta::file_info_extended(&info.header())),
                    volume_info: Some(proto::VolumeInfo {
                        cubelet_size: CUBELET as i32,
                        levels,
                        content_id: file.content_id.to_string(),
                        noise_sigma: info.sigma.unwrap_or(0.0),
                    }),
                    ..Default::default()
                };
                self.file = Some(file);
                ack
            }
            Err(message) => proto::OpenFileAck {
                file_id,
                message,
                ..Default::default()
            },
        };
        self.send(frame(EventType::OpenFileAck, request_id, &ack));
    }

    /// The file at `path`, once its header has been read and the file before has gone, along with every job that held
    /// some of it. When the data then fails to read, the client is told that the file before is closed.
    async fn read(
        &mut self,
        path: &str,
        open: &proto::OpenFile,
    ) -> Result<(Open, Metadata), String> {
        let full = self.server.resolve(path)?;
        let hdu = hdu(&open.hdu)?;
        let budget = open.volume_budget.unwrap_or(DEFAULT_BUDGET);
        let gpu_memory_limit_mb = u64::try_from(budget.gpu_memory_limit_mb)
            .ok()
            .filter(|&mb| mb > 0);
        let max_depth = usize::try_from(budget.max_depth)
            .ok()
            .filter(|&depth| depth > 0);
        let (Some(gpu_memory_limit_mb), Some(max_depth)) = (gpu_memory_limit_mb, max_depth) else {
            return Err("a volume budget of nothing".into());
        };
        let relative = full.strip_prefix(&self.server.root).unwrap_or(&full);
        let canonical = relative.to_string_lossy().into_owned();
        let header = spawn_blocking(move || {
            let file = File::open(full).map_err(|e| e.to_string())?;
            fits::read_image_header(&file, hdu)?;
            let metadata = file.metadata().map_err(|e| e.to_string())?;
            Ok::<_, String>((file, metadata))
        });
        let (file, metadata) = header.await.map_err(|e| e.to_string())??;

        let closed = self.close_file();
        for job in std::mem::take(&mut self.jobs) {
            let _ = job.handle.await;
        }
        let opens = self.server.opens.clone();
        let read = async {
            // Held by the read itself, which can't be stopped once it's begun.
            let permit = opens.acquire_owned().await.map_err(|e| e.to_string())?;
            let read = spawn_blocking(move || {
                let _permit = permit;
                let mut cube = fits::read_fits(&file, hdu)?;
                nan_infinities(&mut cube.data);
                Ok(cube)
            });
            read.await.map_err(|e| e.to_string())?
        };
        let cube = match read.await {
            Ok(cube) => cube,
            Err(e) => {
                if let Some((file_id, closed)) = closed {
                    let error = proto::ErrorData {
                        severity: proto::ErrorSeverity::Error.into(),
                        tags: vec![FILE_CLOSED.into()],
                        message: format!("{closed} was closed to open {path}, which failed"),
                        data: file_id.to_string(),
                    };
                    self.send(frame(EventType::ErrorData, 0, &error));
                }
                return Err(e);
            }
        };
        let levels = fits::levels(cube.info.dims, gpu_memory_limit_mb, max_depth);
        let (chain, levels) = (levels.chain.len(), levels.list().0);
        let modified = modified(&metadata).as_nanos();
        let content_id = format!(
            "{canonical}:{}:{modified}:{}",
            metadata.len(),
            cube.info.hdu
        );
        let info = cube.info.clone();
        let mut engine = Engine::new(cube);
        engine.request(Request::Levels {
            factors: levels.clone(),
            chain,
        });
        let file = Open {
            file_id: open.file_id,
            path: path.into(),
            info,
            engine,
            content_id: content_id.into(),
            levels,
            tolerance: None,
            sent: Sent::default(),
        };
        Ok((file, metadata))
    }

    /// Drops the file open, and what its jobs are cutting, and returns its ID and path.
    fn close_file(&mut self) -> Option<(i32, String)> {
        for job in &self.jobs {
            job.cancelled.store(true, Ordering::Relaxed);
        }
        let file = self.file.take()?;
        let Sent {
            messages,
            bytes,
            cubelets,
            cached,
        } = file.sent;
        println!(
            "Session {}: closed {}, after sending {messages} messages, {bytes} bytes, with {cubelets} cubelets \
             ({cached} from the cache)",
            self.id, file.path
        );
        Some((file.file_id, file.path))
    }

    /// CARTA's -1 for every file.
    fn close(&mut self, close: proto::CloseFile) {
        if (self.file.as_ref()).is_some_and(|f| close.file_id == -1 || f.file_id == close.file_id) {
            self.close_file();
        }
    }

    fn open_file(&mut self, file_id: i32) -> Option<&mut Open> {
        self.file.as_mut().filter(|file| file.file_id == file_id)
    }

    fn histogram(&mut self, requirements: proto::SetHistogramRequirements) {
        let id = self.id;
        let Some(file) = self.open_file(requirements.file_id) else {
            return;
        };
        let config = requirements
            .histograms
            .into_iter()
            .find(|h| h.channel == CUBE);
        if requirements.region_id != CUBE || config.is_none() {
            return self.send(error_data("Only the whole cube's histogram is served"));
        }
        let (histogram, clamped) = carta::histogram(&file.info.histogram, &file.info.stats);
        if clamped {
            println!(
                "Session {id}: warning: {}'s histogram has a bin past int32's range, clamped to it",
                file.path
            );
        }
        let data = proto::RegionHistogramData {
            file_id: file.file_id,
            region_id: CUBE,
            channel: CUBE,
            histograms: Some(histogram),
            progress: 1.0,
            config,
            ..Default::default()
        };
        self.send(frame(EventType::RegionHistogramData, 0, &data));
    }

    fn stats(&mut self, requirements: proto::SetStatsRequirements) {
        let Some(file) = self.open_file(requirements.file_id) else {
            return;
        };
        if requirements.region_id != CUBE {
            return self.send(error_data("Only the whole cube's statistics are served"));
        }
        let voxels = file.info.dims.iter().product();
        let data = proto::RegionStatsData {
            file_id: file.file_id,
            region_id: CUBE,
            channel: CUBE,
            statistics: carta::statistics(&file.info.stats, voxels),
            ..Default::default()
        };
        self.send(frame(EventType::RegionStatsData, 0, &data));
    }

    /// The cubelets in `keys` that are in the file's levels, in `mode`.
    fn cubelets(file: &Open, mode: Mode, keys: &[u64]) -> Vec<CubeletId> {
        let dims = file.info.dims;
        (keys.iter().map(|&key| cubelet_at(key)))
            .map(|(level, at)| (usize::from(level), at.map(usize::from)))
            .filter(|&(level, at)| {
                let factors = file.levels.get(level);
                factors.is_some_and(|&f| fits::cubelet_region(dims, f, at).is_some())
            })
            .map(|(level, at)| CubeletId::new(mode, Default::default(), level, at))
            .collect()
    }

    /// Those in the cache at once, the rest once they're cut.
    fn add(&mut self, add: proto::AddRequiredCubelets) {
        let tolerance = match CompressionType::try_from(add.compression_type) {
            Ok(CompressionType::None) => None,
            Ok(CompressionType::ZfpExplicit)
                if add.tolerance.is_finite() && add.tolerance > 0.0 =>
            {
                Some(add.tolerance)
            }
            _ => {
                return self.send(error_data(
                    "Cubelets need NONE, or ZFP_EXPLICIT with a positive tolerance",
                ));
            }
        };
        let mode = match carta::mode(add.mode) {
            Ok(mode) => mode,
            Err(e) => return self.send(error_data(e)),
        };
        let Some(file) = self.file.as_mut().filter(|f| f.file_id == add.file_id) else {
            return;
        };
        file.tolerance = tolerance;
        let (mut cached, mut cut) = (Vec::new(), Vec::new());
        let mut cache = self.server.cache();
        for id in Self::cubelets(file, mode, &add.cubelets) {
            match cache.get(&file.key(id).0) {
                Some(payload) => cached.push((id, payload)),
                None => cut.push(id),
            }
        }
        drop(cache);
        file.sent.cached += cached.len();
        file.engine.request(Request::AddCubelets { ids: cut });
        let file_id = file.file_id;
        for (id, payload) in cached {
            self.send(cubelet_frame(file_id, id, &payload));
        }
    }

    /// Out of the queue, and dropped uncompressed if they're being cut.
    fn remove(&mut self, remove: proto::RemoveRequiredCubelets) {
        let mode = match carta::mode(remove.mode) {
            Ok(mode) => mode,
            Err(e) => return self.send(error_data(e)),
        };
        let Some(file) = self.file.as_mut().filter(|f| f.file_id == remove.file_id) else {
            return;
        };
        let ids = Self::cubelets(file, mode, &remove.cubelets);
        let removed: HashSet<_> = ids.iter().collect();
        for job in &self.jobs {
            if job.cubelet.is_some_and(|id| removed.contains(&id)) {
                job.cancelled.store(true, Ordering::Relaxed);
            }
        }
        file.engine.request(Request::RemoveCubelets { ids });
    }

    fn pixel_value(&mut self, request_id: u32, request: proto::PixelValueRequest) {
        let Some(file) = self.open_file(request.file_id) else {
            return;
        };
        let voxel = [request.x, request.y, request.z].map(|v| usize::try_from(v).ok());
        let value = match voxel {
            [Some(x), Some(y), Some(z)] => file.engine.request(Request::PixelValue {
                id: request_id,
                voxel: [x, y, z],
            }),
            _ => None,
        };
        let value = match value {
            Some(Event::PixelValue { value, .. }) => value,
            _ => f32::NAN,
        };
        let response = proto::PixelValueResponse {
            file_id: file.file_id,
            value,
        };
        self.send(frame(EventType::PixelValueResponse, request_id, &response));
    }

    /// Hands out jobs up to a core's worth, in the order the engine has them. Each compresses its cubelet into the
    /// cache, once it has dropped what it held of the engine's, unless it's cancelled by then.
    fn start_jobs(&mut self) {
        while self.jobs.len() < self.server.workers {
            let Some(file) = self.file.as_mut() else {
                return;
            };
            let Some(job) = file.engine.next_job() else {
                return;
            };
            let cubelet = job.cubelet();
            let target = cubelet.map(|id| file.key(id));
            self.spawn_job(cubelet, target, move || {
                let events = job.run();
                drop(job);
                events
            });
        }
    }

    /// `work` on a blocking thread, which always ends in a Done, even when it panics.
    fn spawn_job(
        &mut self,
        cubelet: Option<CubeletId>,
        target: Option<(Key, Option<f64>)>,
        work: impl FnOnce() -> Vec<Event> + Send + 'static,
    ) {
        let cancelled = Arc::new(AtomicBool::new(false));
        let (flag, done, server) = (cancelled.clone(), self.done.clone(), self.server.clone());
        let handle = spawn_blocking(move || {
            let cut = std::panic::catch_unwind(AssertUnwindSafe(|| {
                let events = work();
                let mut cubelets = Vec::new();
                if let Some((key, tolerance)) = target.filter(|_| !flag.load(Ordering::Relaxed)) {
                    for event in events {
                        if let Event::Cubelet { id, cubelet } = event {
                            let payload: Payload =
                                codec::encode(&cubelet, tolerance).encode_to_vec().into();
                            server.cache().insert(key.clone(), payload.clone());
                            cubelets.push((id, payload));
                        }
                    }
                }
                cubelets
            }));
            let cubelets = cut.map_err(|panic| {
                let message = (panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .or_else(|| panic.downcast_ref::<String>().cloned());
                message.unwrap_or_else(|| "a panic".into())
            });
            let _ = done.send(Done {
                cancelled: flag,
                cubelets,
            });
        });
        self.jobs.push(Running {
            cubelet,
            cancelled,
            handle,
        });
    }

    /// A job's cubelets, unless they're no longer wanted.
    fn finished(&mut self, done: Done) {
        let Some(at) =
            (self.jobs.iter()).position(|job| Arc::ptr_eq(&job.cancelled, &done.cancelled))
        else {
            return;
        };
        let job = self.jobs.swap_remove(at);
        let cubelets = match done.cubelets {
            Ok(cubelets) => cubelets,
            Err(panic) => {
                let what = job.cubelet.map_or("a job".into(), |id| format!("{id:?}"));
                println!("Session {}: {what} panicked: {panic}", self.id);
                return self.send(error_data(format!("Cutting {what} failed: {panic}")));
            }
        };
        let file_id = self.file.as_ref().map(|file| file.file_id);
        let Some(file_id) = file_id.filter(|_| !done.cancelled.load(Ordering::Relaxed)) else {
            return;
        };
        for (id, payload) in cubelets {
            self.send(cubelet_frame(file_id, id, &payload));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use fits_core::{Cell, Cubelet, Voxels};

    use super::*;

    /// A registered session on a server of `root`, and what it hands its loop and its socket.
    fn session(
        root: &Path,
    ) -> (
        Session,
        mpsc::UnboundedReceiver<Done>,
        mpsc::UnboundedReceiver<Message>,
    ) {
        let server = Arc::new(Server {
            root: root.canonicalize().unwrap(),
            token: String::new(),
            cache: Mutex::default(),
            opens: Arc::new(Semaphore::new(1)),
            sessions: AtomicU32::new(0),
            workers: 4,
        });
        let ((done, finished), (out, outgoing)) =
            (mpsc::unbounded_channel(), mpsc::unbounded_channel());
        let session = Session {
            server,
            id: 1,
            authorised: true,
            registered: true,
            file: None,
            jobs: Vec::new(),
            done,
            out,
            closing: None,
        };
        (session, finished, outgoing)
    }

    async fn next(finished: &mut mpsc::UnboundedReceiver<Done>) -> Done {
        let done = tokio::time::timeout(Duration::from_secs(10), finished.recv()).await;
        done.expect("a Done in time").unwrap()
    }

    #[tokio::test]
    async fn a_job_that_panics_frees_its_place_and_says_so_and_a_poisoned_cache_still_serves() {
        let (mut session, mut finished, mut outgoing) = session(&std::env::temp_dir());
        session.spawn_job(None, None, || panic!("on purpose"));
        session.finished(next(&mut finished).await);
        assert!(session.jobs.is_empty());
        let Some(Message::Binary(bytes)) = outgoing.recv().await else {
            panic!("no error")
        };
        let error: proto::ErrorData = carta::unframe(&bytes).unwrap().decode().unwrap();
        assert!(error.message.contains("on purpose"), "{}", error.message);

        let server = session.server.clone();
        let _ = std::thread::spawn(move || {
            let _cache = server.cache.lock().unwrap();
            panic!("holding the cache");
        })
        .join();
        assert!(session.server.cache.is_poisoned());
        let id = CubeletId::new(Mode::Max, Default::default(), 0, [0; 3]);
        let key: Key = ("cube".into(), Mode::Max, [1; 3], [0; 3], None);
        let cubelet = Cubelet {
            dims: [1; 3],
            voxels: Voxels::Data(vec![1.0]),
            cell: Cell {
                max: 1.0,
                has_nan: false,
            },
        };
        session.spawn_job(Some(id), Some((key.clone(), None)), move || {
            vec![Event::Cubelet { id, cubelet }]
        });
        let done = next(&mut finished).await;
        assert!(matches!(&done.cubelets, Ok(cubelets) if cubelets.len() == 1));
        assert!(session.server.cache().get(&key).is_some());
    }

    /// A root of its own with a small cube in it, and the OPEN_FILE for it.
    fn root_with_cube(test: &str) -> (PathBuf, Vec<u8>) {
        let root = std::env::temp_dir().join(format!("fits-server-{test}-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let mut bytes: Vec<u8> = [
            "SIMPLE  =                    T",
            "BITPIX  =                  -32",
            "NAXIS   =                    3",
            "NAXIS1  =                    4",
            "NAXIS2  =                    4",
            "NAXIS3  =                    4",
            "END",
        ]
        .iter()
        .flat_map(|card| format!("{card:<80}").into_bytes())
        .collect();
        bytes.resize(2 * 2880, 0);
        std::fs::write(root.join("cube.fits"), bytes).unwrap();
        let open = proto::OpenFile {
            file: "cube.fits".into(),
            file_id: 1,
            ..Default::default()
        };
        (root, frame(EventType::OpenFile, 0, &open))
    }

    #[tokio::test]
    async fn an_open_waiting_to_read_is_given_up_when_the_client_goes() {
        let (root, message) = root_with_cube("unit-gone");
        let (mut session, _finished, _outgoing) = session(&root);

        // Another session's read has the only place.
        let held = session.server.opens.clone().try_acquire_owned().unwrap();
        let mut gone = futures_util::stream::empty::<Result<Message, axum::Error>>();
        let mut deferred = VecDeque::new();
        let handled = session.handle(&message, &mut gone, &mut deferred);
        let handled = tokio::time::timeout(Duration::from_secs(10), handled).await;
        handled.expect("the open given up").unwrap();
        assert!(session.closing.is_some() && session.file.is_none());
        drop(held);
        assert_eq!(session.server.opens.available_permits(), 1);
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn a_session_stops_reading_while_a_file_opens_once_it_holds_its_limit() {
        let (root, message) = root_with_cube("unit-held");
        // Past the message limit, then past the byte limit.
        for (count, size, holds) in [(MAX_HELD + 1, 8, MAX_HELD), (3, MAX_REQUEST, 2)] {
            let (mut session, _finished, _outgoing) = session(&root);
            let permit = session.server.opens.clone().try_acquire_owned().unwrap();
            let sent = (0..count).map(|_| Ok(Message::Binary(vec![0; size].into())));
            let mut stream =
                futures_util::stream::iter(sent).chain(futures_util::stream::pending());
            let mut deferred = VecDeque::new();
            {
                let handled = session.handle(&message, &mut stream, &mut deferred);
                tokio::pin!(handled);
                // The open waits for the read permit while the client sends.
                let waiting = tokio::time::timeout(Duration::from_millis(300), &mut handled);
                assert!(waiting.await.is_err(), "the open ended early");
                drop(permit);
                let handled = tokio::time::timeout(Duration::from_secs(10), handled).await;
                handled.expect("the open done").unwrap();
            }
            assert_eq!(deferred.len(), holds);
            assert!(session.closing.is_none() && session.file.is_some());
            // What it didn't read is still there for the session to read next.
            assert!(matches!(stream.next().await, Some(Ok(Message::Binary(_)))));
        }
        let _ = std::fs::remove_dir_all(root);
    }
}
