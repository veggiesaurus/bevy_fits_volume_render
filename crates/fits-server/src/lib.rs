//! FITS cubes under a root folder, served as cubelets over a subset of CARTA's protocol with the volume messages added
//! to it. A session has one cube open at a time, cutting cubelets through fits-core's engine on blocking threads. The
//! cubes are held whole, once for every session that opens them and within a budget, and the cubelets compressed are
//! kept for every session in one cache.

use std::{
    collections::{HashSet, VecDeque},
    fs::{File, Metadata},
    panic::AssertUnwindSafe,
    path::{Component, Path, PathBuf},
    sync::{
        Arc, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    time::{Duration, Instant, UNIX_EPOCH},
};

pub mod tls;

use axum::{
    Router,
    body::Bytes,
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade, rejection::WebSocketUpgradeRejection},
    },
    http::{HeaderMap, StatusCode, Uri},
    response::{Html, IntoResponse, Response},
    routing::get,
};
use fits_core::{
    self as fits, CUBELET, Cube, FileInfo,
    cache::Lru,
    carta::{self, CUBE, FILE_CLOSED, Frame, ICD_VERSION, cubelet_at, frame, proto},
    codec,
    engine::{CubeletId, Engine, Event, Mode, Queue, Request},
};
use futures_util::{SinkExt, Stream, StreamExt};
use prost::Message as _;
use proto::{CompressionType, EventType, SmoothingMode};
use tokio::{
    net::TcpListener,
    sync::{
        OwnedSemaphorePermit, Semaphore,
        mpsc::{self, OwnedPermit, error::TrySendError},
    },
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
/// Replies and errors queued for the socket, which go ahead of any cubelets queued. Once this many are queued, the
/// session stops reading the client's messages until some have gone.
const CONTROL: usize = 64;
/// Cubelets being cut or queued for the socket, a session's at most, so that a client slow to read holds up the cutting
/// rather than filling the server's memory.
const BULK: usize = 32;
/// How long a session that's ending waits for the client to take what's left to send.
const CLOSE_WITHIN: Duration = Duration::from_secs(10);
/// A client that doesn't say what it can hold gets the viewer's defaults.
const DEFAULT_BUDGET: proto::VolumeBudget = proto::VolumeBudget {
    gpu_memory_limit_mb: 384,
    max_depth: 2048,
};

struct Server {
    /// Canonical, so that a resolved path can be checked against it.
    root: PathBuf,
    token: String,
    cache: Mutex<Lru<Key, Payload>>,
    cubes: Mutex<Cubes>,
    /// read_fits reads on every core, so one read at a time.
    opens: Arc<Semaphore>,
    sessions: AtomicU32,
    /// Jobs a session has out at a time.
    workers: usize,
}

/// Serves the cubes under `root` until the listener fails, holding at most `memory_limit` bytes of them. A client
/// connects to `/` with `token` as `?token=`, as `Authorization: Bearer`, or in the `carta-auth-token` header, or else in
/// REGISTER_VIEWER. With `tls`, the same port takes `wss://` as well as `ws://`.
pub async fn serve(
    listener: TcpListener,
    root: &Path,
    token: String,
    tls: Option<Arc<rustls::ServerConfig>>,
    memory_limit: u64,
) -> std::io::Result<()> {
    if token.is_empty() {
        return Err(std::io::Error::other("an empty token would let anyone in"));
    }
    let server = Arc::new(Server {
        root: root.canonicalize()?,
        token,
        cache: Mutex::new(Lru::new(CACHE_BYTES)),
        cubes: Mutex::new(Cubes::new(memory_limit)),
        opens: Arc::new(Semaphore::new(1)),
        sessions: AtomicU32::new(0),
        workers: std::thread::available_parallelism().map_or(4, |n| n.get()),
    });
    let app = Router::new().route("/", get(upgrade)).with_state(server);
    match tls {
        Some(tls) => axum::serve(tls::Listener::new(listener, tls)?, app).await,
        None => axum::serve(listener, app).await,
    }
}

/// For `GET /` in a browser, which opens with no warning once the browser trusts the server's certificate.
const RUNNING: &str = "<!doctype html><meta charset=utf-8><title>fits-server</title>\
    <p>fits-server is running. You can close this tab.";

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
        // A browser's plain GET, rather than an upgrade gone wrong, whose rejection says why.
        Err(_) if !headers.contains_key("upgrade") => Html(RUNNING).into_response(),
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
    fn cache(&self) -> MutexGuard<'_, Lru<Key, Payload>> {
        self.cache.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn cubes(&self) -> MutexGuard<'_, Cubes> {
        self.cubes.lock().unwrap_or_else(PoisonError::into_inner)
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
    [".fits", ".fit", ".fts", ".fz"]
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

/// The cubes read, by content ID, which the sessions that open one share. One that's no longer open stays until another
/// needs its room. Open or not, they all count against the budget.
// ponytail: a session's pyramid, up to a third of its cube more, isn't counted; count it if the budget proves too loose.
struct Cubes {
    budget: u64,
    /// Least recently used first.
    held: Vec<Held>,
}

struct Held {
    content_id: Arc<str>,
    cube: Arc<Cube>,
    /// A clone for each session with the cube open, beside this one, which jobs that still have the cube don't count.
    open: Arc<()>,
    bytes: u64,
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

impl Cubes {
    fn new(budget: u64) -> Self {
        Self {
            budget,
            held: Vec::new(),
        }
    }

    /// The cube, and a hold on it as open, as the most recently used.
    fn get(&mut self, content_id: &str) -> Option<(Arc<Cube>, Arc<()>)> {
        let at = (self.held.iter()).position(|held| &*held.content_id == content_id)?;
        let held = self.held.remove(at);
        let got = (held.cube.clone(), held.open.clone());
        self.held.push(held);
        Some(got)
    }

    /// The bytes of the cubes open, not counting one that only `own`, a session's hold on its own, has.
    fn open(&self, own: Option<&Arc<()>>) -> u64 {
        let only_own = |held: &Held| own.is_some_and(|own| Arc::ptr_eq(own, &held.open));
        (self.held.iter())
            .filter(|held| Arc::strong_count(&held.open) > 1 + usize::from(only_own(held)))
            .map(|held| held.bytes)
            .sum()
    }

    /// Whether `bytes` more can be held, once the cubes no session has open go, and the one only the session opening it
    /// has open, held by `own`.
    fn check(&self, bytes: u64, own: Option<&Arc<()>>) -> Result<(), String> {
        let open = self.open(own);
        match open.saturating_add(bytes) <= self.budget {
            true => Ok(()),
            false => Err(self.refusal(bytes, open)),
        }
    }

    /// Room for `bytes` more, by dropping cubes no session has open, least recently used first, which are returned to
    /// be freed once the lock is let go; or why there's none, with nothing dropped.
    fn make_room(&mut self, bytes: u64) -> Result<Vec<Held>, String> {
        self.check(bytes, None)?;
        let mut dropped = Vec::new();
        while self.held.iter().map(|held| held.bytes).sum::<u64>() + bytes > self.budget {
            let closed = (self.held.iter()).position(|held| Arc::strong_count(&held.open) == 1);
            let Some(closed) = closed else { break };
            dropped.push(self.held.remove(closed));
        }
        Ok(dropped)
    }

    fn refusal(&self, bytes: u64, open: u64) -> String {
        let (needs, budget) = (gb(bytes), gb(self.budget));
        match bytes > self.budget {
            true => format!(
                "not enough memory on the server: the cube needs {needs:.2} GB, and the server may hold {budget:.2} \
                 GB of cubes (its --memory-limit-gb)"
            ),
            false => format!(
                "not enough memory on the server: the cube needs {needs:.2} GB, and {:.2} GB of the {budget:.2} GB \
                 it may hold is taken by cubes open now",
                gb(open)
            ),
        }
    }

    /// The cube, held as open by the hold returned.
    fn insert(&mut self, content_id: Arc<str>, cube: Arc<Cube>, bytes: u64) -> Arc<()> {
        let open = Arc::new(());
        self.held.push(Held {
            content_id,
            cube,
            open: open.clone(),
            bytes,
        });
        open
    }
}

/// Compressed cubelets by the content they're of, its pyramid, level factors, place and tolerance, least recently used
/// first out.
type Key = (Arc<str>, Mode, [usize; 3], [usize; 3], Option<u64>);

/// A cubelet's CUBELET_DATA without its address, encoded, to follow the address of each one it's sent as.
type Payload = Arc<[u8]>;

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
    /// Replies and errors, to the task writing to the socket, which sends them ahead of cubelets.
    control: mpsc::Sender<Message>,
    /// What didn't fit in `control`, in order. While there's any, the session doesn't read the client's messages or
    /// send cubelets, so that nothing for a file overtakes its ack.
    overflow: VecDeque<Message>,
    /// Cubelets, with the epoch of the file they're of. A place is taken before a cubelet is cut.
    bulk: mpsc::Sender<(u32, Message)>,
    /// Moved on as a file closes, so that the writer drops its cubelets still queued.
    epoch: Arc<AtomicU32>,
    /// The open file's cubelets that the cache has, to send as there's room, each once however often it's asked for.
    hits: Queue,
    /// Whether there was no room for a cubelet, so that the session waits for some.
    blocked: bool,
    /// Why the session ends, once it's to end after what it's sending.
    closing: Option<String>,
}

/// A job out, the flag that drops what it cuts once that's no longer wanted, and for a cut, its cubelet's place in the
/// queue to the socket.
struct Running {
    cubelet: Option<CubeletId>,
    cancelled: Arc<AtomicBool>,
    room: Option<OwnedPermit<(u32, Message)>>,
    handle: JoinHandle<()>,
}

struct Open {
    file_id: i32,
    path: String,
    info: FileInfo,
    engine: Engine,
    content_id: Arc<str>,
    /// The session's hold on the cube as open.
    holding: Arc<()>,
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
    let (sink, mut stream) = socket.split();
    let (control, controls) = mpsc::channel(CONTROL);
    let (bulk, cubelets) = mpsc::channel(BULK);
    let epoch = Arc::new(AtomicU32::new(0));
    let mut writer = tokio::spawn(write(sink, controls, cubelets, epoch.clone()));
    let (done, mut finished) = mpsc::unbounded_channel();
    let mut session = Session {
        server,
        id,
        authorised,
        registered: false,
        file: None,
        jobs: Vec::new(),
        done,
        control,
        overflow: VecDeque::new(),
        bulk,
        epoch,
        hits: Queue::default(),
        blocked: false,
        closing: None,
    };
    println!("Session {id}: connected");
    // What came in while a file was opening, to be handled after it in turn.
    let mut deferred: VecDeque<Bytes> = VecDeque::new();
    let deadline = tokio::time::sleep(REGISTER_WITHIN);
    tokio::pin!(deadline);
    let reason = loop {
        session.start_jobs();
        let (control, bulk) = (session.control.clone(), session.bulk.clone());
        let reading = session.overflow.is_empty();
        let handled = if reading && let Some(bytes) = deferred.pop_front() {
            session.handle(&bytes, &mut stream, &mut deferred).await
        } else {
            tokio::select! {
                Ok(permit) = control.reserve(), if !reading => {
                    permit.send(session.overflow.pop_front().expect("an overflow"));
                    // And as many more as there's room for.
                    while let Some(message) = session.overflow.pop_front() {
                        if let Err(TrySendError::Full(message)) = session.control.try_send(message) {
                            session.overflow.push_front(message);
                            break;
                        }
                    }
                    Ok(())
                }
                // The writer has gone, which a session that isn't reading would otherwise never see.
                () = control.closed() => break "the client went".to_string(),
                Ok(_) = bulk.reserve(), if session.blocked => {
                    session.blocked = false;
                    Ok(())
                }
                message = stream.next(), if reading => match message {
                    Some(Ok(Message::Binary(bytes))) => {
                        session.handle(&bytes, &mut stream, &mut deferred).await
                    }
                    Some(Ok(Message::Text(text))) if text.as_str() == "PING" => {
                        session.queue(Message::Text("PONG".into()));
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
            }
        };
        // A message that makes no sense: nothing after it from this client can be trusted.
        if let Err(e) = handled {
            writer.abort();
            break format!("a bad message, {e}");
        }
        if session.control.is_closed() {
            break "the client went".into();
        }
        if let Some(reason) = session.closing.take() {
            break reason;
        }
    };
    session.close_file();
    // What's left to send goes, then the socket closes, unless the client has stopped reading.
    for message in std::mem::take(&mut session.overflow) {
        let _ = session.control.try_send(message);
    }
    drop(session);
    if tokio::time::timeout(CLOSE_WITHIN, &mut writer)
        .await
        .is_err()
    {
        writer.abort();
    }
    println!("Session {id}: {reason}");
}

/// Sends what's queued for the socket, replies and errors ahead of cubelets, and drops the cubelets of a file that has
/// closed since they were queued.
async fn write(
    mut sink: impl futures_util::Sink<Message> + Unpin,
    mut control: mpsc::Receiver<Message>,
    mut bulk: mpsc::Receiver<(u32, Message)>,
    epoch: Arc<AtomicU32>,
) {
    loop {
        let message = tokio::select! {
            biased;
            Some(message) = control.recv() => message,
            Some((of, message)) = bulk.recv() => {
                if of != epoch.load(Ordering::Relaxed) {
                    continue;
                }
                message
            }
            else => break,
        };
        if sink.send(message).await.is_err() {
            return;
        }
    }
    let _ = sink.close().await;
}

impl Session {
    fn send(&mut self, bytes: Vec<u8>) {
        if let Some(Open { sent, .. }) = &mut self.file {
            sent.messages += 1;
            sent.bytes += bytes.len();
        }
        self.queue(Message::Binary(bytes.into()));
    }

    /// Into `control`, or after what's waiting for room in it.
    fn queue(&mut self, message: Message) {
        let message = match self.overflow.is_empty() {
            true => match self.control.try_send(message) {
                Ok(()) | Err(TrySendError::Closed(_)) => return,
                Err(TrySendError::Full(message)) => message,
            },
            false => message,
        };
        self.overflow.push_back(message);
    }

    /// A cubelet of the open file, in the place `room` took for it.
    fn send_cubelet(&mut self, room: OwnedPermit<(u32, Message)>, id: CubeletId, payload: &[u8]) {
        let Some(file) = &mut self.file else {
            return;
        };
        let bytes = cubelet_frame(file.file_id, id, payload);
        file.sent.messages += 1;
        file.sent.bytes += bytes.len();
        file.sent.cubelets += 1;
        let epoch = self.epoch.load(Ordering::Relaxed);
        room.send((epoch, Message::Binary(bytes.into())));
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
        let control = self.control.clone();
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
                        // Dropped if the queue is full, as it can only be for a client that isn't reading.
                        Some(Ok(Message::Text(text))) if text.as_str() == "PING" => {
                            let _ = control.try_send(Message::Text("PONG".into()));
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
    /// a bad path or a cube there's no room for leaves it open.
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
            let header = fits::read_image_header(&file, hdu)?;
            let metadata = file.metadata().map_err(|e| e.to_string())?;
            Ok::<_, String>((file, header, metadata))
        });
        let (file, header, metadata) = header.await.map_err(|e| e.to_string())??;
        let modified = modified(&metadata).as_nanos();
        // A cube whose Stokes axis is shown a plane at a time says which, so that cubelets cached when it was shown
        // along z don't stand for it.
        let plane = header
            .stokes()
            .map_or(String::new(), |_| ":stokes 0".into());
        let content_id: Arc<str> = format!(
            "{canonical}:{}:{modified}:{}{plane}",
            metadata.len(),
            header.hdu
        )
        .into();
        let bytes = (header.dims.iter().product::<usize>() * size_of::<f32>()) as u64;
        // A cube a session read is shared. Otherwise it's read, one read at a time, and that's when it's checked
        // against the budget, so that no other read can take the room in between.
        let mut held = self.server.cubes().get(&content_id);
        let permit = match held {
            Some(_) => None,
            None => {
                let opens = self.server.opens.clone();
                let permit = opens.acquire_owned().await.map_err(|e| e.to_string())?;
                held = self.server.cubes().get(&content_id);
                let own = self.file.as_ref().map(|file| &file.holding);
                if held.is_none() {
                    self.server.cubes().check(bytes, own)?;
                }
                Some(permit)
            }
        };

        let closed = self.close_file();
        for job in std::mem::take(&mut self.jobs) {
            let _ = job.handle.await;
        }
        let read = match (held, permit) {
            (Some(held), _) => Ok(held),
            (None, Some(permit)) => {
                self.read_cube(permit, &content_id, file, header.hdu, bytes)
                    .await
            }
            (None, None) => unreachable!("a permit is taken for a cube that isn't held"),
        };
        let (cube, holding) = match read {
            Ok(read) => read,
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
            content_id,
            holding,
            levels,
            tolerance: None,
            sent: Sent::default(),
        };
        Ok((file, metadata))
    }

    /// HDU `hdu` of `file`, the cube of `content_id`, which takes `bytes` in memory, read with `permit` once there's
    /// room for it, and held as open.
    async fn read_cube(
        &self,
        permit: OwnedSemaphorePermit,
        content_id: &Arc<str>,
        file: File,
        hdu: usize,
        bytes: u64,
    ) -> Result<(Arc<Cube>, Arc<()>), String> {
        let (server, content_id) = (self.server.clone(), content_id.clone());
        // The permit is held by the read itself, which can't be stopped once it's begun, until the cube is in. Cubes
        // dropped for room are freed here too, off the async threads.
        let read = spawn_blocking(move || {
            let _permit = permit;
            let dropped = server.cubes().make_room(bytes)?;
            drop(dropped);
            let mut cube = fits::read_fits(&file, Some(hdu))?;
            nan_infinities(&mut cube.data);
            let cube = Arc::new(cube);
            let holding = server.cubes().insert(content_id, cube.clone(), bytes);
            Ok((cube, holding))
        });
        read.await.map_err(|e| e.to_string())?
    }

    /// Drops the file open, and what its jobs are cutting, and returns its ID and path.
    fn close_file(&mut self) -> Option<(i32, String)> {
        for job in &self.jobs {
            job.cancelled.store(true, Ordering::Relaxed);
        }
        self.hits.clear();
        self.epoch.fetch_add(1, Ordering::Relaxed);
        let file = self.file.take()?;
        // The cube that has gone least recently is the first to make room.
        drop(self.server.cubes().get(&file.content_id));
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
        let mut cut = Vec::new();
        let mut cache = self.server.cache();
        for id in Self::cubelets(file, mode, &add.cubelets) {
            match cache.get(&file.key(id).0).is_some() {
                true => _ = self.hits.push_back(id),
                false => cut.push(id),
            }
        }
        drop(cache);
        file.engine.request(Request::AddCubelets { ids: cut });
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
        let removed: HashSet<_> = ids.iter().copied().collect();
        self.hits.remove_all(&removed);
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

    /// Sends the cubelets the cache has while there's room for them, and hands out jobs up to a core's worth, in the
    /// order the engine has them: cuts while there's room for what they cut, and otherwise the jobs cuts wait on. Each
    /// cut compresses its cubelet into the cache, once it has dropped what it held of the engine's, unless it's
    /// cancelled by then. Nothing goes while a reply waits for room.
    fn start_jobs(&mut self) {
        while self.overflow.is_empty() && self.file.is_some() {
            let room = self.bulk.clone().try_reserve_owned().ok();
            self.blocked = room.is_none();
            let hit = if room.is_some() {
                self.hits.pop_front()
            } else {
                None
            };
            let room = match (hit, room) {
                (Some(id), Some(room)) => {
                    self.send_hit(room, id);
                    continue;
                }
                (_, room) => room,
            };
            let Some(file) = self
                .file
                .as_mut()
                .filter(|_| self.jobs.len() < self.server.workers)
            else {
                return;
            };
            let job = match room {
                Some(_) => file.engine.next_job(),
                None => file.engine.next_job_but_a_cut(),
            };
            let Some(job) = job else {
                return;
            };
            let cubelet = job.cubelet();
            let target = cubelet.map(|id| file.key(id));
            let room = room.filter(|_| cubelet.is_some());
            self.spawn_job(cubelet, target, room, move || {
                let events = job.run();
                drop(job);
                events
            });
        }
    }

    /// A cubelet the cache had when it was asked for, in the place `room` took for it, or else asked of the engine.
    fn send_hit(&mut self, room: OwnedPermit<(u32, Message)>, id: CubeletId) {
        let Some(file) = self.file.as_mut() else {
            return;
        };
        let payload = self.server.cache().get(&file.key(id).0).cloned();
        match payload {
            Some(payload) => {
                file.sent.cached += 1;
                self.send_cubelet(room, id, &payload);
            }
            // Gone from the cache since.
            None => _ = file.engine.request(Request::AddCubelets { ids: vec![id] }),
        }
    }

    /// `work` on a blocking thread, which always ends in a Done, even when it panics.
    fn spawn_job(
        &mut self,
        cubelet: Option<CubeletId>,
        target: Option<(Key, Option<f64>)>,
        room: Option<OwnedPermit<(u32, Message)>>,
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
                            server
                                .cache()
                                .insert(key.clone(), payload.clone(), payload.len());
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
            room,
            handle,
        });
    }

    /// A job's cubelet, unless it's no longer wanted, in the place it took. A job that cut nothing frees its place.
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
        if self.file.is_none() || done.cancelled.load(Ordering::Relaxed) {
            return;
        }
        // A cut has one cubelet at most, or none for a cubelet past the cube's edge.
        if let (Some(room), Some((id, payload))) = (job.room, cubelets.into_iter().next()) {
            self.send_cubelet(room, id, &payload);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use fits_core::{Cell, Cubelet, Voxels};

    use super::*;

    type Opener = (
        Session,
        mpsc::UnboundedReceiver<Done>,
        mpsc::Receiver<Message>,
    );

    /// A registered session on a server of `root`, and what it hands its loop and its socket's replies.
    fn session(root: &Path) -> Opener {
        let server = Arc::new(Server {
            root: root.canonicalize().unwrap(),
            token: String::new(),
            cache: Mutex::new(Lru::new(CACHE_BYTES)),
            cubes: Mutex::new(Cubes::new(1 << 30)),
            opens: Arc::new(Semaphore::new(1)),
            sessions: AtomicU32::new(0),
            workers: 4,
        });
        let ((done, finished), (control, controls)) =
            (mpsc::unbounded_channel(), mpsc::channel(CONTROL));
        let session = Session {
            server,
            id: 1,
            authorised: true,
            registered: true,
            file: None,
            jobs: Vec::new(),
            done,
            control,
            overflow: VecDeque::new(),
            bulk: mpsc::channel(BULK).0,
            epoch: Arc::default(),
            hits: Queue::default(),
            blocked: false,
            closing: None,
        };
        (session, finished, controls)
    }

    async fn next(finished: &mut mpsc::UnboundedReceiver<Done>) -> Done {
        let done = tokio::time::timeout(Duration::from_secs(10), finished.recv()).await;
        done.expect("a Done in time").unwrap()
    }

    #[tokio::test]
    async fn a_job_that_panics_frees_its_place_and_says_so_and_a_poisoned_cache_still_serves() {
        let (mut session, mut finished, mut outgoing) = session(&std::env::temp_dir());
        session.spawn_job(None, None, None, || panic!("on purpose"));
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
        session.spawn_job(Some(id), Some((key.clone(), None)), None, move || {
            vec![Event::Cubelet { id, cubelet }]
        });
        let done = next(&mut finished).await;
        assert!(matches!(&done.cubelets, Ok(cubelets) if cubelets.len() == 1));
        assert!(session.server.cache().get(&key).is_some());
    }

    /// A float cube of zeros, of at most 720 voxels.
    fn cube_bytes(dims: [usize; 3]) -> Vec<u8> {
        let mut bytes: Vec<u8> = ([
            "SIMPLE  =                    T",
            "BITPIX  =                  -32",
        ]
        .iter())
        .map(|card| card.to_string())
        .chain([format!("NAXIS   = {:>20}", 3)])
        .chain((1..).zip(dims).map(|(i, n)| format!("NAXIS{i}  = {n:>20}")))
        .chain(["END".to_string()])
        .flat_map(|card| format!("{card:<80}").into_bytes())
        .collect();
        bytes.resize(2 * 2880, 0);
        bytes
    }

    /// A root of its own with a small cube in it, and the OPEN_FILE for it.
    fn root_with_cube(test: &str) -> (PathBuf, Vec<u8>) {
        let root = std::env::temp_dir().join(format!("fits-server-{test}-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("cube.fits"), cube_bytes([4; 3])).unwrap();
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

    #[tokio::test]
    async fn sessions_share_a_cube_and_an_open_past_the_budget_fails_until_one_closes() {
        let (root, cube) = root_with_cube("unit-share");
        std::fs::copy(root.join("cube.fits"), root.join("other.fits")).unwrap();
        let other = proto::OpenFile {
            file: "other.fits".into(),
            file_id: 1,
            ..Default::default()
        };
        let other = frame(EventType::OpenFile, 0, &other);
        let mut sessions: Vec<_> = (0..3).map(|_| session(&root)).collect();
        let server = sessions[0].0.server.clone();
        // Room for one 4 x 4 x 4 cube.
        server.cubes().budget = 300;
        for (session, ..) in &mut sessions {
            session.server = server.clone();
        }
        async fn open(sessions: &mut [Opener], at: usize, message: &[u8]) -> proto::OpenFileAck {
            let (session, _, replies) = &mut sessions[at];
            let mut stream = futures_util::stream::pending();
            (session
                .handle(message, &mut stream, &mut VecDeque::new())
                .await)
                .unwrap();
            let Some(Message::Binary(bytes)) = replies.recv().await else {
                panic!("no ack")
            };
            carta::unframe(&bytes).unwrap().decode().unwrap()
        }
        assert!(open(&mut sessions, 0, &cube).await.success);
        assert!(open(&mut sessions, 1, &cube).await.success);
        // Read once, held by the two sessions and the server.
        let held = |server: &Server| {
            let cubes = server.cubes();
            (cubes.held.iter())
                .map(|h| (h.content_id.clone(), Arc::strong_count(&h.open)))
                .collect::<Vec<_>>()
        };
        assert!(matches!(&held(&server)[..], [(id, 3)] if id.starts_with("cube.fits:")));

        let refused = open(&mut sessions, 2, &other).await;
        assert!(
            !refused.success && refused.message.contains("not enough memory"),
            "{}",
            refused.message
        );
        // Nor is the cube a session would close room while another has it open, and it stays open.
        assert!(!open(&mut sessions, 0, &other).await.success);
        assert!(sessions[0].0.file.is_some());
        sessions[0].0.close_file();
        assert!(!open(&mut sessions, 2, &other).await.success);
        // Once neither has it open, the cube goes to make room.
        sessions[1].0.close_file();
        assert!(open(&mut sessions, 2, &other).await.success);
        assert!(matches!(&held(&server)[..], [(id, 2)] if id.starts_with("other.fits:")));
        // A cube that can't fit is refused before the one open goes.
        std::fs::write(root.join("big.fits"), cube_bytes([8; 3])).unwrap();
        let big = proto::OpenFile {
            file: "big.fits".into(),
            file_id: 2,
            ..Default::default()
        };
        assert!(
            !open(&mut sessions, 2, &frame(EventType::OpenFile, 0, &big))
                .await
                .success
        );
        assert!(sessions[2].0.file.is_some());
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn cached_cubelets_asked_for_again_before_theyve_gone_are_queued_once() {
        let (root, _) = root_with_cube("unit-hits");
        std::fs::write(root.join("wide.fits"), cube_bytes([130, 4, 1])).unwrap();
        let (mut session, _, _replies) = session(&root);
        let open = proto::OpenFile {
            file: "wide.fits".into(),
            file_id: 1,
            ..Default::default()
        };
        let mut stream = futures_util::stream::pending();
        let open = frame(EventType::OpenFile, 0, &open);
        (session
            .handle(&open, &mut stream, &mut VecDeque::new())
            .await)
            .unwrap();
        let ids: Vec<_> = (0..3)
            .map(|x| CubeletId::new(Mode::Max, Default::default(), 0, [x, 0, 0]))
            .collect();
        let file = session.file.as_ref().unwrap();
        for &id in &ids {
            let key = file.key(id).0;
            session.server.cache().insert(key, Arc::from(&[][..]), 0);
        }
        let keys = ids
            .iter()
            .map(|id| carta::cubelet_key(0, [id.x, id.y, id.z]));
        let add = proto::AddRequiredCubelets {
            file_id: 1,
            cubelets: keys.clone().chain(keys).collect(),
            ..Default::default()
        };
        // Nothing goes out, since the socket's queue is gone.
        for _ in 0..3 {
            session.add(add.clone());
        }
        assert_eq!(session.hits.len(), ids.len());
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn a_compressed_cube_that_unpacks_past_the_budget_is_refused_before_its_read() {
        let (root, _) = root_with_cube("unit-fz-budget");
        // 40 x 30 16-bit integers, fpacked, which unpack to 4.8 kB of floats.
        let fz = include_bytes!("../../fits-core/testdata/image16.fits.fz");
        std::fs::write(root.join("image.fits.fz"), fz).unwrap();
        let (mut session, _, mut replies) = session(&root);
        session.server.cubes().budget = 4000;
        let open = proto::OpenFile {
            file: "image.fits.fz".into(),
            file_id: 1,
            ..Default::default()
        };
        let mut stream = futures_util::stream::pending();
        let open = frame(EventType::OpenFile, 0, &open);
        (session
            .handle(&open, &mut stream, &mut VecDeque::new())
            .await)
            .unwrap();
        let Some(Message::Binary(bytes)) = replies.recv().await else {
            panic!("no ack")
        };
        let ack: proto::OpenFileAck = carta::unframe(&bytes).unwrap().decode().unwrap();
        assert!(
            !ack.success && ack.message.contains("not enough memory"),
            "{}",
            ack.message
        );
        assert!(session.server.cubes().held.is_empty());
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn one_port_answers_plain_and_tls_with_the_certificate_it_made() {
        use rustls::pki_types::pem::PemObject;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let dir = std::env::temp_dir().join(format!("fits-server-unit-tls-{}", std::process::id()));
        let (config, cert) = tls::own_config(&dir).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let root = dir.clone();
        tokio::spawn(
            async move { serve(listener, &root, "token".into(), Some(config), 1 << 30).await },
        );

        async fn get(mut stream: impl AsyncReadExt + AsyncWriteExt + Unpin) -> String {
            let request = b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
            stream.write_all(request).await.unwrap();
            let mut page = String::new();
            stream.read_to_string(&mut page).await.unwrap();
            page
        }
        let plain = tokio::net::TcpStream::connect(address).await.unwrap();
        assert!(get(plain).await.contains("fits-server is running"));

        // A client that trusts only that certificate takes it for 127.0.0.1.
        let mut roots = rustls::RootCertStore::empty();
        for der in rustls::pki_types::CertificateDer::pem_file_iter(&cert).unwrap() {
            roots.add(der.unwrap()).unwrap();
        }
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let client = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_root_certificates(roots)
            .with_no_client_auth();
        let connector = tokio_rustls::TlsConnector::from(Arc::new(client));
        let tcp = tokio::net::TcpStream::connect(address).await.unwrap();
        let name = rustls::pki_types::ServerName::try_from("127.0.0.1").unwrap();
        let secure = connector.connect(name, tcp).await.unwrap();
        assert!(get(secure).await.contains("fits-server is running"));

        // A pair that no longer loads, such as a key that was cut short, is made again rather than kept.
        let made = std::fs::read(&cert).unwrap();
        std::fs::write(
            dir.join("localhost-key.pem"),
            "-----BEGIN PRIVATE KEY-----\n",
        )
        .unwrap();
        let (_, again) = tls::own_config(&dir).unwrap();
        assert_ne!(std::fs::read(again).unwrap(), made);
        let _ = std::fs::remove_dir_all(dir);
    }
}
