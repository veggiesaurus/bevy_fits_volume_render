//! The browser's data source. web/index.html hands the file to a Web Worker (web/worker.js and fits-worker), which
//! reads it and runs fits-core's engine on it there, and gives the viewer a way to post to the worker. What the
//! worker sends back comes in here: a file's info once it's read, then the source's events. Both ways a message
//! carries the id of the page's load whose file it's about.
//!
//! The worker also holds the socket to a fits-server, for `?server=`. The viewer's commands for it go to the worker,
//! and what the server sends comes back here decoded, both ways with the id of the viewer's connection.

use std::{
    cell::{Cell, RefCell},
    sync::{
        Mutex,
        atomic::{AtomicU32, Ordering},
    },
};

use bevy::prelude::*;
use js_sys::{ArrayBuffer, Float32Array, Function, Uint8Array};
use wasm_bindgen::{JsValue, prelude::wasm_bindgen};

use crate::{
    Dataset, DatasetEntity, config,
    fits::{
        FileInfo,
        remote::{Command, Received},
        wire::{self, Wire},
    },
    remote::Remote,
    source::{DataSource, Event, Request},
    upload::Upsamples,
    wake,
};

/// The file to show next, or None to show none.
static PENDING: Mutex<Option<Option<Dataset>>> = Mutex::new(None);
static EVENTS: Mutex<Vec<(u32, Event)>> = Mutex::new(Vec::new());
/// The server panel to show, from `?server=`.
static SERVER: Mutex<Option<Remote>> = Mutex::new(None);
/// How many files have been picked, a file in the page or a cube on a server, so that only the latest is shown.
static PICKS: AtomicU32 = AtomicU32::new(0);

type Receive = Box<dyn FnMut(Received)>;

thread_local! {
    static POST: RefCell<Option<Function>> = const { RefCell::new(None) };
    static CONNECT: RefCell<Option<Function>> = const { RefCell::new(None) };
    static RELEASE: RefCell<Option<Function>> = const { RefCell::new(None) };
    /// The latest connection's id, and the function that takes what arrives for it.
    static REMOTE: RefCell<Option<(u32, Receive)>> = const { RefCell::new(None) };
    static CONNECTIONS: Cell<u32> = const { Cell::new(0) };
}

fn post(message: ArrayBuffer, remote: bool) {
    POST.with_borrow(|post| match post {
        Some(post) => {
            if let Err(e) = post.call2(&JsValue::NULL, &message, &remote.into()) {
                error!("Posting to the worker: {e:?}");
            }
        }
        None => error!("No worker to post to"),
    });
}

/// Called from web/index.html as a file starts to load, and by the server panel as it opens one. A file that comes
/// in after a later pick isn't shown.
#[wasm_bindgen]
pub fn pick() -> u32 {
    PICKS.fetch_add(1, Ordering::Relaxed) + 1
}

pub fn latest_pick() -> u32 {
    PICKS.load(Ordering::Relaxed)
}

/// Tells the worker it can drop the file of load `load`, since nothing shows it any more.
fn release(load: u32) {
    RELEASE.with_borrow(|release| {
        if let Some(release) = release {
            let _ = release.call1(&JsValue::NULL, &load.into());
        }
    });
}

/// The worker, for the file of one of the page's loads: requests go out with its id, and only events with it come
/// back. Once it's dropped, with the view of it, the worker lets go of the file.
pub struct WorkerSource {
    load: u32,
}

impl Drop for WorkerSource {
    fn drop(&mut self) {
        release(self.load);
    }
}

impl WorkerSource {
    pub fn send(&self, request: Request) {
        post(
            Uint8Array::from(&wire::encode(&(self.load, request))[..]).buffer(),
            false,
        );
    }

    pub fn drain(&self) -> Vec<Event> {
        let events = std::mem::take(&mut *EVENTS.lock().unwrap());
        (events.into_iter())
            .filter_map(|(load, event)| (load == self.load).then_some(event))
            .collect()
    }
}

/// A connection to a fits-server, held by the worker. Commands go out with its id, and what comes back for an older
/// connection is dropped, so that file IDs from one can't be taken for another's.
#[derive(Clone)]
pub struct Connection(u32);

impl Connection {
    /// `receive` is called with what the server sends, last with `Received::Closed`. Up to `memory` bytes of the
    /// cubelets sent are kept in memory, and the worker keeps them in OPFS as well.
    pub fn connect(
        url: &str,
        api_key: String,
        memory: usize,
        receive: impl FnMut(Received) + 'static,
    ) -> Self {
        let id = CONNECTIONS.get() + 1;
        CONNECTIONS.set(id);
        REMOTE.set(Some((id, Box::new(receive))));
        let called = CONNECT.with_borrow(|connect| {
            connect.as_ref().map(|connect| {
                let (url, api_key) = (url.into(), api_key.into());
                connect.call4(&JsValue::NULL, &url, &api_key, &id.into(), &memory.into())
            })
        });
        if !matches!(called, Some(Ok(_))) {
            to_connection(
                id,
                Received::Closed("couldn't reach the data worker".into()),
            );
        }
        Self(id)
    }

    pub fn send(&self, command: Command) {
        post(
            Uint8Array::from(&wire::encode(&(self.0, command))[..]).buffer(),
            true,
        );
    }
}

/// Called once from web/index.html with the functions that post a message to the worker, transferring its buffer,
/// that have it connect to a server, and that have it let go of a file.
#[wasm_bindgen]
pub fn connect(post: Function, connect: Function, release: Function) {
    POST.set(Some(post));
    CONNECT.set(Some(connect));
    RELEASE.set(Some(release));
}

/// The config from `config.toml` and the page's query parameters, one `key=value` per line.
fn page_config(config_toml: &str, overrides: &str) -> Result<config::Config, String> {
    let overrides: Vec<(String, String)> = (overrides.lines())
        .filter_map(|kv| kv.split_once('='))
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();
    config::from_toml((!config_toml.is_empty()).then_some(config_toml), &overrides)
}

/// Called from web/index.html when the worker has read the file of load `load` in `seconds`, which was the viewer's
/// pick `pick`. An error is thrown to the page.
#[wasm_bindgen]
pub fn opened(
    load: u32,
    name: String,
    info: ArrayBuffer,
    seconds: f32,
    config_toml: String,
    overrides: String,
    pick: u32,
) -> Result<(), String> {
    // A cube was opened on a server after this load began, so that one stays on screen.
    if pick != latest_pick() {
        release(load);
        return Ok(());
    }
    let config = page_config(&config_toml, &overrides)?;
    let info: FileInfo = decode(&info)?;
    let source = DataSource::Local(WorkerSource { load });
    let dataset = Dataset::new(name, info, source, config, seconds);
    info!("{}", dataset.summary());
    *PENDING.lock().unwrap() = Some(Some(dataset));
    // The app only updates on input, so a file handed over by the page has to wake it.
    wake();
    Ok(())
}

/// Called from web/index.html for `?server=url`, with the token from the page's `#token=` if it had one, and `path`
/// to open once connected. With the token it connects at once; without, the panel asks for it.
#[wasm_bindgen]
pub fn server(
    url: String,
    token: String,
    path: Option<String>,
    hdu: Option<String>,
    config_toml: String,
    overrides: String,
) -> Result<(), String> {
    let config = page_config(&config_toml, &overrides)?;
    let hdu = hdu.as_deref().map(crate::fits::parse_hdu).transpose()?;
    let connect = !token.is_empty();
    let mut remote = Remote::new(url, config, path, hdu, token);
    if connect {
        remote.connect();
    }
    *SERVER.lock().unwrap() = Some(remote);
    wake();
    Ok(())
}

/// Called from web/index.html when the worker no longer holds the file shown, which goes, since nothing would
/// answer for it.
#[wasm_bindgen]
pub fn clear() {
    *PENDING.lock().unwrap() = Some(None);
    wake();
}

/// Called from web/index.html with each of the source's events.
#[wasm_bindgen]
pub fn receive(message: ArrayBuffer) {
    match decode(&message) {
        Ok(event) => {
            EVENTS.lock().unwrap().push(event);
            wake();
        }
        Err(e) => error!("From the worker: {e}"),
    }
}

/// Called from web/index.html with each of the worker's messages for a server's connection.
#[wasm_bindgen]
pub fn receive_remote(message: ArrayBuffer) {
    match decode::<(u32, Received)>(&message) {
        Ok((id, received)) => to_connection(id, received),
        Err(e) => error!("From the worker: {e}"),
    }
}

/// Called from web/index.html when the worker has failed, and with it the connection it held.
#[wasm_bindgen]
pub fn server_lost(reason: String) {
    to_connection(CONNECTIONS.get(), Received::Closed(reason));
}

fn to_connection(id: u32, received: Received) {
    REMOTE.with_borrow_mut(|remote| {
        if let Some((latest, receive)) = remote
            && *latest == id
        {
            receive(received);
        }
    });
}

/// A message read out of its buffer: the header, then any voxels copied straight into a vector of their own.
fn decode<M: Wire>(message: &ArrayBuffer) -> Result<M, String> {
    let bytes = Uint8Array::new(message);
    let mut prefix = [0; 4];
    if bytes.length() >= 4 {
        bytes.subarray(0, 4).copy_to(&mut prefix);
    }
    let header_len = wire::header_len(prefix).min(bytes.length() as usize);
    let header = bytes.subarray(0, header_len as u32).to_vec();
    wire::decode(&header, |n| {
        let fits = header_len.is_multiple_of(4) && header_len + 4 * n <= bytes.length() as usize;
        fits.then(|| {
            Float32Array::new_with_byte_offset_and_length(message, header_len as u32, n as u32)
                .to_vec()
        })
    })
}

pub fn poll(
    mut commands: Commands,
    data: Option<Res<Dataset>>,
    old: Query<Entity, With<DatasetEntity>>,
    mut upsamples: ResMut<Upsamples>,
) {
    if let Some(remote) = SERVER.lock().unwrap().take() {
        commands.insert_resource(remote);
    }
    match PENDING.lock().unwrap().take() {
        Some(Some(dataset)) => commands.insert_resource(dataset),
        // Only a file from the worker: a server's cube isn't the worker's to clear.
        Some(None) if data.is_none_or(|data| !data.source.is_remote()) => {
            crate::clear_dataset(&mut commands, &old, &mut upsamples)
        }
        _ => {}
    }
}
