//! The browser's data source, in a Web Worker so that nothing heavy runs on the page's thread. web/worker.js fetches
//! or reads the file and hands it to `open`, passes the viewer's requests to `request` and calls `work` until it says
//! there's nothing left. The engine's events go back to the page as transferred ArrayBuffers. Both ways a message
//! carries the id of the page's load whose file it's about, so a request for a file the worker no longer holds is
//! dropped, and the page can tell an event for one it no longer shows.
//!
//! For a cube on a fits-server, worker.js holds the socket and the `remote_` functions turn the server's messages into
//! the viewer's and back, with fits-core's remote client, so that the cubelets are decoded here too. Those messages
//! carry the id of the viewer's connection instead. The client keeps the cubelets in a pack per cube in OPFS, whose
//! file worker.js opens.
//!
//! A baked cube is a load too, read through the `baked_` functions: worker.js reads its index and opens its pack, then
//! makes the range requests its source asks for and hands back what they bring.
#![cfg(target_arch = "wasm32")]

use std::{
    cell::RefCell,
    io::{self, ErrorKind},
};

use fits_core::{
    self as fits,
    baked::{self, Index},
    cache::Store,
    engine::{Engine, Event, Request},
    remote::{Client, Command, Received},
    wire::{self, Wire},
};
use js_sys::{
    Array, ArrayBuffer, Float32Array, Function, Object, Reflect, Uint8Array, Uint32Array,
};
use wasm_bindgen::prelude::*;

thread_local! {
    /// With the load it was opened for.
    static ENGINE: RefCell<Option<(u32, Engine)>> = const { RefCell::new(None) };
    /// A baked cube in place of a file, with its load.
    static BAKED: RefCell<Option<(u32, baked::Source)>> = const { RefCell::new(None) };
    /// The connection to a server, with the viewer's id for it, while worker.js has its socket.
    static CLIENT: RefCell<Option<(u32, Client)>> = const { RefCell::new(None) };
    /// worker.js's function that deletes every pack in OPFS but those it's given the names of, which are open.
    static CLEAR_PACKS: RefCell<Option<Function>> = const { RefCell::new(None) };
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = postMessage)]
    fn post_message(message: &JsValue, transfer: &Array);
    #[wasm_bindgen(js_namespace = console)]
    fn log(text: &str);
    #[wasm_bindgen(js_namespace = console, js_name = debug)]
    fn log_debug(text: &str);
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn log_error(text: &str);
    #[wasm_bindgen(js_namespace = performance)]
    fn now() -> f64;

    /// An OPFS file's FileSystemSyncAccessHandle, which only a dedicated worker can have.
    pub type SyncAccessHandle;
    #[wasm_bindgen(method, catch)]
    fn read(this: &SyncAccessHandle, buffer: &mut [u8], options: &Object) -> Result<f64, JsValue>;
    #[wasm_bindgen(method, catch)]
    fn write(this: &SyncAccessHandle, buffer: &[u8], options: &Object) -> Result<f64, JsValue>;
    #[wasm_bindgen(method, catch, js_name = getSize)]
    fn get_size(this: &SyncAccessHandle) -> Result<f64, JsValue>;
    #[wasm_bindgen(method, catch)]
    fn truncate(this: &SyncAccessHandle, size: f64) -> Result<(), JsValue>;
    #[wasm_bindgen(method)]
    fn close(this: &SyncAccessHandle);
}

/// A pack's OPFS file, closed once it's dropped so that the next open of it can have it.
struct Opfs(SyncAccessHandle);

fn at(offset: u64) -> Object {
    let options = Object::new();
    let _ = Reflect::set(&options, &"at".into(), &(offset as f64).into());
    options
}

impl Store for Opfs {
    fn size(&mut self) -> u64 {
        self.0.get_size().map_or(0, |size| size as u64)
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> bool {
        let len = buf.len();
        self.0
            .read(buf, &at(offset))
            .is_ok_and(|n| n as usize == len)
    }

    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> bool {
        (self.0.write(bytes, &at(offset))).is_ok_and(|n| n as usize == bytes.len())
    }

    fn truncate(&mut self, size: u64) -> bool {
        self.0.truncate(size as f64).is_ok()
    }
}

impl Drop for Opfs {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// A panic aborts, so it's posted to the page as `{failed}` first.
#[wasm_bindgen(start)]
fn start() {
    std::panic::set_hook(Box::new(|info| {
        let message = info.to_string();
        log_error(&message);
        let failed = Object::new();
        let _ = Reflect::set(&failed, &"failed".into(), &message.into());
        post_message(&failed, &Array::new());
    }));
}

/// The file as the worker has it, read a chunk at a time so it's never copied whole into wasm memory.
struct JsBytes(Uint8Array);

impl fits::Source for JsBytes {
    fn size(&self) -> io::Result<u64> {
        Ok(u64::from(self.0.length()))
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let start = u32::try_from(offset).map_err(|_| ErrorKind::UnexpectedEof)?;
        let end = start
            .checked_add(buf.len() as u32)
            .filter(|&end| end <= self.0.length())
            .ok_or(ErrorKind::UnexpectedEof)?;
        self.0.subarray(start, end).copy_to(buf);
        Ok(())
    }
}

/// Reads the file of the page's load `load` in place of the one before, which goes first so that the two are never
/// in memory together, and returns its info for the viewer. An error is thrown to worker.js.
#[wasm_bindgen]
pub fn open(load: u32, bytes: Uint8Array, hdu: Option<String>) -> Result<ArrayBuffer, String> {
    let hdu = hdu.as_deref().map(fits::parse_hdu).transpose()?;
    drop_held();
    let cube = fits::read_fits(&JsBytes(bytes), hdu)?;
    let info = Uint8Array::from(&wire::encode(&cube.info)[..]).buffer();
    ENGINE.set(Some((load, Engine::new(cube))));
    Ok(info)
}

/// The file or baked cube held goes, so that two are never in memory together.
fn drop_held() {
    ENGINE.set(None);
    BAKED.set(None);
}

/// The viewer no longer shows the file of load `load`, so it's dropped, unless a later one has taken its place.
#[wasm_bindgen]
pub fn release(load: u32) {
    if held() == Some(load) {
        drop_held();
    }
}

/// The load whose file the worker holds, if any, so that after a failed load the page can tell whether it's still
/// the one shown.
#[wasm_bindgen]
pub fn held() -> Option<u32> {
    let file = ENGINE.with_borrow(|engine| engine.as_ref().map(|(load, _)| *load));
    file.or_else(|| BAKED.with_borrow(|baked| baked.as_ref().map(|(load, _)| *load)))
}

/// One of the viewer's requests, answered at once when it needs no work.
#[wasm_bindgen]
pub fn request(message: Uint8Array) {
    let (load, request) = match wire::decode::<(u32, Request)>(&message.to_vec(), |_| None) {
        Ok(request) => request,
        Err(e) => return log_error(&format!("From the viewer: {e}")),
    };
    let baked = BAKED.with_borrow(|baked| baked.as_ref().is_some_and(|(held, _)| *held == load));
    let answer = match baked {
        true => BAKED.with_borrow_mut(|baked| baked.as_mut()?.1.request(request)),
        false => ENGINE.with_borrow_mut(|engine| {
            let (_, engine) = engine.as_mut().filter(|(held, _)| *held == load)?;
            engine.request(request)
        }),
    };
    if let Some(event) = answer {
        post(&(load, event));
    }
}

/// Runs the next job, if there is one, and says whether there was. For a baked cube, that's decoding a cubelet that
/// was in the cache.
#[wasm_bindgen]
pub fn work() -> bool {
    if let Some((load, cubelet)) = BAKED.with_borrow_mut(|baked| {
        let (load, source) = baked.as_mut().filter(|(_, source)| source.has_cached())?;
        Some((*load, source.cached()))
    }) {
        if let Some(event) = cubelet {
            post(&(load, event));
        }
        return true;
    }
    let Some((load, job)) = ENGINE.with_borrow_mut(|engine| {
        let (load, engine) = engine.as_mut()?;
        Some((*load, engine.next_job()?))
    }) else {
        return false;
    };
    let start = now();
    let events = job.run();
    let seconds = (now() - start) / 1000.0;
    if seconds >= 0.1 {
        // Cuts of the overview from level 0 are slow on a big cube, and there are a lot of them.
        let line = format!("{job} in {seconds:.2} s");
        if job.cubelet().is_some() {
            log_debug(&line);
        } else {
            log(&line);
        }
    }
    for event in events {
        post(&(load, event));
    }
    true
}

fn post(message: &(u32, Event)) {
    let buffer = buffer(message);
    post_message(&buffer, &Array::of1(&buffer));
}

/// The header and the voxels written straight into the buffer that's transferred.
fn buffer(message: &impl Wire) -> ArrayBuffer {
    let (header, voxels) = (wire::header(message), message.voxels());
    let at = header.len() as u32;
    let buffer = ArrayBuffer::new(at + 4 * voxels.len() as u32);
    Uint8Array::new_with_byte_offset_and_length(&buffer, 0, at).copy_from(&header);
    Float32Array::new_with_byte_offset_and_length(&buffer, at, voxels.len() as u32)
        .copy_from(voxels);
    buffer
}

/// The baked cube at `url` for load `load`, in place of any file or cube before, from the index at the start of its
/// file. Returns the name of its pack in OPFS, for worker.js to open and hand to `baked_attach`, and then for the
/// viewer the cube's info, its levels, three to a level, and the tolerance its cursor values are within, if they're
/// not exact.
#[wasm_bindgen]
pub fn baked_open(
    load: u32,
    index: Uint8Array,
    url: String,
    size: Option<f64>,
) -> Result<Array, String> {
    drop_held();
    let index = Index::read(&index.to_vec(), size.map(|size| size as u64))?;
    let source = baked::Source::new(&index, &url);
    let pack = JsValue::from(source.pack());
    BAKED.set(Some((load, source)));
    let info = Uint8Array::from(&wire::encode(&index.info)[..]).buffer();
    let levels: Vec<u32> = index.levels.iter().flatten().map(|&f| f as u32).collect();
    let approximate = index.tolerance.filter(|_| !index.exact_level_0);
    Ok(Array::of4(
        &pack,
        &info,
        &Uint32Array::from(&levels[..]),
        &approximate.into(),
    ))
}

/// The pack in OPFS for load `load`'s baked cube, or none when it couldn't be opened, and how long it may grow.
#[wasm_bindgen]
pub fn baked_attach(load: u32, handle: Option<SyncAccessHandle>, room: f64) {
    let store = handle.map(|handle| (Box::new(Opfs(handle)) as Box<dyn Store>, room as u64));
    BAKED.with_borrow_mut(|baked| {
        if let Some((_, source)) = baked.as_mut().filter(|(held, _)| *held == load) {
            source.attach(store);
        }
    });
}

/// The range reads the baked cube's source has asked for: its load, then the token, start and end of each.
#[wasm_bindgen]
pub fn baked_fetches() -> Vec<f64> {
    BAKED.with_borrow_mut(|baked| {
        let Some((load, source)) = baked.as_mut() else {
            return Vec::new();
        };
        let fetches = source.fetches().into_iter();
        let ranges = fetches.flat_map(|f| [f64::from(f.token), f.start as f64, f.end as f64]);
        std::iter::once(f64::from(*load)).chain(ranges).collect()
    })
}

/// Whether the read for `token` is still wanted. One that isn't is forgotten, as it won't be made.
#[wasm_bindgen]
pub fn baked_wanted(load: u32, token: u32) -> bool {
    BAKED.with_borrow_mut(|baked| {
        let (_, source) = baked.as_mut().filter(|(held, _)| *held == load)?;
        Some(source.wanted(token))
    }) == Some(true)
}

/// The read for `token` failed, after worker.js tried it again.
#[wasm_bindgen]
pub fn baked_failed(load: u32, token: u32) {
    BAKED.with_borrow_mut(|baked| {
        if let Some((_, source)) = baked.as_mut().filter(|(held, _)| *held == load) {
            source.failed(token);
        }
    });
}

/// What the read for `token` of load `load`'s cube brought.
#[wasm_bindgen]
pub fn baked_fetched(load: u32, token: u32, bytes: Uint8Array) {
    let event = BAKED.with_borrow_mut(|baked| {
        let (_, source) = baked.as_mut().filter(|(held, _)| *held == load)?;
        Some(source.fetched(token, &bytes.to_vec()))
    });
    match event {
        Some(Ok(events)) => events.into_iter().for_each(|event| post(&(load, event))),
        Some(Err(e)) => log_error(&format!("The baked cube: {e}")),
        None => {}
    }
}

/// Starts connection `id` to the server at `url`, and returns the REGISTER_VIEWER to send once the socket is open,
/// which carries the token. Up to `memory` bytes of the cubelets sent are kept in memory.
#[wasm_bindgen]
pub fn remote_start(id: u32, url: &str, api_key: String, memory: f64) -> Uint8Array {
    let (client, register) = Client::new(url, api_key, memory as usize);
    CLIENT.set(Some((id, client)));
    Uint8Array::from(&register[..])
}

/// One of the viewer's commands, as the messages to send the server. A command for a connection that's gone is
/// dropped, except Clear the cache, which is for whatever's open and comes with no connection for a baked cube.
#[wasm_bindgen]
pub fn remote_command(message: Uint8Array) -> Array {
    let (id, command) = match wire::decode::<(u32, Command)>(&message.to_vec(), |_| None) {
        Ok(message) => message,
        Err(e) => {
            log_error(&format!("From the viewer: {e}"));
            return Array::new();
        }
    };
    if let Command::ClearCache = command {
        clear_cache();
        return Array::new();
    }
    let (messages, told) = CLIENT.with_borrow_mut(|client| match client {
        Some((held, client)) if *held == id => client.command(command),
        _ => (Vec::new(), None),
    });
    if let Some(told) = told {
        post_remote(id, told);
    }
    messages.iter().map(|m| Uint8Array::from(&m[..])).collect()
}

/// Called once with worker.js's function that deletes every pack in OPFS but those it's given the names of.
#[wasm_bindgen]
pub fn opfs(clear_packs: Function) {
    CLEAR_PACKS.set(Some(clear_packs));
}

/// The cache emptied, from either of the viewer's buttons and whichever connection it came over: the server's cube's
/// pack and the baked cube's, which are open, and every other pack in OPFS.
fn clear_cache() {
    let mut open = Array::new();
    CLIENT.with_borrow_mut(|client| {
        if let Some((_, client)) = client {
            open.extend(client.pack().map(JsValue::from));
            client.clear();
        }
    });
    BAKED.with_borrow_mut(|baked| {
        if let Some((_, source)) = baked {
            open.push(&source.pack().into());
            source.clear();
        }
    });
    CLEAR_PACKS.with_borrow(|clear| clear.as_ref().map(|f| f.call1(&JsValue::NULL, &open)));
}

/// The name of the pack worker.js is to open for the file just opened, once, and then hand to `remote_attach`.
#[wasm_bindgen]
pub fn remote_pack() -> Option<String> {
    CLIENT.with_borrow_mut(|client| client.as_mut()?.1.pack_to_open())
}

/// The pack `name`'s file, or none when OPFS couldn't open it, and how long it may grow. The viewer is told of the file
/// opened once it's here.
#[wasm_bindgen]
pub fn remote_attach(name: String, handle: Option<SyncAccessHandle>, room: f64) {
    let store = handle.map(|handle| (Box::new(Opfs(handle)) as Box<dyn Store>, room as u64));
    let opened = CLIENT.with_borrow_mut(|client| {
        let (id, client) = client.as_mut()?;
        Some((*id, client.attach(&name, store)?))
    });
    if let Some((id, opened)) = opened {
        post_remote(id, opened);
    }
}

/// Decodes the next of the cubelets asked for that were in the cache, and returns what to send the server, or None
/// when there were none.
#[wasm_bindgen]
pub fn remote_work() -> Option<Array> {
    let (id, cubelet, replies) = CLIENT.with_borrow_mut(|client| {
        let (id, client) = client.as_mut()?;
        (client.has_cached()).then(|| (*id, client.cached(), client.replies()))
    })?;
    if let Some(cubelet) = cubelet {
        post_remote(id, cubelet);
    }
    Some(replies.iter().map(|m| Uint8Array::from(&m[..])).collect())
}

/// A message from the server, as the replies to send it. An error ends the connection, and worker.js closes the
/// socket and calls `remote_closed` with it.
#[wasm_bindgen]
pub fn remote_receive(message: Uint8Array) -> Result<Array, String> {
    let Some((id, received, replies)) = CLIENT.with_borrow_mut(|client| {
        let (id, client) = client.as_mut()?;
        let received = client.receive(&message.to_vec());
        Some((*id, received, client.replies()))
    }) else {
        return Ok(Array::new());
    };
    received?.into_iter().for_each(|r| post_remote(id, r));
    Ok(replies.iter().map(|m| Uint8Array::from(&m[..])).collect())
}

/// The socket has closed. The viewer is told why, once.
#[wasm_bindgen]
pub fn remote_closed(reason: String) {
    if let Some((id, _)) = CLIENT.take() {
        post_remote(id, Received::Closed(reason));
    }
}

/// As `post`, inside `{ remote }` so that the page can tell it from the local file's.
fn post_remote(id: u32, received: Received) {
    let buffer = buffer(&(id, received));
    let wrapped = Object::new();
    let _ = Reflect::set(&wrapped, &"remote".into(), &buffer);
    post_message(&wrapped, &Array::of1(&buffer));
}
