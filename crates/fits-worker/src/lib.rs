//! The browser's data source, in a Web Worker so that nothing heavy runs on the page's thread. web/worker.js fetches
//! or reads the file and hands it to `open`, passes the viewer's requests to `request` and calls `work` until it says
//! there's nothing left. The engine's events go back to the page as transferred ArrayBuffers. Both ways a message
//! carries the id of the page's load whose file it's about, so a request for a file the worker no longer holds is
//! dropped, and the page can tell an event for one it no longer shows.
#![cfg(target_arch = "wasm32")]

use std::{
    cell::RefCell,
    io::{self, ErrorKind},
};

use fits_core::{
    self as fits,
    engine::{Engine, Event, Request},
    wire::{self, Wire},
};
use js_sys::{Array, ArrayBuffer, Float32Array, Object, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;

thread_local! {
    /// With the load it was opened for.
    static ENGINE: RefCell<Option<(u32, Engine)>> = const { RefCell::new(None) };
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
    ENGINE.set(None);
    let cube = fits::read_fits(&JsBytes(bytes), hdu)?;
    let info = Uint8Array::from(&wire::encode(&cube.info)[..]).buffer();
    ENGINE.set(Some((load, Engine::new(cube))));
    Ok(info)
}

/// The load whose file the worker holds, if any, so that after a failed load the page can tell whether it's still
/// the one shown.
#[wasm_bindgen]
pub fn held() -> Option<u32> {
    ENGINE.with_borrow(|engine| engine.as_ref().map(|(load, _)| *load))
}

/// One of the viewer's requests, answered at once when it needs no work.
#[wasm_bindgen]
pub fn request(message: Uint8Array) {
    let (load, request) = match wire::decode::<(u32, Request)>(&message.to_vec(), |_| None) {
        Ok(request) => request,
        Err(e) => return log_error(&format!("From the viewer: {e}")),
    };
    let answer = ENGINE.with_borrow_mut(|engine| {
        let (_, engine) = engine.as_mut().filter(|(held, _)| *held == load)?;
        engine.request(request)
    });
    if let Some(event) = answer {
        post(&(load, event));
    }
}

/// Runs the next job, if there is one, and says whether there was.
#[wasm_bindgen]
pub fn work() -> bool {
    let Some((load, job)) = ENGINE.with_borrow_mut(|engine| {
        let (load, engine) = engine.as_mut()?;
        Some((*load, engine.next_job()?))
    }) else {
        return false;
    };
    let start = now();
    let event = job.run();
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
    if let Some(event) = event {
        post(&(load, event));
    }
    true
}

/// Writes the header and the voxels straight into the buffer that's transferred.
fn post(message: &(u32, Event)) {
    let (header, voxels) = (wire::header(message), message.voxels());
    let at = header.len() as u32;
    let buffer = ArrayBuffer::new(at + 4 * voxels.len() as u32);
    Uint8Array::new_with_byte_offset_and_length(&buffer, 0, at).copy_from(&header);
    Float32Array::new_with_byte_offset_and_length(&buffer, at, voxels.len() as u32)
        .copy_from(voxels);
    post_message(&buffer, &Array::of1(&buffer));
}
