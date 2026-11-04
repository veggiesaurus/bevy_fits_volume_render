//! The browser's data source. web/index.html hands the file to a Web Worker (web/worker.js and fits-worker), which
//! reads it and runs fits-core's engine on it there, and gives the viewer a way to post to the worker. What the
//! worker sends back comes in here: a file's info once it's read, then the source's events. Both ways a message
//! carries the id of the page's load whose file it's about.

use std::{cell::RefCell, sync::Mutex};

use bevy::prelude::*;
use js_sys::{ArrayBuffer, Float32Array, Function, Uint8Array};
use wasm_bindgen::{JsValue, prelude::wasm_bindgen};

use crate::{
    Dataset, DatasetEntity,
    axes::AxisAnnotation,
    config,
    fits::{
        FileInfo,
        wire::{self, Wire},
    },
    source::{Event, Request},
    upload::Upsamples,
    view::CubeState,
    wake,
};

/// The file to show next, or None to show none.
static PENDING: Mutex<Option<Option<Dataset>>> = Mutex::new(None);
static EVENTS: Mutex<Vec<(u32, Event)>> = Mutex::new(Vec::new());

thread_local! {
    static POST: RefCell<Option<Function>> = const { RefCell::new(None) };
}

/// The worker, for the file of one of the page's loads: requests go out with its id, and only events with it come
/// back.
pub struct DataSource {
    load: u32,
}

impl DataSource {
    pub fn send(&self, request: Request) {
        let message = Uint8Array::from(&wire::encode(&(self.load, request))[..]).buffer();
        POST.with_borrow(|post| match post {
            Some(post) => {
                if let Err(e) = post.call1(&JsValue::NULL, &message) {
                    error!("Posting to the worker: {e:?}");
                }
            }
            None => error!("No worker to post to"),
        });
    }

    pub fn drain(&self) -> Vec<Event> {
        let events = std::mem::take(&mut *EVENTS.lock().unwrap());
        (events.into_iter())
            .filter_map(|(load, event)| (load == self.load).then_some(event))
            .collect()
    }
}

/// Called once from web/index.html with a function that posts a message to the worker, transferring its buffer.
#[wasm_bindgen]
pub fn connect(post: Function) {
    POST.set(Some(post));
}

/// Called from web/index.html when the worker has read the file of load `load` in `seconds`; `overrides` is one
/// `key=value` config key per line. An error is thrown to the page.
#[wasm_bindgen]
pub fn opened(
    load: u32,
    name: String,
    info: ArrayBuffer,
    seconds: f32,
    config_toml: String,
    overrides: String,
) -> Result<(), String> {
    let overrides: Vec<(String, String)> = (overrides.lines())
        .filter_map(|kv| kv.split_once('='))
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();
    let config = config::from_toml(
        (!config_toml.is_empty()).then_some(config_toml.as_str()),
        &overrides,
    )?;
    let info: FileInfo = decode(&info)?;
    let dataset = Dataset::new(name, info, DataSource { load }, config, seconds);
    info!("{}", dataset.summary());
    *PENDING.lock().unwrap() = Some(Some(dataset));
    // The app only updates on input, so a file handed over by the page has to wake it.
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
    old: Query<Entity, With<DatasetEntity>>,
    mut upsamples: ResMut<Upsamples>,
) {
    match PENDING.lock().unwrap().take() {
        Some(Some(dataset)) => commands.insert_resource(dataset),
        Some(None) => {
            for entity in &old {
                commands.entity(entity).despawn();
            }
            upsamples.0.clear();
            commands.remove_resource::<Dataset>();
            commands.remove_resource::<CubeState>();
            commands.remove_resource::<AxisAnnotation>();
        }
        None => {}
    }
}
