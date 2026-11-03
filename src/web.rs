//! Loading files handed over by web/index.html.

use std::{
    cell::RefCell,
    io::{self, ErrorKind},
    sync::Mutex,
};

use bevy::{
    platform::time::Instant,
    prelude::*,
    winit::{EventLoopProxy, EventLoopProxyWrapper, WinitUserEvent},
};
use js_sys::Uint8Array;

use crate::{Dataset, config, fits, ui::format_dims};

static PENDING: Mutex<Option<Dataset>> = Mutex::new(None);

thread_local! {
    // The app only updates on input, so a file handed over by the page has to wake it.
    static WAKE: RefCell<Option<EventLoopProxy<WinitUserEvent>>> = const { RefCell::new(None) };
}

pub fn keep_wake_proxy(proxy: Res<EventLoopProxyWrapper>) {
    WAKE.with(|wake| *wake.borrow_mut() = Some((*proxy).clone()));
}

/// The page's copy of the file, read a chunk at a time so it's never copied whole into wasm memory.
struct JsBytes(Uint8Array);

impl fits::Source for JsBytes {
    fn size(&self) -> io::Result<u64> {
        Ok(self.0.length() as u64)
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

/// Called from web/index.html with a picked or fetched file; `overrides` is one `key=value` per line,
/// where `hdu` picks the HDU and anything else is a config key. An error is thrown to the page.
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn load_fits(
    name: String,
    bytes: Uint8Array,
    config_toml: String,
    overrides: String,
) -> Result<(), String> {
    let start = Instant::now();
    let mut hdu = None;
    let mut config_overrides = Vec::new();
    for (key, value) in overrides.lines().filter_map(|kv| kv.split_once('=')) {
        match key.replace('-', "_").as_str() {
            "hdu" => {
                hdu = Some(
                    value
                        .parse()
                        .map_err(|_| format!("hdu needs a number, got '{value}'"))?,
                )
            }
            key => config_overrides.push((key.to_string(), value.to_string())),
        }
    }
    let config = config::from_toml(
        (!config_toml.is_empty()).then_some(config_toml.as_str()),
        &config_overrides,
    )?;
    let cube = fits::read_fits(&JsBytes(bytes), hdu)?;
    let dataset = Dataset::new(name, cube, config, start);
    info!(
        "Loaded {} (HDU {}): {} in {:.2} s",
        dataset.name,
        dataset.cube.hdu,
        format_dims(dataset.cube.dims),
        dataset.load_seconds
    );
    *PENDING.lock().unwrap() = Some(dataset);
    WAKE.with(|wake| {
        wake.borrow()
            .as_ref()
            .map(|proxy| proxy.send_event(WinitUserEvent::WakeUp))
    });
    Ok(())
}

pub fn poll(mut commands: Commands) {
    if let Some(dataset) = PENDING.lock().unwrap().take() {
        commands.insert_resource(dataset);
    }
}
