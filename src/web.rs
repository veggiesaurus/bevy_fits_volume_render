//! Loading files handed over by web/index.html.

use std::{io::Read, sync::Mutex};

use bevy::{platform::time::Instant, prelude::*};
use js_sys::Uint8Array;

use crate::{Dataset, config, fits, ui::format_dims};

static PENDING: Mutex<Option<Dataset>> = Mutex::new(None);

/// Reads the page's copy of the file a chunk at a time, so it's never copied whole into wasm memory.
struct JsReader {
    array: Uint8Array,
    at: u32,
}

impl Read for JsReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = (self.array.length() - self.at).min(buf.len() as u32);
        self.array
            .subarray(self.at, self.at + n)
            .copy_to(&mut buf[..n as usize]);
        self.at += n;
        Ok(n as usize)
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
    let cube = fits::read_fits(
        JsReader {
            array: bytes,
            at: 0,
        },
        hdu,
    )?;
    let dataset = Dataset::new(name, cube, config, start);
    info!(
        "Loaded {} (HDU {}): {} in {:.2} s",
        dataset.name,
        dataset.cube.hdu,
        format_dims(dataset.cube.dims),
        dataset.load_seconds
    );
    *PENDING.lock().unwrap() = Some(dataset);
    Ok(())
}

pub fn poll(mut commands: Commands) {
    if let Some(dataset) = PENDING.lock().unwrap().take() {
        commands.insert_resource(dataset);
    }
}
