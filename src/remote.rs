//! A fits-server's cubes: the connection to it, a panel to browse its files and open or close one, and the data source
//! for the cube open, which sends the viewer's requests over the connection. Natively the connection is a thread of
//! its own; in the browser the Web Worker holds it.

use std::sync::{Arc, Mutex};

use bevy::{platform::time::Instant, prelude::*, render::renderer::RenderDevice};
use bevy_egui::{EguiContexts, egui};

#[cfg(not(target_arch = "wasm32"))]
use crate::fits::remote::Connection;
#[cfg(target_arch = "wasm32")]
use crate::web::{Connection, latest_pick, pick};
use crate::{
    Dataset, DatasetEntity, clear_dataset,
    config::{Config, TOLERANCES},
    fits::remote::{Budget, Command, Listing, Received},
    max_depth,
    source::{DataSource, Event, Request},
    ui::format_dims,
    upload::Upsamples,
    wake,
};

/// Natively a server's cube is the only kind to pick, so the latest open always wins.
#[cfg(not(target_arch = "wasm32"))]
fn pick() -> u32 {
    0
}

#[cfg(not(target_arch = "wasm32"))]
fn latest_pick() -> u32 {
    0
}

/// The cube open on the server, as the viewer's data source. Once it's dropped, with the view of it, the server
/// is told to close the file, over the connection it was opened on.
pub struct Source {
    connection: Connection,
    file_id: i32,
    events: Arc<Mutex<Vec<(i32, Event)>>>,
}

impl Source {
    pub fn send(&self, request: Request) {
        let file_id = self.file_id;
        self.connection.send(Command::Request { file_id, request });
    }

    /// Those for a file opened before are dropped.
    pub fn drain(&self) -> Vec<Event> {
        let events = std::mem::take(&mut *self.events.lock().unwrap());
        (events.into_iter())
            .filter_map(|(file_id, event)| (file_id == self.file_id).then_some(event))
            .collect()
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        self.connection.send(Command::Close(self.file_id));
    }
}

/// A file's dims, HDU and size.
type About = ([usize; 3], usize, u64);

#[derive(Resource)]
pub struct Remote {
    url: String,
    /// None until it connects, which in the browser waits for the token.
    connection: Option<Connection>,
    /// The token, for a server whose URL doesn't have it. It goes in the first message, never in a URL.
    api_key: String,
    config: Config,
    /// All but the events, which go straight to the source.
    received: Arc<Mutex<Vec<Received>>>,
    events: Arc<Mutex<Vec<(i32, Event)>>>,
    /// The server, once registered.
    server: Option<String>,
    closed: Option<String>,
    listing: Option<Listing>,
    /// A file picked, and what's known of it once it's in.
    picked: Option<(String, Option<About>)>,
    /// The file given on the command line or in `?path=`, opened once connected, and when the viewer started.
    start: Option<(String, Instant)>,
    /// The file being opened, when it was asked for, and its number among the viewer's picks.
    opening: Option<(String, Instant, u32)>,
    /// For the file on the command line only.
    hdu: Option<usize>,
    error: Option<String>,
    sigmas: f64,
    /// Set by the Close button: the server's cube on screen is to be closed.
    close: bool,
    /// With SCREENSHOT, the app gives up when the connection or the open does.
    exit_on_failure: bool,
}

impl Remote {
    /// A server at `url`, not yet connected to. `open` is opened once it is.
    pub fn new(
        url: String,
        config: Config,
        open: Option<String>,
        hdu: Option<usize>,
        api_key: String,
    ) -> Self {
        Self {
            url,
            connection: None,
            api_key,
            sigmas: config.zfp_tolerance_sigmas,
            config,
            received: default(),
            events: default(),
            server: None,
            closed: None,
            listing: None,
            picked: None,
            start: open.map(|path| (path, Instant::now())),
            opening: None,
            hdu,
            error: None,
            close: false,
            exit_on_failure: std::env::var_os("SCREENSHOT").is_some(),
        }
    }

    /// Connects, or connects again after the connection before closed. What comes from this connection goes to new
    /// queues, so that nothing more from the one before is taken.
    pub fn connect(&mut self) {
        let (received, events) = <(Arc<Mutex<Vec<_>>>, Arc<Mutex<Vec<_>>>)>::default();
        let (to_viewer, to_source) = (received.clone(), events.clone());
        let api_key = self.api_key.clone();
        let receive = move |message| {
            match message {
                Received::Event { file_id, event } => {
                    to_source.lock().unwrap().push((file_id, event))
                }
                message => to_viewer.lock().unwrap().push(message),
            }
            wake();
        };
        let memory = usize::try_from(self.config.cache_memory_mb.saturating_mul(1 << 20));
        let memory = memory.unwrap_or(usize::MAX);
        #[cfg(not(target_arch = "wasm32"))]
        let connection = {
            let folder = crate::config::cache_dir();
            Connection::connect(&self.url, api_key, memory, folder, receive)
        };
        #[cfg(target_arch = "wasm32")]
        let connection = Connection::connect(&self.url, api_key, memory, receive);
        self.connection = Some(connection);
        (self.received, self.events) = (received, events);
        (self.server, self.closed, self.listing, self.error) = (None, None, None, None);
        (self.picked, self.opening) = (None, None);
    }

    fn send(&self, command: Command) {
        if let Some(connection) = &self.connection {
            connection.send(command);
        }
    }

    fn open(&mut self, path: String, hdu: Option<usize>, device: &RenderDevice) {
        let budget = Budget {
            gpu_memory_limit_mb: self.config.gpu_memory_limit_mb,
            max_depth: max_depth(device),
        };
        let sigmas = self.sigmas;
        let command = Command::Open {
            path: path.clone(),
            hdu,
            budget,
            sigmas,
        };
        self.send(command);
        self.opening = Some((path, Instant::now(), pick()));
        self.error = None;
    }
}

/// What the server has sent but cubelets and values: the panel's listings and errors, and an open cube, which takes
/// the place of the one shown, or the one shown closing.
pub fn poll(
    mut commands: Commands,
    mut remote: ResMut<Remote>,
    device: Res<RenderDevice>,
    data: Option<Res<Dataset>>,
    old: Query<Entity, With<DatasetEntity>>,
    mut upsamples: ResMut<Upsamples>,
    mut exit: MessageWriter<AppExit>,
) {
    // Dropping the view drops its source, which tells the server to close the file.
    if std::mem::take(&mut remote.close)
        && data.as_ref().is_some_and(|data| data.source.is_remote())
    {
        clear_dataset(&mut commands, &old, &mut upsamples);
    }
    let received = std::mem::take(&mut *remote.received.lock().unwrap());
    let mut failed = false;
    for message in received {
        match message {
            Received::Registered { server } => {
                info!("Connected to {server}");
                remote.server = Some(server);
                remote.send(Command::List(String::new()));
                // From the command line, timed from the start.
                if let Some((path, at)) = remote.start.take() {
                    let hdu = remote.hdu.take();
                    remote.open(path, hdu, &device);
                    if let Some((_, when, _)) = remote.opening.as_mut() {
                        *when = at;
                    }
                }
            }
            Received::Listing(listing) => remote.listing = Some(listing),
            Received::Info {
                path,
                dims,
                hdu,
                bytes,
            } => {
                if let Some((picked, info)) = remote.picked.as_mut()
                    && *picked == path
                {
                    *info = Some((dims, hdu, bytes));
                }
            }
            Received::Opened(opened) => {
                let connection =
                    (remote.connection.clone()).expect("what's received comes over the connection");
                let (seconds, pick) = (remote.opening.take())
                    .map_or((0.0, 0), |(_, at, pick)| (at.elapsed().as_secs_f32(), pick));
                // A local file was picked after this open was asked for, so that file stays on screen.
                if pick != latest_pick() {
                    connection.send(Command::Close(opened.file_id));
                    continue;
                }
                let tolerance = opened
                    .tolerance
                    .map_or("exact".into(), |t| format!("within {t:.3e} at level 0"));
                let source = Source {
                    connection,
                    file_id: opened.file_id,
                    events: remote.events.clone(),
                };
                let (path, config) = (opened.path, remote.config.clone());
                let name = path.rsplit('/').next().unwrap_or(&path).to_string();
                let dataset = Dataset::new(
                    name,
                    opened.info,
                    DataSource::Remote(source),
                    config,
                    seconds,
                );
                info!(
                    "{}\nFrom {}, cubelets {tolerance}",
                    dataset.summary(),
                    remote.url_without_token()
                );
                commands.insert_resource(dataset);
            }
            Received::Failed(message) => {
                warn!("From the server: {message}");
                remote.error = Some(message);
            }
            Received::OpenFailed(message) => {
                warn!("Opening: {message}");
                remote.opening = None;
                remote.error = Some(message);
                failed = true;
            }
            Received::FileClosed { file_id, reason } => {
                warn!("Closed: {reason}");
                let shown = data.as_ref().map(|data| &data.source);
                if let Some(DataSource::Remote(source)) = shown
                    && source.file_id == file_id
                {
                    clear_dataset(&mut commands, &old, &mut upsamples);
                    failed = true;
                }
                remote.error = Some(reason);
            }
            Received::Closed(reason) => {
                error!("The server's connection: {reason}");
                remote.opening = None;
                // The first reason is the one that says what happened.
                remote.closed.get_or_insert(reason);
                failed = true;
            }
            Received::Event { .. } => {}
        }
    }
    if failed && remote.exit_on_failure {
        exit.write(AppExit::error());
    }
}

impl Remote {
    /// For the log and the panel, which have no use for the token.
    fn url_without_token(&self) -> &str {
        self.url.split('?').next().unwrap_or(&self.url)
    }
}

/// The server's files: folders to go into, cubes to pick and open, and the tolerance the next one opens at.
pub fn browser(
    mut contexts: EguiContexts,
    mut remote: ResMut<Remote>,
    device: Res<RenderDevice>,
    data: Option<Res<Dataset>>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    let remote = &mut *remote;
    egui::Window::new("Server")
        .default_width(320.0)
        .default_pos([440.0, 10.0])
        .show(ctx, |ui| {
            let url = remote.url_without_token().to_string();
            match (&remote.connection, &remote.server, &remote.closed) {
                (_, _, Some(reason)) => ui.colored_label(ui.visuals().error_fg_color, format!("{url}: {reason}")),
                (None, _, None) => ui.label(format!("Not connected to {url}")),
                (Some(_), Some(server), None) => ui.label(format!("{server} at {url}")),
                (Some(_), None, None) => ui.label(format!("Connecting to {url}")),
            };
            if remote.connection.is_none() || remote.closed.is_some() {
                connect_ui(ui, remote);
            }
            if let Some(listing) = &remote.listing {
                ui.label(format!("/{}", listing.directory));
                let (mut go, mut pick) = (None, None);
                egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                    if let Some(parent) = &listing.parent
                        && ui.selectable_label(false, "..").clicked()
                    {
                        go = Some(parent.clone());
                    }
                    let path = |name: &str| match listing.directory.as_str() {
                        "" => name.to_string(),
                        directory => format!("{directory}/{name}"),
                    };
                    for directory in &listing.directories {
                        if ui.selectable_label(false, format!("{directory}/")).clicked() {
                            go = Some(path(directory));
                        }
                    }
                    for (name, bytes) in &listing.files {
                        let file = path(name);
                        let picked = remote.picked.as_ref().is_some_and(|(p, _)| *p == file);
                        let label = format!("{name}  ({:.2} GB)", *bytes as f64 / 1e9);
                        if ui.selectable_label(picked, label).clicked() && !picked {
                            pick = Some(file);
                        }
                    }
                });
                if let Some(directory) = go {
                    remote.send(Command::List(directory));
                }
                if let Some(file) = pick {
                    remote.send(Command::Info(file.clone()));
                    remote.picked = Some((file, None));
                }
            }
            ui.horizontal(|ui| {
                ui.label("Compression");
                for sigmas in TOLERANCES {
                    ui.selectable_value(&mut remote.sigmas, sigmas, format!("{sigmas}σ"));
                }
            })
            .response
            .on_hover_text(
                "What the cubelets of the next cube opened may be off by, in units of its noise σ. The statistics, \
                 the histogram and the cursor's value are always exact",
            );
            if let Some((path, info)) = remote.picked.clone() {
                if let Some((dims, hdu, bytes)) = info {
                    ui.label(format!("{} (HDU {hdu}), {:.2} GB", format_dims(dims), bytes as f64 / 1e9));
                }
                ui.horizontal(|ui| match &remote.opening {
                    Some((opening, _, _)) => {
                        ui.label(format!("Opening {opening}"));
                        ui.spinner();
                    }
                    None => {
                        let connected = remote.server.is_some() && remote.closed.is_none();
                        if ui.add_enabled(connected, egui::Button::new("Open")).clicked() {
                            remote.open(path, None, &device);
                        }
                    }
                });
            }
            let shown = data.as_ref().is_some_and(|data| data.source.is_remote());
            if shown && ui.button("Close the cube shown").clicked() {
                remote.close = true;
                wake();
            }
            if remote.server.is_some() && remote.closed.is_none() {
                let clear = ui.button("Clear the cache").on_hover_text(
                    "Forgets the cubelets kept from servers and baked cubes, in memory and on disk, so that cubes \
                     opened again come from the server or static hosting",
                );
                if clear.clicked() {
                    remote.send(Command::ClearCache);
                }
            }
            if let Some(error) = &remote.error {
                ui.colored_label(ui.visuals().error_fg_color, error);
            }
        });
    Ok(())
}

/// In the browser a field for the token, which the page fills from a `#token=` in its URL; and a button to connect,
/// or to connect again once the connection has closed.
fn connect_ui(ui: &mut egui::Ui, remote: &mut Remote) {
    ui.horizontal(|ui| {
        #[cfg(target_arch = "wasm32")]
        let entered = {
            let field = egui::TextEdit::singleline(&mut remote.api_key)
                .password(true)
                .hint_text("Token")
                .desired_width(180.0);
            let field = ui.add(field);
            field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))
        };
        #[cfg(not(target_arch = "wasm32"))]
        let entered = false;
        let ready = !remote.api_key.is_empty() || cfg!(not(target_arch = "wasm32"));
        let button = ui.add_enabled(ready, egui::Button::new("Connect"));
        if ready && (button.clicked() || entered) {
            remote.connect();
        }
    });
}
