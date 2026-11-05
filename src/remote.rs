//! A fits-server's cubes, natively: the connection to it, a panel to browse its files and open one, and the data
//! source for the cube open, which sends the viewer's requests over the connection.

use std::sync::{Arc, Mutex};

use bevy::{platform::time::Instant, prelude::*, render::renderer::RenderDevice};
use bevy_egui::{EguiContexts, egui};

use crate::{
    Dataset, DatasetEntity, clear_dataset,
    config::{Config, TOLERANCES},
    fits::remote::{Budget, Command, Connection, Listing, Received},
    max_depth,
    source::{DataSource, Event, Request},
    ui::format_dims,
    upload::Upsamples,
    wake,
};

/// The cube open on the server, as the viewer's data source.
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

/// A file's dims, HDU and size.
type About = ([usize; 3], usize, u64);

#[derive(Resource)]
pub struct Remote {
    url: String,
    connection: Connection,
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
    /// The file on its way, and when it was asked for; at start the one given on the command line.
    opening: Option<(String, Instant)>,
    /// The file ID of the cube shown.
    shown: Option<i32>,
    /// For the file on the command line only.
    hdu: Option<usize>,
    error: Option<String>,
    sigmas: f64,
    /// With SCREENSHOT, the app gives up when the connection or the open does.
    exit_on_failure: bool,
}

impl Remote {
    pub fn connect(url: String, config: Config, open: Option<String>, hdu: Option<usize>) -> Self {
        let (received, events) = <(Arc<Mutex<Vec<_>>>, Arc<Mutex<Vec<_>>>)>::default();
        let (to_viewer, to_source) = (received.clone(), events.clone());
        let connection = Connection::connect(&url, move |message| {
            match message {
                Received::Event { file_id, event } => {
                    to_source.lock().unwrap().push((file_id, event))
                }
                message => to_viewer.lock().unwrap().push(message),
            }
            wake();
        });
        Self {
            url,
            connection,
            sigmas: config.zfp_tolerance_sigmas,
            config,
            received,
            events,
            server: None,
            closed: None,
            listing: None,
            picked: None,
            opening: open.map(|path| (path, Instant::now())),
            shown: None,
            hdu,
            error: None,
            exit_on_failure: std::env::var_os("SCREENSHOT").is_some(),
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
        self.connection.send(command);
        self.opening = Some((path, Instant::now()));
        self.error = None;
    }
}

/// What the server has sent but cubelets and values: the panel's listings and errors, and an open cube, which takes
/// the place of the one shown, or the one shown closing.
pub fn poll(
    mut commands: Commands,
    mut remote: ResMut<Remote>,
    device: Res<RenderDevice>,
    old: Query<Entity, With<DatasetEntity>>,
    mut upsamples: ResMut<Upsamples>,
    mut exit: MessageWriter<AppExit>,
) {
    let received = std::mem::take(&mut *remote.received.lock().unwrap());
    let mut failed = false;
    for message in received {
        match message {
            Received::Registered { server } => {
                info!("Connected to {server}");
                remote.server = Some(server);
                remote.connection.send(Command::List(String::new()));
                // From the command line, timed from the start.
                if let Some((path, at)) = remote.opening.take() {
                    let hdu = remote.hdu.take();
                    remote.open(path, hdu, &device);
                    remote.opening = remote.opening.take().map(|(path, _)| (path, at));
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
                let seconds =
                    (remote.opening.take()).map_or(0.0, |(_, at)| at.elapsed().as_secs_f32());
                let tolerance = opened
                    .tolerance
                    .map_or("exact".into(), |t| format!("within {t:.3e} at level 0"));
                let source = Source {
                    connection: remote.connection.clone(),
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
                remote.shown = Some(opened.file_id);
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
                if remote.shown == Some(file_id) {
                    remote.shown = None;
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
) -> Result {
    let ctx = contexts.ctx_mut()?;
    let remote = &mut *remote;
    egui::Window::new("Server")
        .default_width(320.0)
        .default_pos([440.0, 10.0])
        .show(ctx, |ui| {
            let url = remote.url_without_token().to_string();
            match (&remote.server, &remote.closed) {
                (_, Some(reason)) => ui.colored_label(ui.visuals().error_fg_color, format!("{url}: {reason}")),
                (Some(server), None) => ui.label(format!("{server} at {url}")),
                (None, None) => ui.label(format!("Connecting to {url}")),
            };
            if let Some(listing) = &remote.listing {
                ui.label(format!("/{}", listing.directory));
                let mut go = None;
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
                            remote.connection.send(Command::Info(file.clone()));
                            remote.picked = Some((file, None));
                        }
                    }
                });
                if let Some(directory) = go {
                    remote.connection.send(Command::List(directory));
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
                    Some((opening, _)) => {
                        ui.label(format!("Opening {opening}"));
                        ui.spinner();
                    }
                    None => {
                        if ui.add_enabled(remote.closed.is_none(), egui::Button::new("Open")).clicked() {
                            remote.open(path, None, &device);
                        }
                    }
                });
            }
            if let Some(error) = &remote.error {
                ui.colored_label(ui.visuals().error_fg_color, error);
            }
        });
    Ok(())
}
