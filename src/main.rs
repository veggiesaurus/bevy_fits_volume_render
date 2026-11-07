mod axes;
mod camera;
mod config;
mod grips;
mod modes;
mod outline;
mod remote;
mod selection;
mod shape;
mod source;
mod ui;
mod upload;
mod view;
mod volume;
mod wcs;
#[cfg(target_arch = "wasm32")]
mod web;

use std::{sync::OnceLock, time::Duration};

use bevy::{
    diagnostic::FrameTimeDiagnosticsPlugin,
    platform::time::Instant,
    prelude::*,
    render::{
        RenderPlugin,
        renderer::RenderDevice,
        settings::{WgpuFeatures, WgpuSettings},
        storage::ShaderBuffer,
    },
    window::RequestRedraw,
    winit::{EventLoopProxy, EventLoopProxyWrapper, WinitSettings, WinitUserEvent},
};
use bevy_egui::{EguiPlugin, EguiPrimaryContextPass};

use fits_core as fits;

use axes::AxisAnnotation;
use camera::Orbit;
use config::Config;
use fits::{FileInfo, Region};
use selection::{CropRequest, Gesture, Hover};
use shape::BoxShape;
use source::{DataSource, Event, Mode, Request};
use upload::{UploadPlugin, Uploads, Upsamples};
use view::{CubeState, Levels, VolumeTexture};
use volume::{VolumeMaterial, VolumeParams, VolumePlugin};

/// Bevy's local space is right-handed; iDaVIE's (Unity) object space mirrors z. Its own inverse, so it
/// converts either way. `BoxShape` adds the box's scale.
const UNITY_FROM_LOCAL: Vec3 = Vec3::new(1.0, 1.0, -1.0);
/// iDaVIE's VolumeDataSetRenderer.SelectionSaturateFactor.
const SELECTION_SATURATE_FACTOR: f32 = 0.7;

static WAKE: OnceLock<EventLoopProxy<WinitUserEvent>> = OnceLock::new();

/// Wakes the app from its reactive wait, from any thread: the data source has something for it.
pub fn wake() {
    if let Some(proxy) = WAKE.get() {
        let _ = proxy.send_event(WinitUserEvent::WakeUp);
    }
}

fn keep_wake_proxy(proxy: Res<EventLoopProxyWrapper>) {
    let _ = WAKE.set((**proxy).clone());
}

#[derive(Resource)]
struct Dataset {
    name: String,
    info: FileInfo,
    source: DataSource,
    config: Config,
    wcs: Option<wcs::Wcs>,
    load_seconds: f32,
    /// The levels a baked cube has, which the view takes in place of its own.
    levels: Option<Vec<[usize; 3]>>,
    /// What the cursor's values may be off by, when they come from compressed cubelets.
    approximate: Option<f64>,
}

impl Dataset {
    fn new(
        name: String,
        info: FileInfo,
        source: DataSource,
        config: Config,
        load_seconds: f32,
    ) -> Self {
        Self {
            name,
            wcs: wcs::Wcs::from_cards(&info.cards, info.axes).map(|wcs| {
                match info.stokes_plane() {
                    Some((axis, plane)) => wcs.at_plane(axis, plane),
                    None => wcs,
                }
            }),
            info,
            source,
            config,
            load_seconds,
            levels: None,
            approximate: None,
        }
    }

    /// What was loaded, for the terminal or the browser console.
    fn summary(&self) -> String {
        let s = self.info.stats;
        let mut summary = format!(
            "Loaded {} (HDU {}): {} in {:.2} s\nRange {} .. {}, mean {:.4e}, std {:.4e}, {} non-finite voxels",
            self.name,
            self.info.hdu,
            ui::format_dims(self.info.dims),
            self.load_seconds,
            s.min,
            s.max,
            s.mean,
            s.std,
            s.non_finite
        );
        if let Some(planes) = self.info.other_planes() {
            summary.push_str(&format!("\nWarning: showing plane {planes}"));
        }
        summary
    }
}

#[derive(Resource, Clone, PartialEq)]
struct Settings {
    params: VolumeParams,
    modes: modes::ModeSettings,
    wcs_axes: bool,
    /// The z axis's scale relative to cubic voxels.
    z_scale: f32,
}

impl Settings {
    /// The config's rendering defaults, scaled to the cube's range.
    fn new(data: &Dataset) -> Self {
        let (config, dims) = (&data.config, data.info.dims);
        let asked = config.z_scale.value(dims);
        let range = shape::z_scale_range(dims);
        let z_scale = asked.clamp(*range.start(), *range.end());
        if z_scale != asked {
            warn!("z_scale {asked} is outside {range:?} for this cube, so it opens at {z_scale}");
        }
        Self {
            params: VolumeParams {
                scale_min: data.info.stats.min,
                scale_max: data.info.stats.max,
                jitter: config.jitter,
                threshold_min: config.threshold_min,
                threshold_max: config.threshold_max,
                max_steps: config.max_raymarching_steps,
                color_map_index: config.default_color_map.0,
                scale_type: config.default_scaling_type.0,
                ..default()
            },
            modes: default(),
            wcs_axes: true,
            z_scale,
        }
    }

    /// The pyramid the projection mode reads.
    fn mode(&self) -> Mode {
        if self.params.projection_mode == 1 {
            Mode::Mean
        } else {
            Mode::Max
        }
    }
}

/// Everything spawned for the loaded file, despawned when another is loaded.
#[derive(Component)]
struct DatasetEntity;

fn main() -> AppExit {
    #[cfg(not(target_arch = "wasm32"))]
    let start = load_native();

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "iDaVIE volume renderer (Bevy)".into(),
                    canvas: Some("#bevy".into()),
                    fit_canvas_to_parent: true,
                    ..default()
                }),
                ..default()
            })
            // Keeps R32Float unfilterable on every device, which the cube's non-filtering sampler needs.
            .set(RenderPlugin {
                render_creation: WgpuSettings {
                    disabled_features: Some(WgpuFeatures::FLOAT32_FILTERABLE),
                    ..default()
                }
                .into(),
                ..default()
            }),
    )
    .add_plugins((
        EguiPlugin::default(),
        FrameTimeDiagnosticsPlugin::default(),
        VolumePlugin,
        UploadPlugin,
        modes::ModesPlugin,
    ))
    .insert_resource(ClearColor(Color::BLACK))
    .init_resource::<Hover>()
    .init_resource::<grips::Grips>()
    .init_resource::<Gesture>()
    .add_message::<CropRequest>()
    .add_systems(Startup, (camera::spawn_camera, keep_wake_proxy))
    .add_systems(PreUpdate, receive.run_if(resource_exists::<Dataset>))
    .add_systems(
        Update,
        (
            selection::end_gesture,
            spawn_volume.run_if(resource_exists_and_changed::<Dataset>),
            // Grips go first so grabbing one takes the click away from orbiting and shift-drag selection,
            // and a crop or a new z scale retargets the orbit before the camera moves this frame.
            (
                shape::reshape.run_if(resource_exists_and_changed::<Settings>),
                grips::drag_grips,
                selection::select_region,
                selection::selection_keys,
                selection::apply_crop,
                view::refine,
            )
                .chain()
                .run_if(resource_exists::<CubeState>),
            camera::orbit_camera,
            // After the orbit, so hidden edges are shaded and grips drawn from this frame's camera.
            (
                update_material.run_if(
                    resource_exists_and_changed::<Settings>
                        .or_eager(resource_exists_and_changed::<CubeState>),
                ),
                show_volume,
                axes::update_axes,
                // Each outline is moved before its hidden edges are worked out.
                (
                    outline::update_outlines.run_if(
                        resource_exists_and_changed::<CubeState>
                            .or_eager(resource_exists_and_changed::<BoxShape>),
                    ),
                    outline::shade_hidden_edges,
                )
                    .chain(),
                grips::draw_grips,
            )
                .run_if(resource_exists::<CubeState>),
        )
            .chain(),
    )
    // Before retire_uploads, so that a finished upload still asks for the frame that shows its texture.
    .add_systems(Last, keep_redrawing.before(upload::retire_uploads))
    .add_systems(
        EguiPrimaryContextPass,
        (
            ui::ui,
            axes::draw_axes.run_if(resource_exists::<AxisAnnotation>),
        ),
    );

    // Redraw only on input, so an idle view doesn't keep the GPU ray-marching. SCREENSHOT counts frames, so
    // it keeps Bevy's continuous default.
    let continuous = cfg!(not(target_arch = "wasm32")) && std::env::var("SCREENSHOT").is_ok();
    if !continuous {
        app.insert_resource(WinitSettings::desktop_app());
    }

    #[cfg(target_arch = "wasm32")]
    app.add_systems(PreUpdate, web::poll);
    // A fits-server, natively from --server and in the browser from ?server=.
    app.add_systems(
        PreUpdate,
        remote::poll
            .before(receive)
            .run_if(resource_exists::<remote::Remote>),
    )
    .add_systems(
        EguiPrimaryContextPass,
        remote::browser.run_if(resource_exists::<remote::Remote>),
    );
    #[cfg(not(target_arch = "wasm32"))]
    {
        app.add_systems(Last, log_view.run_if(resource_exists::<Dataset>));
        match start {
            Start::Local(dataset) => {
                app.insert_resource(dataset);
            }
            Start::Remote(remote) => {
                app.insert_resource(remote);
            }
        }
        if let Ok(path) = std::env::var("SCREENSHOT") {
            // Frames count from when the view is in, however long the source took to cut it.
            let settled = |state: Option<Res<CubeState>>, uploads: Res<Uploads>| {
                state.is_some_and(|state| state.settled(&uploads))
            };
            app.add_systems(Update, screenshot_and_exit(path).run_if(settled));
        }
    }

    app.run()
}

/// Takes what the data source has sent since the last frame. Only `view::refine` marks the state changed, once a
/// frame at most.
fn receive(
    data: Res<Dataset>,
    mut uploads: ResMut<Uploads>,
    mut hover: ResMut<Hover>,
    mut state: Option<ResMut<CubeState>>,
    mut extras: ResMut<modes::Extras>,
) {
    for event in data.source.drain() {
        match event {
            Event::Cubelet { id, cubelet } => {
                if let Some(state) = state.as_mut() {
                    state
                        .bypass_change_detection()
                        .receive(id, cubelet, &mut uploads);
                }
            }
            Event::PixelValue { id, value } => hover.receive(id, value),
            event => modes::receive(event, &mut extras, &mut uploads),
        }
    }
}

/// Frames otherwise only come with input or when the data source wakes the app, so keep asking for them while a
/// texture is still being written or upsampled, or still on its way to the screen.
fn keep_redrawing(
    uploads: Res<Uploads>,
    state: Option<Res<CubeState>>,
    mut settle_until: Local<Option<Instant>>,
    mut redraw: MessageWriter<RequestRedraw>,
) {
    if state.as_ref().is_some_and(|state| state.is_changed()) {
        *settle_until = Some(Instant::now() + Duration::from_secs(1));
    }
    if state.is_some_and(|state| state.next.is_some())
        || !uploads.is_empty()
        || settle_until.is_some_and(|until| Instant::now() < until)
    {
        redraw.write(RequestRedraw);
    }
}

/// When the view of a cube is first drawn and when it's all in, counted from when the open began, which for a local
/// file is when it began to be read.
#[cfg(not(target_arch = "wasm32"))]
fn log_view(
    data: Res<Dataset>,
    state: Option<Res<CubeState>>,
    uploads: Res<Uploads>,
    mut since: Local<Option<(Instant, bool)>>,
) {
    if data.is_changed() {
        *since = Some((
            Instant::now() - Duration::from_secs_f32(data.load_seconds),
            false,
        ));
    }
    let (Some(state), Some((start, drawn))) = (state, since.as_mut()) else {
        return;
    };
    let seconds = start.elapsed().as_secs_f32();
    if !*drawn && state.shown().drawable() {
        *drawn = true;
        info!(
            "{}: first drawn {seconds:.2} s after the open began",
            data.name
        );
    }
    let texture = state.shown();
    if *drawn
        && state.settled(&uploads)
        && state
            .levels
            .shows(texture.level, texture.texels, state.visible)
    {
        let [x, y, z] = texture.factors;
        info!(
            "{}: all in at {x} x {y} x {z}, {seconds:.2} s after the open began",
            data.name
        );
        *since = None;
    }
}

/// Frames since rendering settled, and when the 30th was.
#[cfg(not(target_arch = "wasm32"))]
type Frames<'a> = Local<'a, (u32, Option<Instant>)>;

/// `SCREENSHOT=out.png` saves the window once rendering has settled, logs the frame time after it, then exits.
#[cfg(not(target_arch = "wasm32"))]
fn screenshot_and_exit(path: String) -> impl FnMut(Commands, Frames<'_>, MessageWriter<AppExit>) {
    use bevy::render::view::screenshot::{Screenshot, save_to_disk};
    move |mut commands, mut frames, mut exit| {
        let (frames, since) = &mut *frames;
        *frames += 1;
        if *frames == 30 {
            *since = Some(Instant::now());
        }
        if *frames == 60 {
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(path.clone()));
        }
        if *frames == 90 {
            let ms = since.map_or(0.0, |since| since.elapsed().as_secs_f32() * 1000.0 / 60.0);
            info!("Frame time {ms:.2} ms over the 60 frames from the 30th after settling");
            exit.write(AppExit::Success);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::large_enum_variant, reason = "made once, at start")]
enum Start {
    Local(Dataset),
    Remote(remote::Remote),
}

#[cfg(not(target_arch = "wasm32"))]
fn load_native() -> Start {
    fn fail(msg: impl std::fmt::Display) -> ! {
        eprintln!("{msg}");
        std::process::exit(1)
    }
    let args = match config::parse_args(std::env::args().skip(1)) {
        Ok(Some(args)) => args,
        Ok(None) => {
            println!("{}", config::USAGE);
            std::process::exit(0)
        }
        Err(usage) => {
            eprintln!("{usage}");
            std::process::exit(2)
        }
    };
    if let Some(path) = args.config.as_ref().filter(|p| !p.exists()) {
        fail(format!("{}: no such config file", path.display()));
    }
    let config_path = args
        .config
        .or_else(config::default_path)
        .unwrap_or_else(|| fail("No config location found; pass --config"));
    let config = config::load(&config_path, &args.overrides)
        .unwrap_or_else(|e| fail(format!("{}: {e}", config_path.display())));

    // Printed, since logging only starts with the app.
    println!("Config: {}", config_path.display());
    if let Some(url) = args.server {
        let path = args.file.map(|path| path.to_string_lossy().into_owned());
        let mut remote = remote::Remote::new(url, config, path, args.hdu, String::new());
        remote.connect();
        return Start::Remote(remote);
    }
    let path = args.file.expect("a file without a server");
    // Bevy's own task pools, made before the app so loading doesn't leave it a differently sized one.
    bevy::app::TaskPoolOptions::default().create_default_pools();
    let start = Instant::now();
    let file =
        std::fs::File::open(&path).unwrap_or_else(|e| fail(format!("{}: {e}", path.display())));
    let cube = fits::read_stokes(&file, args.hdu, args.stokes.unwrap_or(0))
        .unwrap_or_else(|e| fail(format!("{}: {e}", path.display())));
    let name = path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    let info = cube.info.clone();
    let seconds = start.elapsed().as_secs_f32();
    let dataset = Dataset::new(name, info, DataSource::local(cube), config, seconds);
    println!("{}", dataset.summary());
    Start::Local(dataset)
}

/// Object-space box of a voxel region, as iDaVIE's CropToRegion and highlight bounds compute it.
fn region_bounds(region: Region, dims: [usize; 3]) -> (Vec3, Vec3) {
    let dims = Vec3::from_array(dims.map(|d| d as f32));
    let min = Vec3::from_array(region.min.map(|v| v as f32));
    let max = Vec3::from_array(region.max.map(|v| v as f32));
    (min / dims - 0.5, (max + 1.0) / dims - 0.5)
}

/// The most layers a volume texture may have: the device's limit, and iDaVIE's 2048 at most. A server is told it, so
/// that its levels are the ones `spawn_volume` makes.
pub fn max_depth(device: &RenderDevice) -> usize {
    device.limits().max_texture_array_layers.min(2048) as usize
}

/// The cube shown goes, for when nothing answers for it any more.
fn clear_dataset(
    commands: &mut Commands,
    old: &Query<Entity, With<DatasetEntity>>,
    upsamples: &mut Upsamples,
) {
    for entity in old {
        commands.entity(entity).despawn();
    }
    upsamples.0.clear();
    commands.remove_resource::<Dataset>();
    commands.remove_resource::<CubeState>();
    commands.remove_resource::<AxisAnnotation>();
    commands.remove_resource::<BoxShape>();
}

/// The whole cube starts as the coarsest level's one cubelet, which the source can cut at once, and `view::refine`
/// then upsamples that into the whole cube's level.
fn spawn_volume(
    mut commands: Commands,
    data: Res<Dataset>,
    old: Query<Entity, With<DatasetEntity>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<VolumeMaterial>>,
    mut line_materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut upsamples: ResMut<Upsamples>,
    mut orbit: Single<&mut Orbit>,
    device: Res<RenderDevice>,
) {
    for entity in &old {
        commands.entity(entity).despawn();
    }
    // The old cube's requests went with its source, but an upsample still to run would keep its texture forever.
    upsamples.0.clear();
    let dims = data.info.dims;
    let (budget, given) = (data.config.gpu_memory_limit_mb, data.levels.as_deref());
    let levels = Levels::new(dims, budget, max_depth(&device), given);
    data.source.send(Request::Levels {
        factors: levels.factors.clone(),
        chain: levels.chain,
    });
    let full = VolumeTexture::new(
        &levels,
        Region::whole(dims),
        levels.coarsest(),
        false,
        None,
        &mut images,
        &mut buffers,
    );

    let settings = Settings::new(&data);
    let shape = BoxShape::new(dims, settings.z_scale);
    let material = materials.add(VolumeMaterial {
        params: settings.params,
        data_cube: full.handle.clone(),
        color_map: images.add(volume::color_map_image()),
        grid: full.grid.clone(),
        modes: default(),
        mask: modes::NO_MASK,
        key: default(),
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(material.clone()),
        Transform::from_scale(shape.size()),
        // Until its texture has been filled.
        Visibility::Hidden,
        DatasetEntity,
    ));
    outline::spawn_outlines(&mut commands, &mut meshes, &mut line_materials);
    commands.insert_resource(settings);
    commands.insert_resource(modes::Extras::default());
    commands.insert_resource(Gesture::default());
    commands.insert_resource(Hover::default());
    commands.insert_resource(CubeState::new(material, levels, full, dims));
    commands.insert_resource(shape);
    **orbit = Orbit::default();
    orbit.frame(shape, Region::whole(dims), dims);
}

fn update_material(
    settings: Res<Settings>,
    state: Res<CubeState>,
    data: Res<Dataset>,
    shape: Res<BoxShape>,
    mut materials: ResMut<Assets<VolumeMaterial>>,
) {
    let Some(mut material) = materials.get_mut(&state.material) else {
        return;
    };
    let dims = data.info.dims;
    let mut params = settings.params;
    params.box_size = shape.size();
    (params.slice_min, params.slice_max) = region_bounds(state.visible, dims);
    if let Some(selection) = state.selection {
        (params.highlight_min, params.highlight_max) = region_bounds(selection, dims);
        params.highlight_saturate_factor = SELECTION_SATURATE_FACTOR;
    }
    let texture = state.shown();
    (params.texture_min, params.texture_max) = texture.bounds(dims);
    let to_vec = |v: [usize; 3]| Vec3::from_array(v.map(|v| v as f32));
    let visible = to_vec(state.visible.extent()) / to_vec(texture.factors);
    let (cells, offset) = (texture.cells(), texture.grid_offset());
    params.set_texture(texture.dims(), visible, cells, offset, texture.has_nan);
    material.params = params;
    material.data_cube = texture.handle.clone();
    material.grid = texture.grid.clone();
}

fn show_volume(
    state: Res<CubeState>,
    mut volume: Query<&mut Visibility, With<MeshMaterial3d<VolumeMaterial>>>,
) {
    let shown = if state.shown().drawable() {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut visibility in &mut volume {
        visibility.set_if_neq(shown);
    }
}
