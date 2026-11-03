mod axes;
mod camera;
mod config;
mod fits;
mod grips;
mod outline;
mod selection;
mod ui;
mod upload;
mod volume;
mod wcs;
#[cfg(target_arch = "wasm32")]
mod web;

use std::{sync::Arc, time::Duration};

use bevy::{
    diagnostic::FrameTimeDiagnosticsPlugin,
    platform::time::Instant,
    prelude::*,
    render::{
        RenderPlugin,
        settings::{WgpuFeatures, WgpuSettings},
    },
    window::RequestRedraw,
    winit::WinitSettings,
};
use bevy_egui::{EguiPlugin, EguiPrimaryContextPass};

use axes::AxisAnnotation;
use camera::Orbit;
use config::Config;
use fits::{Cube, Region};
use selection::{CropRequest, Gesture, Hover, PendingCrop};
use upload::{UploadPlugin, Uploads};
use volume::{VolumeMaterial, VolumeParams, VolumePlugin};

/// Bevy's local space is right-handed; iDaVIE's (Unity) object space mirrors z. Its own inverse, so it
/// converts either way.
const UNITY_FROM_LOCAL: Vec3 = Vec3::new(1.0, 1.0, -1.0);
/// iDaVIE's VolumeDataSetRenderer.SelectionSaturateFactor.
const SELECTION_SATURATE_FACTOR: f32 = 0.7;

/// A texture made from a region of the cube, and the downsampling it took.
struct VolumeTexture {
    region: Region,
    handle: Handle<Image>,
    dims: [usize; 3],
    factors: [usize; 3],
}

/// A VolumeTexture's bytes, built off the main thread and then uploaded a slab at a time.
struct TextureBytes {
    region: Region,
    bytes: Vec<u8>,
    dims: [usize; 3],
    factors: [usize; 3],
}

impl TextureBytes {
    /// The texture is empty until `uploads` has filled it.
    fn upload(self, images: &mut Assets<Image>, uploads: &mut Uploads) -> VolumeTexture {
        let handle = images.add(volume::empty_cube_image(self.dims));
        uploads.push(handle.id(), self.dims, self.bytes);
        VolumeTexture {
            region: self.region,
            handle,
            dims: self.dims,
            factors: self.factors,
        }
    }
}

#[derive(Resource)]
struct Dataset {
    name: String,
    // Shared with crops being built in the background.
    cube: Arc<Cube>,
    config: Config,
    wcs: Option<wcs::Wcs>,
    load_seconds: f32,
}

impl Dataset {
    fn new(name: String, cube: Cube, config: Config, start: Instant) -> Self {
        let wcs = wcs::Wcs::from_cards(&cube.cards, cube.axes);
        Self {
            name,
            cube: Arc::new(cube),
            config,
            wcs,
            load_seconds: start.elapsed().as_secs_f32(),
        }
    }

    /// What was loaded, for the terminal or the browser console.
    fn summary(&self) -> String {
        let s = self.cube.stats;
        let mut summary = format!(
            "Loaded {} (HDU {}): {} in {:.2} s\nRange {} .. {}, mean {:.4e}, std {:.4e}, {} non-finite voxels",
            self.name,
            self.cube.hdu,
            ui::format_dims(self.cube.dims),
            self.load_seconds,
            s.min,
            s.max,
            s.mean,
            s.std,
            s.non_finite
        );
        if let Some(planes) = self.cube.other_planes() {
            summary.push_str(&format!("\nWarning: showing plane {planes}"));
        }
        summary
    }
}

#[derive(Resource)]
struct CubeState {
    material: Handle<VolumeMaterial>,
    full: VolumeTexture,
    crop: Option<VolumeTexture>,
    selection: Option<Region>,
}

impl CubeState {
    fn texture(&self) -> &VolumeTexture {
        self.crop.as_ref().unwrap_or(&self.full)
    }

    /// The region on screen: the crop, or the whole cube.
    fn visible(&self) -> Region {
        self.texture().region
    }
}

#[derive(Resource, Clone, PartialEq)]
struct Settings {
    params: VolumeParams,
    wcs_axes: bool,
}

impl Settings {
    /// The config's rendering defaults, scaled to the cube's range.
    fn new(data: &Dataset) -> Self {
        let config = &data.config;
        Self {
            params: VolumeParams {
                scale_min: data.cube.stats.min,
                scale_max: data.cube.stats.max,
                jitter: config.jitter,
                threshold_min: config.threshold_min,
                threshold_max: config.threshold_max,
                max_steps: config.max_raymarching_steps,
                color_map_index: config.default_color_map.0,
                scale_type: config.default_scaling_type.0,
                ..default()
            },
            wcs_axes: true,
        }
    }
}

/// Everything spawned for the loaded file, despawned when another is loaded.
#[derive(Component)]
struct DatasetEntity;

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    let dataset = load_native();

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
    ))
    .insert_resource(ClearColor(Color::BLACK))
    .init_resource::<Hover>()
    .init_resource::<grips::Grips>()
    .init_resource::<Gesture>()
    .init_resource::<PendingCrop>()
    .add_message::<CropRequest>()
    .add_systems(Startup, camera::spawn_camera)
    .add_systems(
        Update,
        (
            selection::end_gesture,
            spawn_volume.run_if(resource_exists_and_changed::<Dataset>),
            // Grips go first so grabbing one takes the click away from orbiting and shift-drag selection,
            // and a crop retargets the orbit before the camera moves this frame.
            (
                grips::drag_grips,
                selection::select_region,
                selection::selection_keys,
                selection::apply_crop,
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
                    outline::update_outlines.run_if(resource_exists_and_changed::<CubeState>),
                    outline::shade_hidden_edges,
                )
                    .chain(),
                grips::draw_grips,
            )
                .run_if(resource_exists::<CubeState>),
        )
            .chain(),
    )
    .add_systems(Update, keep_redrawing)
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
    app.add_systems(Startup, web::keep_wake_proxy)
        .add_systems(PreUpdate, web::poll);
    #[cfg(not(target_arch = "wasm32"))]
    {
        app.insert_resource(dataset);
        if let Ok(path) = std::env::var("SCREENSHOT") {
            app.add_systems(Update, screenshot_and_exit(path));
        }
    }

    app.run();
}

/// Frames otherwise only come with input, so keep asking for them while something is still settling: a crop
/// being built, or a new texture still on its way to the screen.
fn keep_redrawing(
    pending: Res<PendingCrop>,
    uploads: Res<Uploads>,
    state: Option<Res<CubeState>>,
    mut settle_until: Local<Option<Instant>>,
    mut redraw: MessageWriter<RequestRedraw>,
) {
    if state.is_some_and(|state| state.is_changed()) {
        *settle_until = Some(Instant::now() + Duration::from_secs(1));
    }
    if pending.0.is_some()
        || !uploads.is_empty()
        || settle_until.is_some_and(|until| Instant::now() < until)
    {
        redraw.write(RequestRedraw);
    }
}

/// `SCREENSHOT=out.png` saves the window once rendering has settled, then exits.
#[cfg(not(target_arch = "wasm32"))]
fn screenshot_and_exit(path: String) -> impl FnMut(Commands, Local<u32>, MessageWriter<AppExit>) {
    use bevy::render::view::screenshot::{Screenshot, save_to_disk};
    move |mut commands, mut frames, mut exit| {
        *frames += 1;
        if *frames == 60 {
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(path.clone()));
        }
        if *frames == 90 {
            exit.write(AppExit::Success);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn load_native() -> Dataset {
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

    // Bevy's own task pools, made before the app so loading doesn't leave it a differently sized one.
    bevy::app::TaskPoolOptions::default().create_default_pools();
    let start = Instant::now();
    let file = std::fs::File::open(&args.file)
        .unwrap_or_else(|e| fail(format!("{}: {e}", args.file.display())));
    let cube = fits::read_fits(&file, args.hdu)
        .unwrap_or_else(|e| fail(format!("{}: {e}", args.file.display())));
    let name = args.file.file_name().map_or_else(
        || args.file.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    let dataset = Dataset::new(name, cube, config, start);
    // Printed, since logging only starts with the app.
    println!("Config: {}", config_path.display());
    println!("{}", dataset.summary());
    dataset
}

/// Texture for a region of the cube, downsampled to fit the configured GPU budget.
fn region_bytes(cube: &Cube, config: &Config, region: Region) -> TextureBytes {
    let start = Instant::now();
    let factors = fits::find_downsample_factors(config.gpu_memory_limit_mb, region.extent());
    let (dims, bytes) =
        fits::crop_and_downsample(cube, region, factors, config.max_mode_downsampling);
    let texture = TextureBytes {
        region,
        bytes,
        dims,
        factors,
    };
    info!(
        "Texture {dims:?} for {region:?}, downsampled by {factors:?}, in {:.2} s",
        start.elapsed().as_secs_f32()
    );
    texture
}

/// Object-space box of a voxel region, as iDaVIE's CropToRegion and highlight bounds compute it.
fn region_bounds(region: Region, dims: [usize; 3]) -> (Vec3, Vec3) {
    let dims = Vec3::from_array(dims.map(|d| d as f32));
    let min = Vec3::from_array(region.min.map(|v| v as f32));
    let max = Vec3::from_array(region.max.map(|v| v as f32));
    (min / dims - 0.5, (max + 1.0) / dims - 0.5)
}

fn spawn_volume(
    mut commands: Commands,
    data: Res<Dataset>,
    old: Query<Entity, With<DatasetEntity>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<VolumeMaterial>>,
    mut line_materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut uploads: ResMut<Uploads>,
    mut orbit: Single<&mut Orbit>,
) {
    for entity in &old {
        commands.entity(entity).despawn();
    }
    let full = region_bytes(&data.cube, &data.config, Region::whole(data.cube.dims))
        .upload(&mut images, &mut uploads);

    let settings = Settings::new(&data);
    let material = materials.add(VolumeMaterial {
        params: settings.params,
        data_cube: full.handle.clone(),
        color_map: images.add(volume::color_map_image()),
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(material.clone()),
        Transform::IDENTITY,
        // Until its texture has been filled.
        Visibility::Hidden,
        DatasetEntity,
    ));
    outline::spawn_outlines(&mut commands, &mut meshes, &mut line_materials);
    commands.insert_resource(settings);
    commands.insert_resource(Gesture::default());
    commands.insert_resource(PendingCrop::default());
    commands.insert_resource(CubeState {
        material,
        full,
        crop: None,
        selection: None,
    });
    **orbit = Orbit::default();
}

fn update_material(
    settings: Res<Settings>,
    state: Res<CubeState>,
    data: Res<Dataset>,
    mut materials: ResMut<Assets<VolumeMaterial>>,
) {
    let Some(mut material) = materials.get_mut(&state.material) else {
        return;
    };
    let mut params = settings.params;
    (params.slice_min, params.slice_max) = region_bounds(state.visible(), data.cube.dims);
    if let Some(selection) = state.selection {
        (params.highlight_min, params.highlight_max) = region_bounds(selection, data.cube.dims);
        params.highlight_saturate_factor = SELECTION_SATURATE_FACTOR;
    }
    material.params = params;
    material.data_cube = state.texture().handle.clone();
}

fn show_volume(
    state: Res<CubeState>,
    uploads: Res<Uploads>,
    mut volume: Query<&mut Visibility, With<MeshMaterial3d<VolumeMaterial>>>,
) {
    if !uploads.is_filling(state.full.handle.id()) {
        for mut visibility in &mut volume {
            visibility.set_if_neq(Visibility::Inherited);
        }
    }
}
