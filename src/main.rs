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

use camera::Orbit;
use config::Config;
use fits::Cube;
use outline::{CubeOutline, Outline, OutlineMaterials, SelectionOutline, box_lines};
use selection::{CropCommand, Gesture, Hover, PendingCrop};
use upload::{UploadPlugin, Uploads};
use volume::{VolumeMaterial, VolumeParams, VolumePlugin};

/// Bevy's local space is right-handed; iDaVIE's (Unity) object space mirrors z.
const UNITY_FROM_LOCAL: Vec3 = Vec3::new(1.0, 1.0, -1.0);
/// iDaVIE's VolumeDataSetRenderer.SelectionSaturateFactor.
const SELECTION_SATURATE_FACTOR: f32 = 0.7;

/// 0-based inclusive voxel corners.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Region {
    min: [usize; 3],
    max: [usize; 3],
}

impl Region {
    fn whole(dims: [usize; 3]) -> Self {
        Self {
            min: [0; 3],
            max: dims.map(|d| d - 1),
        }
    }

    fn extent(&self) -> [usize; 3] {
        std::array::from_fn(|i| self.max[i] - self.min[i] + 1)
    }
}

/// A texture made from a region of the cube, and the downsampling it took.
#[derive(Clone)]
struct VolumeTexture {
    handle: Handle<Image>,
    dims: [usize; 3],
    factors: [usize; 3],
}

/// A VolumeTexture's bytes, built off the main thread and then uploaded a slab at a time.
struct TextureBytes {
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
        if let Some(planes) = cube.other_planes() {
            warn!("{name}: showing plane {planes}");
        }
        Self {
            name,
            cube: Arc::new(cube),
            config,
            wcs,
            load_seconds: start.elapsed().as_secs_f32(),
        }
    }

    fn full_region(&self) -> Region {
        Region::whole(self.cube.dims)
    }
}

#[derive(Resource)]
struct CubeState {
    material: Handle<VolumeMaterial>,
    full: VolumeTexture,
    crop: Option<(Region, VolumeTexture)>,
    selection: Option<Region>,
}

impl CubeState {
    /// The region on screen: the crop, or the whole cube.
    fn visible(&self, data: &Dataset) -> Region {
        self.crop
            .as_ref()
            .map_or(data.full_region(), |(region, _)| *region)
    }

    fn texture(&self) -> &VolumeTexture {
        self.crop
            .as_ref()
            .map_or(&self.full, |(_, texture)| texture)
    }
}

#[derive(Resource, Clone, PartialEq)]
struct Settings {
    params: VolumeParams,
    wcs_axes: bool,
}

#[derive(Component)]
struct VolumeCube;

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
    .add_message::<CropCommand>()
    .add_systems(Startup, camera::spawn_camera)
    .add_systems(
        Update,
        (
            spawn_volume.run_if(resource_exists_and_changed::<Dataset>),
            // Grips go first so grabbing one takes the click away from orbiting and shift-drag selection,
            // and a crop retargets the orbit before the camera moves this frame.
            (
                grips::drag_grips,
                selection::select_region,
                selection::apply_crop,
            )
                .chain()
                .run_if(resource_exists::<CubeState>),
            camera::orbit_camera,
            (
                update_material,
                show_volume,
                axes::update_axes,
                outline::update_outlines,
                outline::shade_hidden_edges,
                grips::draw_grips,
            )
                .chain()
                .run_if(resource_exists::<CubeState>),
        )
            .chain(),
    )
    .add_systems(Update, keep_redrawing)
    .add_systems(EguiPrimaryContextPass, (ui::ui, axes::draw_axes));

    // Redraw only on input, so an idle view doesn't keep the GPU ray-marching. SCREENSHOT counts frames, so
    // it keeps Bevy's continuous default.
    #[cfg(not(target_arch = "wasm32"))]
    let continuous = std::env::var("SCREENSHOT").is_ok();
    #[cfg(target_arch = "wasm32")]
    let continuous = false;
    if !continuous {
        app.insert_resource(WinitSettings::desktop_app());
    }

    #[cfg(target_arch = "wasm32")]
    app.add_systems(Startup, web::keep_wake_proxy)
        .add_systems(PreUpdate, web::poll);
    #[cfg(not(target_arch = "wasm32"))]
    app.insert_resource(dataset);
    #[cfg(not(target_arch = "wasm32"))]
    if let Ok(path) = std::env::var("SCREENSHOT") {
        app.add_systems(Update, screenshot_and_exit(path));
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
        .clone()
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

    let [x, y, z] = dataset.cube.dims;
    let s = dataset.cube.stats;
    println!("Config: {}", config_path.display());
    println!(
        "Loaded {} (HDU {}): {x} x {y} x {z} in {:.2} s",
        dataset.name, dataset.cube.hdu, dataset.load_seconds
    );
    println!(
        "Range {} .. {}, mean {:.4e}, std {:.4e}, {} non-finite voxels",
        s.min, s.max, s.mean, s.std, s.non_finite
    );
    // Logging only starts with the app, so Dataset::new's warning doesn't reach the terminal.
    if let Some(planes) = dataset.cube.other_planes() {
        println!("Warning: showing plane {planes}");
    }
    dataset
}

/// Texture for a region of the cube, downsampled to fit the configured GPU budget.
fn region_bytes(cube: &Cube, config: &Config, region: Region) -> TextureBytes {
    let start = Instant::now();
    let factors = fits::find_downsample_factors(config.gpu_memory_limit_mb, region.extent());
    let (dims, bytes) = fits::crop_and_downsample(
        cube,
        region.min,
        region.max,
        factors,
        config.max_mode_downsampling,
    );
    let texture = TextureBytes {
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
    old: Query<Entity, With<VolumeCube>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<VolumeMaterial>>,
    mut line_materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut uploads: ResMut<Uploads>,
    mut orbits: Query<&mut Orbit>,
) {
    for entity in &old {
        commands.entity(entity).despawn();
    }
    let full = region_bytes(&data.cube, &data.config, data.full_region())
        .upload(&mut images, &mut uploads);

    let config = &data.config;
    let settings = Settings {
        params: VolumeParams {
            scale_min: data.cube.stats.min,
            scale_max: data.cube.stats.max,
            jitter: config.jitter,
            max_steps: config.max_raymarching_steps,
            color_map_index: config.default_color_map.0,
            scale_type: config.default_scaling_type.0,
            ..default()
        },
        wcs_axes: true,
    };
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
        VolumeCube,
    ));
    // Opaque line meshes rather than gizmos, so they reach the depth prepass and the volume covers the back edges.
    let line_colour = |c: Color| StandardMaterial {
        base_color: c,
        unlit: true,
        ..default()
    };
    let outline = OutlineMaterials {
        white: line_materials.add(line_colour(Color::WHITE)),
        yellow: line_materials.add(line_colour(Color::srgb(1.0, 1.0, 0.0))),
    };
    commands.spawn((
        Mesh3d(meshes.add(box_lines())),
        MeshMaterial3d(outline.white.clone()),
        Outline::with_axes(),
        CubeOutline,
        VolumeCube,
    ));
    commands.spawn((
        Mesh3d(meshes.add(box_lines())),
        MeshMaterial3d(outline.white.clone()),
        Visibility::Hidden,
        Outline::default(),
        SelectionOutline,
        VolumeCube,
    ));
    commands.insert_resource(outline);
    commands.insert_resource(settings);
    commands.insert_resource(Gesture::default());
    commands.insert_resource(PendingCrop::default());
    commands.insert_resource(CubeState {
        material,
        full,
        crop: None,
        selection: None,
    });
    for mut orbit in &mut orbits {
        *orbit = Orbit::default();
    }
}

fn update_material(
    settings: Res<Settings>,
    state: Res<CubeState>,
    data: Res<Dataset>,
    mut materials: ResMut<Assets<VolumeMaterial>>,
) {
    if !settings.is_changed() && !state.is_changed() {
        return;
    }
    let Some(mut material) = materials.get_mut(&state.material) else {
        return;
    };
    let mut params = settings.params;
    (params.slice_min, params.slice_max) = region_bounds(state.visible(&data), data.cube.dims);
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
