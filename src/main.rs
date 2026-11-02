mod axes;
mod camera;
mod config;
mod fits;
mod grips;
mod outline;
mod selection;
mod ui;
mod volume;
mod wcs;
#[cfg(target_arch = "wasm32")]
mod web;

use bevy::{diagnostic::FrameTimeDiagnosticsPlugin, platform::time::Instant, prelude::*};
use bevy_egui::{EguiPlugin, EguiPrimaryContextPass};

use camera::Orbit;
use config::Config;
use fits::Cube;
use outline::{CubeOutline, Outline, OutlineMaterials, SelectionOutline, box_lines};
use selection::{CropCommand, Gesture, Hover};
use volume::{VolumeMaterial, VolumeParams, VolumePlugin};

/// Bevy's local space is right-handed; iDaVIE's (Unity) object space mirrors z.
const UNITY_FROM_LOCAL: Vec3 = Vec3::new(1.0, 1.0, -1.0);
/// iDaVIE's VolumeDataSetRenderer.SelectionSaturateFactor.
const SELECTION_SATURATE_FACTOR: f32 = 0.7;

/// 0-based inclusive voxel corners.
type Region = ([usize; 3], [usize; 3]);

#[derive(Resource)]
struct Dataset {
    name: String,
    cube: Cube,
    config: Config,
    wcs: Option<wcs::Wcs>,
    load_seconds: f32,
}

impl Dataset {
    fn new(name: String, cube: Cube, config: Config, start: Instant) -> Self {
        let wcs = wcs::Wcs::from_cards(&cube.cards);
        Self {
            name,
            cube,
            config,
            wcs,
            load_seconds: start.elapsed().as_secs_f32(),
        }
    }

    fn full_region(&self) -> Region {
        ([0; 3], self.cube.dims.map(|d| d - 1))
    }
}

#[derive(Resource)]
struct CubeState {
    material: Handle<VolumeMaterial>,
    full_texture: Handle<Image>,
    full_dims: [usize; 3],
    full_factors: [usize; 3],
    texture: Handle<Image>,
    texture_dims: [usize; 3],
    factors: [usize; 3],
    crop: Option<Region>,
    selection: Option<Region>,
}

impl CubeState {
    /// The region on screen: the crop, or the whole cube.
    fn visible(&self, data: &Dataset) -> Region {
        self.crop.unwrap_or(data.full_region())
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
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "iDaVIE volume renderer (Bevy)".into(),
            canvas: Some("#bevy".into()),
            fit_canvas_to_parent: true,
            ..default()
        }),
        ..default()
    }))
    .add_plugins((
        EguiPlugin::default(),
        FrameTimeDiagnosticsPlugin::default(),
        VolumePlugin,
    ))
    .insert_resource(ClearColor(Color::BLACK))
    .init_resource::<Hover>()
    .init_resource::<grips::Grips>()
    .init_resource::<Gesture>()
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
    .add_systems(EguiPrimaryContextPass, (ui::ui, axes::draw_axes));

    #[cfg(target_arch = "wasm32")]
    app.add_systems(PreUpdate, web::poll);
    #[cfg(not(target_arch = "wasm32"))]
    app.insert_resource(dataset);
    #[cfg(not(target_arch = "wasm32"))]
    if let Ok(path) = std::env::var("SCREENSHOT") {
        app.add_systems(Update, screenshot_and_exit(path));
    }

    app.run();
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

    let start = Instant::now();
    let file = std::fs::File::open(&args.file)
        .unwrap_or_else(|e| fail(format!("{}: {e}", args.file.display())));
    let cube = fits::read_fits(std::io::BufReader::new(file), args.hdu)
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
    dataset
}

/// Texture for a region of the cube, downsampled to fit the configured GPU budget.
fn region_texture(data: &Dataset, (min, max): Region) -> (Image, [usize; 3], [usize; 3]) {
    let extent: [usize; 3] = std::array::from_fn(|i| max[i] - min[i] + 1);
    let factors = fits::find_downsample_factors(data.config.gpu_memory_limit_mb, extent);
    if factors == [1; 3] && extent == data.cube.dims {
        return (volume::cube_image(extent, &data.cube.data), extent, factors);
    }
    let (dims, values) = fits::crop_and_downsample(
        &data.cube,
        min,
        max,
        factors,
        data.config.max_mode_downsampling,
    );
    (volume::cube_image(dims, &values), dims, factors)
}

/// Object-space box of a voxel region, as iDaVIE's CropToRegion and highlight bounds compute it.
fn region_bounds((min, max): Region, dims: [usize; 3]) -> (Vec3, Vec3) {
    let dims = Vec3::from_array(dims.map(|d| d as f32));
    let min = Vec3::from_array(min.map(|v| v as f32));
    let max = Vec3::from_array(max.map(|v| v as f32));
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
    mut orbits: Query<&mut Orbit>,
) {
    for entity in &old {
        commands.entity(entity).despawn();
    }
    let start = Instant::now();
    let (image, dims, factors) = region_texture(&data, data.full_region());
    info!(
        "Volume texture {dims:?} (downsampled by {factors:?}) in {:.2} s",
        start.elapsed().as_secs_f32()
    );
    let texture = images.add(image);

    let config = &data.config;
    let settings = Settings {
        params: VolumeParams {
            scale_min: data.cube.stats.min,
            scale_max: data.cube.stats.max,
            jitter: config.jitter,
            max_steps: config.max_raymarching_steps as f32,
            color_map_index: config.color_map_index() as f32,
            scale_type: config.scaling_type_index(),
            ..default()
        },
        wcs_axes: true,
    };
    let material = materials.add(VolumeMaterial {
        params: settings.params,
        data_cube: texture.clone(),
        color_map: images.add(volume::color_map_image()),
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(material.clone()),
        Transform::IDENTITY,
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
        Outline::default(),
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
    commands.insert_resource(CubeState {
        material,
        full_texture: texture.clone(),
        full_dims: dims,
        full_factors: factors,
        texture,
        texture_dims: dims,
        factors,
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
    material.data_cube = state.texture.clone();
}
