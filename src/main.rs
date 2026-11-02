mod config;
mod fits;
mod volume;
mod wcs;

use bevy::{
    asset::RenderAssetUsages,
    core_pipeline::prepass::DepthPrepass,
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit},
    mesh::PrimitiveTopology,
    platform::time::Instant,
    prelude::*,
    render::view::{DebandDither, Msaa, Tonemapping},
};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui, input::EguiWantsInput};

use config::{COLOR_MAPS, Config, SCALING_TYPES};
use fits::Cube;
use volume::{VolumeMaterial, VolumeParams, VolumePlugin};

/// Bevy's local space is right-handed; iDaVIE's (Unity) object space mirrors z.
const UNITY_FROM_LOCAL: Vec3 = Vec3::new(1.0, 1.0, -1.0);
/// iDaVIE's VolumeDataSetRenderer.SelectionSaturateFactor.
const SELECTION_SATURATE_FACTOR: f32 = 0.7;
const DEFAULT_DISTANCE: f32 = 2.0;

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

#[derive(Resource, Clone, PartialEq)]
struct Settings {
    params: VolumeParams,
    wcs_axes: bool,
}

/// WCS ticks for the 12 edges of the visible box (see `edge_ends`), recomputed when it changes.
#[derive(Resource)]
struct AxisAnnotation {
    region: Region,
    edges: Vec<Vec<wcs::Tick>>,
    titles: [Option<String>; 3],
}

#[derive(Resource, Default)]
struct Hover(Option<[usize; 3]>);

#[derive(Resource)]
struct LoadError(String);

#[derive(Message)]
enum CropCommand {
    ToSelection,
    Reset,
}

#[derive(Component)]
struct Orbit {
    target: Vec3,
    yaw: f32,
    pitch: f32,
    distance: f32,
}

#[derive(Component)]
struct VolumeCube;

#[derive(Component)]
struct SelectionOutline;

#[derive(Component)]
struct CubeOutline;

/// A box outline drawn as line meshes; `hidden` caches the last hidden-edge mask written to its colours.
#[derive(Component, Default)]
struct Outline {
    hidden: Option<u16>,
}

#[derive(Resource)]
struct OutlineMaterials {
    white: Handle<StandardMaterial>,
    yellow: Handle<StandardMaterial>,
}

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
    .init_resource::<Grips>()
    .add_message::<CropCommand>()
    .add_systems(Startup, spawn_camera)
    .add_systems(
        Update,
        (
            spawn_volume.run_if(resource_exists_and_changed::<Dataset>),
            // Grips go first so grabbing one takes the click away from orbiting and shift-drag selection.
            drag_grips.run_if(resource_exists::<CubeState>),
            orbit_camera,
            (
                select_region,
                apply_crop,
                update_material,
                update_axes,
                update_outlines,
                shade_hidden_edges,
                draw_grips,
            )
                .chain()
                .run_if(resource_exists::<CubeState>),
        )
            .chain(),
    )
    .add_systems(EguiPrimaryContextPass, (ui, draw_axes));

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
    let args = config::parse_args(std::env::args().skip(1)).unwrap_or_else(|usage| {
        eprintln!("{usage}");
        std::process::exit(2)
    });
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
        "Range {} .. {}, mean {:.4e}, std {:.4e}, {} NaN voxels",
        s.min, s.max, s.mean, s.std, s.nan_count
    );
    dataset
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        // Unity's camera field of view is vertical, like Bevy's.
        Projection::Perspective(PerspectiveProjection {
            fov: 60f32.to_radians(),
            near: 0.05,
            ..default()
        }),
        // Unity's linear colour space pipeline: no tone curve, sRGB encode on output.
        Tonemapping::None,
        DebandDither::Disabled,
        Msaa::Off,
        // The volume shader clamps its rays against this, as iDaVIE's does against _CameraDepthTexture.
        DepthPrepass,
        Orbit {
            target: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            distance: DEFAULT_DISTANCE,
        },
    ));
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

/// The 12 edges of a unit cube centred on the origin.
fn box_lines() -> Mesh {
    let corner = |i: u32| Vec3::new((i & 1) as f32, (i >> 1 & 1) as f32, (i >> 2 & 1) as f32) - 0.5;
    let positions: Vec<Vec3> = box_edges()
        .flat_map(|(a, b)| [corner(a), corner(a | b)])
        .collect();
    // Kept in the main world too, so shade_hidden_edges can rewrite the colours.
    Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, vec![[1.0f32; 4]; 24])
}

/// The unit cube's edges as (corner, axis bit): corner bits 1, 2 and 4 put it on the high side of x, y
/// and z, and the edge runs from there along the axis bit.
fn box_edges() -> impl Iterator<Item = (u32, u32)> {
    (0..8u32).flat_map(|a| {
        [1, 2, 4]
            .into_iter()
            .filter(move |b| a & b == 0)
            .map(move |b| (a, b))
    })
}

/// Bit i is set when edge i of `box_edges` is hidden from `eye` (in the unit cube's own coordinates):
/// neither face it borders faces the eye.
fn hidden_edges(eye: Vec3) -> u16 {
    box_edges()
        .enumerate()
        .fold(0, |mask, (i, (corner, along))| {
            let seen = (0..3).filter(|axis| 1 << axis != along).any(|axis| {
                if corner & 1 << axis != 0 {
                    eye[axis] > 0.5
                } else {
                    eye[axis] < -0.5
                }
            });
            if seen { mask } else { mask | 1 << i }
        })
}

/// Draws an outline's hidden edges dark grey so the box reads as solid on a flat screen.
fn shade_hidden_edges(
    cameras: Query<&Transform, With<Orbit>>,
    mut outlines: Query<(&Transform, &Mesh3d, &mut Outline)>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let Ok(camera) = cameras.single() else {
        return;
    };
    for (transform, mesh, mut outline) in &mut outlines {
        let eye = transform
            .compute_affine()
            .inverse()
            .transform_point3(camera.translation);
        let hidden = hidden_edges(eye);
        if outline.hidden == Some(hidden) {
            continue;
        }
        if let Some(mut mesh) = meshes.get_mut(&mesh.0) {
            // Linear 0.1 is about 35% grey once encoded to sRGB.
            let colours: Vec<[f32; 4]> = (0..24)
                .map(|v| {
                    if hidden >> (v / 2) & 1 == 1 {
                        [0.1, 0.1, 0.1, 1.0]
                    } else {
                        [1.0; 4]
                    }
                })
                .collect();
            mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
            outline.hidden = Some(hidden);
        }
    }
}

#[allow(clippy::too_many_arguments)]
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
        *orbit = Orbit {
            target: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            distance: DEFAULT_DISTANCE,
        };
    }
}

fn orbit_camera(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    egui: Res<EguiWantsInput>,
    grips: Res<Grips>,
    mut cameras: Query<(&mut Orbit, &mut Transform)>,
) {
    let Ok((mut orbit, mut transform)) = cameras.single_mut() else {
        return;
    };
    let shift = keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
    if !egui.wants_pointer_input() {
        if buttons.pressed(MouseButton::Left) && !shift && grips.drag.is_none() {
            orbit.yaw -= motion.delta.x * 0.005;
            orbit.pitch = (orbit.pitch - motion.delta.y * 0.005).clamp(-1.55, 1.55);
        }
        let lines = match scroll.unit {
            MouseScrollUnit::Line => scroll.delta.y,
            // A browser wheel notch is about 100 px.
            MouseScrollUnit::Pixel => scroll.delta.y / 100.0,
        };
        orbit.distance = (orbit.distance * (-0.1 * lines).exp()).clamp(0.01, 20.0);
    }
    let rotation = Quat::from_euler(EulerRot::YXZ, orbit.yaw, orbit.pitch, 0.0);
    if !egui.wants_pointer_input() && buttons.any_pressed([MouseButton::Right, MouseButton::Middle])
    {
        // About one pixel at the target's distance with a 60 degree field of view.
        let scale = orbit.distance * 0.0012;
        orbit.target += rotation * Vec3::new(-motion.delta.x, motion.delta.y, 0.0) * scale;
    }
    let offset = rotation * Vec3::Z * orbit.distance;
    *transform =
        Transform::from_translation(orbit.target + offset).looking_at(orbit.target, Vec3::Y);
}

/// An in-progress shift-drag: the axis facing the camera, the voxel it started on, and the depth range it keeps.
#[derive(Clone, Copy)]
struct Drag {
    axis: usize,
    start: [usize; 3],
    depth: [usize; 2],
}

/// Shift-drag selects a box: the two axes facing the camera follow the pointer on a plane through the
/// middle of the visible region, the depth axis keeps the previous selection's range (or the full range).
#[allow(clippy::too_many_arguments)]
fn select_region(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    egui: Res<EguiWantsInput>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    data: Res<Dataset>,
    mut state: ResMut<CubeState>,
    mut hover: ResMut<Hover>,
    mut crop: MessageWriter<CropCommand>,
    grips: Res<Grips>,
    mut drag: Local<Option<Drag>>,
) {
    let Ok((camera, camera_transform)) = cameras.single() else {
        return;
    };
    let dims = data.cube.dims;
    let visible = state.crop.unwrap_or(data.full_region());
    let axis = (camera_transform.forward().as_vec3() * UNITY_FROM_LOCAL)
        .abs()
        .max_position();
    let axis = drag.map_or(axis, |d| d.axis);
    let (lo, hi) = region_bounds(visible, dims);
    let plane = (lo[axis] + hi[axis]) / 2.0;

    let pointer = windows
        .single()
        .ok()
        .and_then(Window::cursor_position)
        .and_then(|p| camera.viewport_to_world(camera_transform, p).ok())
        .and_then(|ray| {
            let origin = ray.origin * UNITY_FROM_LOCAL;
            let direction = *ray.direction * UNITY_FROM_LOCAL;
            let t = (plane - origin[axis]) / direction[axis];
            (t.is_finite() && t > 0.0).then(|| {
                let p = origin + direction * t;
                std::array::from_fn::<i64, 3, _>(|i| ((p[i] + 0.5) * dims[i] as f32).floor() as i64)
            })
        });
    let clamp_to = |v: [i64; 3], (min, max): Region| -> [usize; 3] {
        std::array::from_fn(|i| v[i].clamp(min[i] as i64, max[i] as i64) as usize)
    };
    let inside =
        |v: &[i64; 3]| (0..3).all(|i| v[i] >= visible.0[i] as i64 && v[i] <= visible.1[i] as i64);
    hover.0 = pointer
        .filter(|v| inside(v) && !egui.wants_pointer_input())
        .map(|v| clamp_to(v, visible));

    let shift = keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
    if buttons.just_pressed(MouseButton::Left)
        && shift
        && !egui.wants_pointer_input()
        && grips.drag.is_none()
        && let Some(v) = pointer
    {
        let depth = state.selection.unwrap_or(visible);
        *drag = Some(Drag {
            axis,
            start: clamp_to(v, visible),
            depth: [depth.0[axis], depth.1[axis]],
        });
    }
    if !buttons.pressed(MouseButton::Left) {
        *drag = None;
    }
    if let (Some(Drag { axis, start, depth }), Some(v)) = (*drag, pointer) {
        let end = clamp_to(v, visible);
        let mut min: [usize; 3] = std::array::from_fn(|i| start[i].min(end[i]));
        let mut max: [usize; 3] = std::array::from_fn(|i| start[i].max(end[i]));
        (min[axis], max[axis]) = (depth[0], depth[1]);
        if state.selection != Some((min, max)) {
            state.selection = Some((min, max));
        }
    }

    if !egui.wants_keyboard_input() {
        if keys.just_pressed(KeyCode::Escape) && state.selection.is_some() {
            state.selection = None;
        }
        if keys.any_just_pressed([KeyCode::KeyC, KeyCode::Enter]) {
            crop.write(CropCommand::ToSelection);
        }
        if keys.just_pressed(KeyCode::KeyR) {
            crop.write(CropCommand::Reset);
        }
    }
}

fn apply_crop(
    mut commands: MessageReader<CropCommand>,
    data: Res<Dataset>,
    mut state: ResMut<CubeState>,
    mut images: ResMut<Assets<Image>>,
    mut orbits: Query<&mut Orbit>,
) {
    for command in commands.read() {
        let region = match command {
            CropCommand::ToSelection => match state.selection.take() {
                Some(region) => Some(region),
                None => continue,
            },
            CropCommand::Reset => None,
        };
        if state.texture != state.full_texture {
            images.remove(&state.texture);
        }
        if let Some(region) = region {
            let start = Instant::now();
            let (image, dims, factors) = region_texture(&data, region);
            info!(
                "Cropped to {region:?}: texture {dims:?}, downsampled by {factors:?}, {:.2} s",
                start.elapsed().as_secs_f32()
            );
            (state.texture, state.texture_dims, state.factors) = (images.add(image), dims, factors);
        } else {
            (state.texture, state.texture_dims, state.factors) = (
                state.full_texture.clone(),
                state.full_dims,
                state.full_factors,
            );
        }
        state.crop = region;

        let (lo, hi) = region_bounds(region.unwrap_or(data.full_region()), data.cube.dims);
        for mut orbit in &mut orbits {
            orbit.target = (lo + hi) / 2.0 * UNITY_FROM_LOCAL;
            orbit.distance = DEFAULT_DISTANCE * (hi - lo).max_element();
        }
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
    (params.slice_min, params.slice_max) =
        region_bounds(state.crop.unwrap_or(data.full_region()), data.cube.dims);
    if let Some(selection) = state.selection {
        (params.highlight_min, params.highlight_max) = region_bounds(selection, data.cube.dims);
        params.highlight_saturate_factor = SELECTION_SATURATE_FACTOR;
    }
    material.params = params;
    material.data_cube = state.texture.clone();
}

/// Ends of box edge `e` in unit-box coordinates: it runs along axis `e / 4`, and the low bits of `e`
/// put it on the low or high side of the other two axes.
fn edge_ends(e: usize) -> ([f32; 3], [f32; 3]) {
    let axis = e / 4;
    let mut start = [0.0; 3];
    start[(axis + 1) % 3] = (e & 1) as f32;
    start[(axis + 2) % 3] = ((e >> 1) & 1) as f32;
    let mut end = start;
    end[axis] = 1.0;
    (start, end)
}

fn update_axes(
    mut commands: Commands,
    data: Res<Dataset>,
    state: Res<CubeState>,
    axes: Option<Res<AxisAnnotation>>,
) {
    let region @ (min, max) = state.crop.unwrap_or(data.full_region());
    if !data.is_changed() && axes.is_some_and(|a| a.region == region) {
        return;
    }
    let Some(wcs) = &data.wcs else {
        commands.remove_resource::<AxisAnnotation>();
        return;
    };
    // GRID coordinates put voxel i's centre at i + 1, so the box spans min + 0.5 to max + 1.5.
    let grid = |u: [f32; 3]| -> [f64; 3] {
        std::array::from_fn(|i| min[i] as f64 + 0.5 + u[i] as f64 * (max[i] - min[i] + 1) as f64)
    };
    let edges = (0..12)
        .map(|e| {
            let (start, end) = edge_ends(e);
            wcs.edge_ticks(e / 4 + 1, grid(start), grid(end))
        })
        .collect();
    let titles = std::array::from_fn(|i| wcs.axis_title(i + 1));
    commands.insert_resource(AxisAnnotation {
        region,
        edges,
        titles,
    });
}

/// Ticks on the silhouette edge of each axis (so the labels sit outside the volume), labelled with egui.
fn draw_axes(
    mut contexts: EguiContexts,
    mut gizmos: Gizmos,
    data: Option<Res<Dataset>>,
    settings: Option<Res<Settings>>,
    axes: Option<Res<AxisAnnotation>>,
    cameras: Query<(&Camera, &Transform), With<Orbit>>,
) -> Result {
    let (Some(data), Some(settings), Some(axes)) = (data, settings, axes) else {
        return Ok(());
    };
    let Ok((camera, transform)) = cameras.single() else {
        return Ok(());
    };
    if !settings.wcs_axes {
        return Ok(());
    }
    // The local transform is already this frame's orbit; GlobalTransform may not be propagated yet.
    let view = GlobalTransform::from(*transform);
    let (lo, hi) = region_bounds(axes.region, data.cube.dims);
    let world = |u: [f32; 3]| (lo + (hi - lo) * Vec3::from_array(u)) * UNITY_FROM_LOCAL;
    let screen = |p: Vec3| camera.world_to_viewport(&view, p).ok();
    let centre = world([0.5; 3]);
    let Some(centre_on_screen) = screen(centre) else {
        return Ok(());
    };
    let tick_length = 0.03 * (hi - lo).max_element();
    let painter = contexts
        .ctx_mut()?
        .layer_painter(egui::LayerId::background());
    // Labels that would overlap one already drawn are skipped, which thins out foreshortened edges.
    let mut placed: Vec<egui::Rect> = Vec::new();
    let mut text = |at: Vec3, from: Vec3, label: &str, size: f32| {
        let (Some(at), Some(from)) = (screen(at), screen(from)) else {
            return;
        };
        let away = at - from;
        let align = match (away.x.abs() > away.y.abs(), away.x > 0.0, away.y > 0.0) {
            (true, true, _) => egui::Align2::LEFT_CENTER,
            (true, false, _) => egui::Align2::RIGHT_CENTER,
            (false, _, true) => egui::Align2::CENTER_TOP,
            (false, _, false) => egui::Align2::CENTER_BOTTOM,
        };
        let color = egui::Color32::from_gray(220);
        let galley =
            painter.layout_no_wrap(label.to_string(), egui::FontId::proportional(size), color);
        let rect = align.anchor_size(egui::pos2(at.x, at.y), galley.size());
        if placed.iter().any(|r| r.intersects(rect.expand(2.0))) {
            return;
        }
        placed.push(rect);
        painter.galley(rect.min, galley, color);
    };

    let mut titles = Vec::new();
    for axis in 0..3 {
        let edge_on_screen = |e: usize| {
            let (start, end) = edge_ends(e);
            let mid = std::array::from_fn(|i| (start[i] + end[i]) / 2.0);
            screen(world(mid)).map_or(f32::MIN, |p| p.distance(centre_on_screen) + 0.001 * p.y)
        };
        let e = (axis * 4..axis * 4 + 4)
            .max_by(|a, b| edge_on_screen(*a).total_cmp(&edge_on_screen(*b)))
            .unwrap();
        let (start, end) = edge_ends(e);
        let (start, end) = (world(start), world(end));
        let mid = (start + end) / 2.0;
        let along = (end - start).normalize();
        let outward = {
            let d = mid - centre;
            (d - along * d.dot(along)).normalize_or_zero()
        };
        for tick in &axes.edges[e] {
            let p = start.lerp(end, tick.at);
            gizmos.line(p, p + outward * tick_length, Color::WHITE);
            text(p + outward * tick_length * 1.5, p, &tick.label, 12.0);
        }
        if let Some(title) = &axes.titles[axis] {
            titles.push((mid + outward * tick_length * 5.0, mid, title));
        }
    }
    for (at, from, title) in titles {
        text(at, from, title, 14.0);
    }
    Ok(())
}

fn box_transform(region: Region, dims: [usize; 3]) -> Transform {
    let (lo, hi) = region_bounds(region, dims);
    Transform::from_translation((lo + hi) / 2.0 * UNITY_FROM_LOCAL).with_scale(hi - lo)
}

/// The cube outline follows the visible (cropped) region; the selection outline follows the selection.
fn update_outlines(
    data: Res<Dataset>,
    state: Res<CubeState>,
    outline_materials: Res<OutlineMaterials>,
    mut cube: Query<&mut Transform, (With<CubeOutline>, Without<SelectionOutline>)>,
    mut outline: Query<
        (
            &mut Transform,
            &mut Visibility,
            &mut MeshMaterial3d<StandardMaterial>,
        ),
        With<SelectionOutline>,
    >,
) {
    if !state.is_changed() {
        return;
    }
    for mut transform in &mut cube {
        *transform = box_transform(state.crop.unwrap_or(data.full_region()), data.cube.dims);
    }
    let Ok((mut transform, mut visibility, mut material)) = outline.single_mut() else {
        return;
    };
    let Some(selection) = state.selection else {
        *visibility = Visibility::Hidden;
        return;
    };
    *transform = box_transform(selection, data.cube.dims);
    // iDaVIE colours the region white when cropping to it would load at full resolution, yellow otherwise.
    let full_res = voxel_count(selection) * 4 <= data.config.gpu_memory_limit_mb * 1_000_000;
    material.0 = if full_res {
        outline_materials.white.clone()
    } else {
        outline_materials.yellow.clone()
    };
    *visibility = Visibility::Visible;
}

/// A grip on the selection box, as -1 (low face), 0 (middle) or +1 (high face) per axis: one non-zero
/// component is a face grip and three a corner grip.
fn grip_ids() -> impl Iterator<Item = IVec3> {
    (-1..=1)
        .flat_map(|x| (-1..=1).flat_map(move |y| (-1..=1).map(move |z| IVec3::new(x, y, z))))
        .filter(|g| matches!(g.abs().element_sum(), 1 | 3))
}

fn is_face(grip: IVec3) -> bool {
    grip.abs().element_sum() == 1
}

/// Grips and their face arrows are sized as fractions of the distance to the camera, so they keep a
/// constant size on screen.
const GRIP_SIZE: f32 = 0.01;
const ARROW_LENGTH: f32 = 0.06;

/// The tip of a face grip's arrow, which starts at the face centre and points outward.
fn arrow_tip(grip: IVec3, position: Vec3, eye: Vec3) -> Vec3 {
    position + grip.as_vec3() * (eye - position).length() * ARROW_LENGTH
}

/// A grip's position in iDaVIE (Unity) object space, on the box `lo`..`hi`.
fn grip_position(grip: IVec3, (lo, hi): (Vec3, Vec3)) -> Vec3 {
    let middle = (lo + hi) / 2.0;
    Vec3::select(
        grip.cmplt(IVec3::ZERO),
        lo,
        Vec3::select(grip.cmpgt(IVec3::ZERO), hi, middle),
    )
}

/// Where the pointer ray (object space) meets a grip's constraint: the axis line through a face grip,
/// or the plane facing the camera for a corner grip.
fn grip_hit(grip: IVec3, anchor: Vec3, origin: Vec3, dir: Vec3, forward: Vec3) -> Option<Vec3> {
    if is_face(grip) {
        // Closest point on the line anchor + s * axis to the ray.
        let axis = grip.abs().as_vec3();
        let w = anchor - origin;
        let (b, c, d, e) = (axis.dot(dir), dir.dot(dir), axis.dot(w), dir.dot(w));
        let denom = c - b * b;
        return (denom.abs() > 1e-6).then(|| anchor + axis * ((b * e - c * d) / denom));
    }
    let denom = dir.dot(forward);
    (denom.abs() > 1e-6).then(|| origin + dir * ((anchor - origin).dot(forward) / denom))
}

#[derive(Resource, Default)]
struct Grips {
    hovered: Option<IVec3>,
    drag: Option<GripDrag>,
}

#[derive(Clone, Copy)]
struct GripDrag {
    grip: IVec3,
    start: Region,
    start_hit: Vec3,
}

/// Autodesk-style resizing: drag a face, edge or corner grip of the selection box. Faces snap to voxels
/// and stay inside the visible region, and a face can't pass the opposite one.
fn drag_grips(
    buttons: Res<ButtonInput<MouseButton>>,
    egui: Res<EguiWantsInput>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    data: Res<Dataset>,
    mut state: ResMut<CubeState>,
    mut grips: ResMut<Grips>,
) {
    let (Some(selection), Ok((camera, camera_transform))) = (state.selection, cameras.single())
    else {
        *grips = Grips::default();
        return;
    };
    if !buttons.pressed(MouseButton::Left) {
        grips.drag = None;
    }
    let Some(cursor) = windows.single().ok().and_then(Window::cursor_position) else {
        return;
    };
    let dims = data.cube.dims;
    let screen = |p: Vec3| {
        camera
            .world_to_viewport(camera_transform, p * UNITY_FROM_LOCAL)
            .ok()
    };
    let ray = camera.viewport_to_world(camera_transform, cursor).ok();
    let (origin, dir) = ray.map_or((Vec3::ZERO, Vec3::Z), |r| {
        (r.origin * UNITY_FROM_LOCAL, *r.direction * UNITY_FROM_LOCAL)
    });
    let forward = camera_transform.forward().as_vec3() * UNITY_FROM_LOCAL;
    let eye = camera_transform.translation() * UNITY_FROM_LOCAL;

    if let Some(GripDrag {
        grip,
        start,
        start_hit,
    }) = grips.drag
    {
        let anchor = grip_position(grip, region_bounds(start, dims));
        let Some(hit) = grip_hit(grip, anchor, origin, dir, forward) else {
            return;
        };
        let target = anchor + hit - start_hit;
        let visible = state.crop.unwrap_or(data.full_region());
        let (mut min, mut max) = start;
        for i in (0..3).filter(|&i| grip[i] != 0) {
            // Voxel boundary nearest the target: object coordinate o is boundary (o + 0.5) * dim.
            let boundary = ((target[i] + 0.5) * dims[i] as f32).round() as i64;
            if grip[i] > 0 {
                max[i] = (boundary - 1).clamp(min[i] as i64, visible.1[i] as i64) as usize;
            } else {
                min[i] = boundary.clamp(visible.0[i] as i64, max[i] as i64) as usize;
            }
        }
        if state.selection != Some((min, max)) {
            state.selection = Some((min, max));
        }
        return;
    }

    let bounds = region_bounds(selection, dims);
    grips.hovered = (!egui.wants_pointer_input())
        .then(|| {
            grip_ids()
                .filter_map(|g| {
                    let at = grip_position(g, bounds);
                    // A face grip is its whole arrow; a corner grip is a point.
                    let distance = if is_face(g) {
                        distance_to_segment(cursor, screen(at)?, screen(arrow_tip(g, at, eye))?)
                    } else {
                        screen(at)?.distance(cursor)
                    };
                    Some((g, distance))
                })
                .filter(|(_, d)| *d < 10.0)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(g, _)| g)
        })
        .flatten();
    if buttons.just_pressed(MouseButton::Left)
        && let Some(grip) = grips.hovered
        && let Some(start_hit) = grip_hit(grip, grip_position(grip, bounds), origin, dir, forward)
    {
        grips.drag = Some(GripDrag {
            grip,
            start: selection,
            start_hit,
        });
    }
}

fn draw_grips(
    mut gizmos: Gizmos,
    data: Res<Dataset>,
    state: Res<CubeState>,
    grips: Res<Grips>,
    egui: Res<EguiWantsInput>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &Transform), With<Orbit>>,
) {
    let (Some(selection), Ok((view, camera))) = (state.selection, cameras.single()) else {
        return;
    };
    let cursor = windows
        .single()
        .ok()
        .and_then(Window::cursor_position)
        .filter(|_| !egui.wants_pointer_input());
    let screen = |p: Vec3| {
        view.world_to_viewport(&GlobalTransform::from(*camera), p)
            .ok()
    };
    let bounds = region_bounds(selection, data.cube.dims);
    let active = grips.drag.map(|d| d.grip).or(grips.hovered);
    let eye = camera.translation * UNITY_FROM_LOCAL;
    for grip in grip_ids() {
        let at = grip_position(grip, bounds);
        let p = at * UNITY_FROM_LOCAL;
        let size = (camera.translation - p).length() * GRIP_SIZE;
        let color = if Some(grip) == active {
            Color::srgb(1.0, 0.85, 0.0)
        } else {
            Color::srgb(0.3, 0.6, 1.0)
        };
        if is_face(grip) {
            // An arrow along the direction the face moves.
            let tip = arrow_tip(grip, at, eye) * UNITY_FROM_LOCAL;
            // Arrows stay faint until the pointer comes within about 20 pixels of them.
            let near = match (cursor, screen(p), screen(tip)) {
                (Some(c), Some(a), Some(b)) => distance_to_segment(c, a, b),
                _ => f32::MAX,
            };
            let alpha = if Some(grip) == active {
                1.0
            } else {
                ((80.0 - near) / 60.0).clamp(0.25, 1.0)
            };
            gizmos
                .arrow(p, tip, color.with_alpha(alpha))
                .with_tip_length(size * 1.5);
        } else {
            gizmos.cube(
                Transform::from_translation(p).with_scale(Vec3::splat(size)),
                color,
            );
        }
    }
}

fn distance_to_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-6)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

fn voxel_count((min, max): Region) -> u64 {
    (0..3).map(|i| (max[i] - min[i] + 1) as u64).product()
}

fn format_dims(d: [usize; 3]) -> String {
    format!("{} x {} x {}", d[0], d[1], d[2])
}

/// 1-based inclusive ranges, as iDaVIE displays them.
fn format_region((min, max): Region) -> String {
    format!(
        "x {}..{}, y {}..{}, z {}..{}",
        min[0] + 1,
        max[0] + 1,
        min[1] + 1,
        max[1] + 1,
        min[2] + 1,
        max[2] + 1
    )
}

#[allow(clippy::too_many_arguments)]
fn ui(
    mut contexts: EguiContexts,
    data: Option<Res<Dataset>>,
    state: Option<Res<CubeState>>,
    settings: Option<ResMut<Settings>>,
    error: Option<Res<LoadError>>,
    hover: Res<Hover>,
    diagnostics: Res<DiagnosticsStore>,
    windows: Query<&Window>,
    mut crop: MessageWriter<CropCommand>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    egui::Window::new("iDaVIE volume renderer").default_width(340.0).show(ctx, |ui| {
        let smoothed = |path: &bevy::diagnostic::DiagnosticPath| diagnostics.get(path).and_then(|d| d.smoothed()).unwrap_or(0.0);
        let fps = smoothed(&FrameTimeDiagnosticsPlugin::FPS);
        let frame_ms = smoothed(&FrameTimeDiagnosticsPlugin::FRAME_TIME);
        let resolution = windows.single().map_or(String::new(), |w| format!(", {} x {} px", w.physical_width(), w.physical_height()));
        ui.label(format!("{fps:.0} fps, {frame_ms:.2} ms/frame{resolution}"));

        let (Some(data), Some(state), Some(mut settings)) = (data, state, settings) else {
            ui.label(error.map_or_else(|| "Choose a FITS file to load".to_string(), |e| e.0.clone()));
            return;
        };
        let cube = &data.cube;
        let stats = cube.stats;
        let gb = cube.data.len() as f64 * 4.0 / 1e9;
        let unit = cube.unit().map_or(String::new(), |u| format!(" {u}"));

        egui::CollapsingHeader::new("Cube").default_open(true).show(ui, |ui| {
            egui::Grid::new("cube").num_columns(2).show(ui, |ui| {
                ui.label("File");
                ui.label(format!("{} (HDU {})", data.name, cube.hdu));
                ui.end_row();
                ui.label("Dimensions");
                ui.label(format!("{} ({gb:.2} GB)", format_dims(cube.dims)));
                ui.end_row();
                ui.label("Range");
                ui.label(format!("{:.4e} .. {:.4e}{unit}", stats.min, stats.max));
                ui.end_row();
                ui.label("Mean, std");
                ui.label(format!("{:.4e}, {:.4e}", stats.mean, stats.std));
                ui.end_row();
                ui.label("NaN voxels");
                ui.label(stats.nan_count.to_string());
                ui.end_row();
                ui.label("Load time");
                ui.label(format!("{:.2} s", data.load_seconds));
                ui.end_row();
                ui.label("Texture");
                let [fx, fy, fz] = state.factors;
                let resolution = if state.factors == [1; 3] { "full resolution".to_string() } else { format!("downsampled {fx} x {fy} x {fz}") };
                let mb = state.texture_dims.iter().product::<usize>() as f64 * 4.0 / 1e6;
                ui.label(format!("{} ({mb:.0} MB), {resolution}", format_dims(state.texture_dims)));
                ui.end_row();
                ui.label("Showing");
                ui.label(state.crop.map_or("entire cube".to_string(), format_region));
                ui.end_row();
                if let Some(selection) = state.selection {
                    ui.label("Selection");
                    ui.label(format_region(selection));
                    ui.end_row();
                }
                if let Some(v) = hover.0 {
                    let index = v[0] + cube.dims[0] * (v[1] + cube.dims[1] * v[2]);
                    ui.label("Cursor");
                    ui.label(format!("({}, {}, {}) = {:.4e}{unit}", v[0] + 1, v[1] + 1, v[2] + 1, cube.data[index]));
                    ui.end_row();
                    if let Some(world) = data.wcs.as_ref().and_then(|w| w.grid_to_world(v.map(|i| i as f64 + 1.0))) {
                        ui.label("");
                        ui.label(world.join(", "));
                        ui.end_row();
                    }
                }
            });
            ui.horizontal(|ui| {
                if ui.add_enabled(state.selection.is_some(), egui::Button::new("Crop to selection")).clicked() {
                    crop.write(CropCommand::ToSelection);
                }
                if ui.add_enabled(state.crop.is_some(), egui::Button::new("Reset crop")).clicked() {
                    crop.write(CropCommand::Reset);
                }
            });
        });

        if let Some(wcs) = &data.wcs {
            egui::CollapsingHeader::new("WCS").show(ui, |ui| {
                for line in wcs.summary() {
                    ui.label(line);
                }
            });
        }

        egui::CollapsingHeader::new("Rendering").default_open(true).show(ui, |ui| {
            let mut s = settings.clone();
            let p = &mut s.params;
            egui::ComboBox::from_label("Colour map").selected_text(COLOR_MAPS[p.color_map_index as usize]).show_ui(ui, |ui| {
                for (i, name) in COLOR_MAPS.iter().enumerate() {
                    ui.selectable_value(&mut p.color_map_index, i as f32, *name);
                }
            });
            egui::ComboBox::from_label("Scaling").selected_text(SCALING_TYPES[p.scale_type as usize]).show_ui(ui, |ui| {
                for (i, name) in SCALING_TYPES.iter().enumerate() {
                    ui.selectable_value(&mut p.scale_type, i as u32, *name);
                }
            });
            if SCALING_TYPES[p.scale_type as usize] == "Gamma" {
                ui.add(egui::Slider::new(&mut p.scale_gamma, 0.0..=5.0).text("Gamma"));
            }
            let value = |t: f32| p.scale_min + t * (p.scale_max - p.scale_min);
            let (min_value, max_value) = (value(p.threshold_min), value(p.threshold_max));
            ui.add(egui::Slider::new(&mut p.threshold_min, 0.0..=1.0).text(format!("Min ({min_value:.3e})")));
            ui.add(egui::Slider::new(&mut p.threshold_max, 0.0..=1.0).text(format!("Max ({max_value:.3e})")));
            ui.horizontal(|ui| {
                ui.radio_value(&mut p.projection_mode, 0, "Maximum");
                ui.radio_value(&mut p.projection_mode, 1, "Average");
                ui.label("intensity projection");
            });
            ui.add(egui::Slider::new(&mut p.jitter, 0.0..=1.0).text("Jitter"));
            ui.add(egui::Slider::new(&mut p.max_steps, 16.0..=512.0).step_by(1.0).text("Ray-marching steps"));
            if data.wcs.is_some() {
                ui.checkbox(&mut s.wcs_axes, "WCS axes");
            }
            settings.set_if_neq(s);
        });

        ui.separator();
        ui.small("Drag: orbit. Right or middle drag: pan. Scroll: zoom. Shift+drag: select region. Drag a blue arrow or corner: resize. C or Enter: crop to selection. R: reset crop. Esc: clear selection.");
    });
    Ok(())
}

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use std::sync::Mutex;

    static PENDING: Mutex<Option<(String, Vec<u8>, String, String)>> = Mutex::new(None);

    /// Called from web/index.html with a picked or fetched file; `overrides` is one `key=value` per line.
    #[wasm_bindgen::prelude::wasm_bindgen]
    pub fn load_fits(name: String, bytes: Vec<u8>, config_toml: String, overrides: String) {
        *PENDING.lock().unwrap() = Some((name, bytes, config_toml, overrides));
    }

    pub fn poll(mut commands: Commands) {
        let Some((name, bytes, config_toml, overrides)) = PENDING.lock().unwrap().take() else {
            return;
        };
        let start = Instant::now();
        let overrides: Vec<(String, String)> = overrides
            .lines()
            .filter_map(|kv| kv.split_once('='))
            .map(|(k, v)| (k.replace('-', "_"), v.to_string()))
            .collect();
        let loaded = config::from_toml(
            (!config_toml.is_empty()).then_some(config_toml.as_str()),
            &overrides,
        )
        .and_then(|config| Ok((config, fits::read_fits(bytes.as_slice(), None)?)));
        match loaded {
            Ok((config, cube)) => {
                commands.insert_resource(Dataset::new(name, cube, config, start));
                commands.remove_resource::<LoadError>();
            }
            Err(e) => commands.insert_resource(LoadError(format!("{name}: {e}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_edges_from_a_face_a_corner_and_inside() {
        assert_eq!(hidden_edges(Vec3::new(0.0, 0.0, 10.0)).count_ones(), 8);
        let corner = hidden_edges(Vec3::splat(10.0));
        assert_eq!(corner.count_ones(), 3);
        // Those three meet at the far corner, (-0.5, -0.5, -0.5).
        assert!(
            box_edges()
                .enumerate()
                .all(|(i, (a, _))| corner >> i & 1 == 0 || a == 0)
        );
        assert_eq!(hidden_edges(Vec3::ZERO), 0xfff);
    }

    #[test]
    fn grips_follow_their_constraint() {
        assert_eq!(grip_ids().count(), 14);
        let down = Vec3::NEG_Z;
        let face = IVec3::new(1, 0, 0);
        let hit = grip_hit(
            face,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(2.0, 0.3, 5.0),
            down,
            down,
        );
        assert!(hit.unwrap().distance(Vec3::new(2.0, 0.0, 0.0)) < 1e-5);
        // Looking straight down a face grip's axis gives no hit, rather than a jump.
        assert!(grip_hit(IVec3::new(0, 0, 1), Vec3::ZERO, Vec3::Z, down, down).is_none());
        // A corner moves in the plane facing the camera.
        let corner = IVec3::new(1, 1, 1);
        let hit = grip_hit(
            corner,
            Vec3::splat(0.5),
            Vec3::new(0.7, 0.8, 5.0),
            down,
            down,
        );
        assert!(hit.unwrap().distance(Vec3::new(0.7, 0.8, 0.5)) < 1e-5);
        let bounds = (Vec3::splat(-0.5), Vec3::new(0.5, 0.25, 0.5));
        assert_eq!(
            grip_position(IVec3::new(-1, 1, 1), bounds),
            Vec3::new(-0.5, 0.25, 0.5)
        );
    }
}
