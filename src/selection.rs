//! Shift-drag region selection, and cropping the volume to it.

use bevy::{platform::time::Instant, prelude::*};
use bevy_egui::input::EguiWantsInput;

use crate::{
    CubeState, Dataset, Region, UNITY_FROM_LOCAL,
    camera::{DEFAULT_DISTANCE, Orbit},
    grips::Grips,
    region_bounds, region_texture,
};

#[derive(Resource, Default)]
pub struct Hover(pub Option<[usize; 3]>);

#[derive(Message)]
pub enum CropCommand {
    ToSelection,
    Reset,
}

/// What a left-button drag does, decided when the button goes down so that pressing or releasing Shift
/// mid-drag can't switch it. Grip drags live in `Grips`.
#[derive(Resource, Default, Clone, Copy)]
pub enum Gesture {
    #[default]
    Idle,
    Orbit,
    Select(Drag),
}

/// An in-progress shift-drag: the axis facing the camera, the voxel it started on, and the depth range it keeps.
#[derive(Clone, Copy)]
pub struct Drag {
    pub axis: usize,
    pub start: [usize; 3],
    pub depth: [usize; 2],
}

/// Shift-drag selects a box: the two axes facing the camera follow the pointer on a plane through the
/// middle of the visible region, the depth axis keeps the previous selection's range (or the full range).
pub fn select_region(
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
    mut gesture: ResMut<Gesture>,
) {
    let Ok((camera, camera_transform)) = cameras.single() else {
        return;
    };
    if !buttons.pressed(MouseButton::Left) && matches!(*gesture, Gesture::Select(_)) {
        *gesture = Gesture::Idle;
    }
    let dims = data.cube.dims;
    let visible = state.visible(&data);
    let axis = match *gesture {
        Gesture::Select(drag) => drag.axis,
        _ => (camera_transform.forward().as_vec3() * UNITY_FROM_LOCAL)
            .abs()
            .max_position(),
    };
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
        && matches!(*gesture, Gesture::Idle)
        && let Some(v) = pointer
    {
        let depth = state.selection.unwrap_or(visible);
        *gesture = Gesture::Select(Drag {
            axis,
            start: clamp_to(v, visible),
            depth: [depth.0[axis], depth.1[axis]],
        });
    }
    if let (Gesture::Select(Drag { axis, start, depth }), Some(v)) = (*gesture, pointer) {
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
            if matches!(*gesture, Gesture::Select(_)) {
                *gesture = Gesture::Idle;
            }
        }
        if keys.any_just_pressed([KeyCode::KeyC, KeyCode::Enter]) {
            crop.write(CropCommand::ToSelection);
        }
        // As the Reset button, which is disabled with no crop.
        if keys.just_pressed(KeyCode::KeyR) && state.crop.is_some() {
            crop.write(CropCommand::Reset);
        }
    }
}

pub fn apply_crop(
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

        let (lo, hi) = region_bounds(state.visible(&data), data.cube.dims);
        for mut orbit in &mut orbits {
            orbit.target = (lo + hi) / 2.0 * UNITY_FROM_LOCAL;
            orbit.distance = DEFAULT_DISTANCE * (hi - lo).max_element();
        }
    }
}
