//! Shift-drag region selection, and cropping the volume to it.

use bevy::prelude::*;
use bevy_egui::input::EguiWantsInput;

use crate::{
    CubeState, Dataset, Region, UNITY_FROM_LOCAL,
    camera::{DEFAULT_DISTANCE, Orbit, pointer},
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

    let voxel = pointer(&windows, camera, camera_transform).and_then(|ray| {
        let t = (plane - ray.origin[axis]) / ray.direction[axis];
        (t.is_finite() && t > 0.0).then(|| {
            let p = ray.origin + ray.direction * t;
            std::array::from_fn::<i64, 3, _>(|i| ((p[i] + 0.5) * dims[i] as f32).floor() as i64)
        })
    });
    let clamp_to = |v: [i64; 3], r: Region| -> [usize; 3] {
        std::array::from_fn(|i| v[i].clamp(r.min[i] as i64, r.max[i] as i64) as usize)
    };
    let inside = |v: &[i64; 3]| {
        (0..3).all(|i| v[i] >= visible.min[i] as i64 && v[i] <= visible.max[i] as i64)
    };
    hover.0 = voxel
        .filter(|v| inside(v) && !egui.wants_pointer_input())
        .map(|v| clamp_to(v, visible));

    let shift = keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
    if buttons.just_pressed(MouseButton::Left)
        && shift
        && !egui.wants_pointer_input()
        && grips.drag.is_none()
        && matches!(*gesture, Gesture::Idle)
        && let Some(v) = voxel
    {
        let depth = state.selection.unwrap_or(visible);
        *gesture = Gesture::Select(Drag {
            axis,
            start: clamp_to(v, visible),
            depth: [depth.min[axis], depth.max[axis]],
        });
    }
    if let (Gesture::Select(Drag { axis, start, depth }), Some(v)) = (*gesture, voxel) {
        let end = clamp_to(v, visible);
        let mut selection = Region {
            min: std::array::from_fn(|i| start[i].min(end[i])),
            max: std::array::from_fn(|i| start[i].max(end[i])),
        };
        (selection.min[axis], selection.max[axis]) = (depth[0], depth[1]);
        if state.selection != Some(selection) {
            state.selection = Some(selection);
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

/// Cropping keeps the selection, as iDaVIE's CropToFeature does, so a reset still shows where the crop was.
pub fn apply_crop(
    mut commands: MessageReader<CropCommand>,
    data: Res<Dataset>,
    mut state: ResMut<CubeState>,
    mut images: ResMut<Assets<Image>>,
    mut orbits: Query<&mut Orbit>,
) {
    for command in commands.read() {
        state.crop = match command {
            CropCommand::ToSelection => {
                let Some(region) = state.selection else {
                    continue;
                };
                if state.crop.as_ref().is_some_and(|(crop, _)| *crop == region) {
                    continue;
                }
                Some((region, region_texture(&data, region, &mut images)))
            }
            CropCommand::Reset => None,
        };
        let (lo, hi) = region_bounds(state.visible(&data), data.cube.dims);
        for mut orbit in &mut orbits {
            orbit.target = (lo + hi) / 2.0 * UNITY_FROM_LOCAL;
            orbit.distance = DEFAULT_DISTANCE * (hi - lo).max_element();
        }
    }
}
