//! Shift-drag region selection, and cropping the volume to it.

use bevy::prelude::*;
use bevy_egui::input::EguiWantsInput;

use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future::poll_once};

use crate::{
    CubeState, Dataset, Region, TextureBytes, UNITY_FROM_LOCAL, VolumeTexture,
    camera::{DEFAULT_DISTANCE, Orbit, pointer},
    grips::Grips,
    region_bounds, region_bytes,
    upload::Uploads,
};

#[derive(Resource, Default)]
pub struct Hover(pub Option<[usize; 3]>);

/// A crop on its way: its texture built in the background, then uploaded over a few frames.
#[derive(Resource, Default)]
pub struct PendingCrop(pub Option<(Region, CropStage)>);

pub enum CropStage {
    Building(Task<TextureBytes>),
    Uploading(VolumeTexture),
}

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
/// The crop's texture is built and uploaded in the background, and swapped in when it's ready.
pub fn apply_crop(
    mut commands: MessageReader<CropCommand>,
    data: Res<Dataset>,
    mut state: ResMut<CubeState>,
    mut pending: ResMut<PendingCrop>,
    mut images: ResMut<Assets<Image>>,
    mut uploads: ResMut<Uploads>,
    mut orbits: Query<&mut Orbit>,
) {
    let mut framed = false;
    for command in commands.read() {
        match command {
            CropCommand::ToSelection => {
                let Some(region) = state.selection else {
                    continue;
                };
                let shown = state.crop.as_ref().map(|(crop, _)| *crop);
                let building = pending.0.as_ref().map(|(crop, _)| *crop);
                if shown == Some(region) || building == Some(region) {
                    continue;
                }
                let (cube, config) = (data.cube.clone(), data.config.clone());
                let task = AsyncComputeTaskPool::get()
                    .spawn(async move { region_bytes(&cube, &config, region) });
                pending.0 = Some((region, CropStage::Building(task)));
            }
            CropCommand::Reset => {
                pending.0 = None;
                state.crop = None;
                framed = true;
            }
        }
    }
    pending.0 = match pending.0.take() {
        Some((region, CropStage::Building(mut task))) => match block_on(poll_once(&mut task)) {
            Some(bytes) => {
                let texture = bytes.upload(&mut images, &mut uploads);
                Some((region, CropStage::Uploading(texture)))
            }
            None => Some((region, CropStage::Building(task))),
        },
        Some((region, CropStage::Uploading(texture)))
            if !uploads.is_filling(texture.handle.id()) =>
        {
            state.crop = Some((region, texture));
            framed = true;
            None
        }
        pending => pending,
    };
    if framed {
        let (lo, hi) = region_bounds(state.visible(&data), data.cube.dims);
        for mut orbit in &mut orbits {
            orbit.target = (lo + hi) / 2.0 * UNITY_FROM_LOCAL;
            orbit.distance = DEFAULT_DISTANCE * (hi - lo).max_element();
        }
    }
}
