//! Shift-drag region selection, and cropping the volume to it.

use bevy::prelude::*;
use bevy_egui::input::EguiWantsInput;

use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future::poll_once};

use crate::{
    CubeState, Dataset, TextureBytes, UNITY_FROM_LOCAL, VolumeTexture,
    camera::{DEFAULT_DISTANCE, Orbit, pointer},
    fits::Region,
    grips::GripDrag,
    outline::box_transform,
    region_bounds, region_bytes,
    upload::Uploads,
};

#[derive(Resource, Default)]
pub struct Hover(pub Option<[usize; 3]>);

/// A crop on its way: its texture built in the background, then uploaded over a few frames.
#[derive(Resource, Default)]
pub struct PendingCrop(pub Option<CropStage>);

pub enum CropStage {
    Building(Region, Task<TextureBytes>),
    Uploading(VolumeTexture),
}

impl CropStage {
    fn region(&self) -> Region {
        match self {
            CropStage::Building(region, _) => *region,
            CropStage::Uploading(texture) => texture.region,
        }
    }
}

#[derive(Message)]
pub enum CropRequest {
    ToSelection,
    Reset,
}

/// What a left-button drag does, decided when the button goes down so that pressing or releasing Shift
/// mid-drag can't switch it.
#[derive(Resource, Default, Clone, Copy, PartialEq)]
pub enum Gesture {
    #[default]
    Idle,
    Orbit,
    Select(SelectDrag),
    Grip(GripDrag),
}

impl Gesture {
    pub fn can_start(&self, buttons: &ButtonInput<MouseButton>, egui: &EguiWantsInput) -> bool {
        buttons.just_pressed(MouseButton::Left)
            && !egui.wants_pointer_input()
            && matches!(self, Gesture::Idle)
    }
}

pub fn end_gesture(buttons: Res<ButtonInput<MouseButton>>, mut gesture: ResMut<Gesture>) {
    if !buttons.pressed(MouseButton::Left) {
        gesture.set_if_neq(Gesture::Idle);
    }
}

/// An in-progress shift-drag: the axis facing the camera, the voxel it started on, and the depth range it keeps.
#[derive(Clone, Copy, PartialEq)]
pub struct SelectDrag {
    axis: usize,
    start: [usize; 3],
    depth: [usize; 2],
}

/// Shift-drag selects a box: the two axes facing the camera follow the pointer on a plane through the
/// middle of the visible region, the depth axis keeps the previous selection's range (or the visible region's).
pub fn select_region(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    egui: Res<EguiWantsInput>,
    window: Single<&Window>,
    camera: Single<(&Camera, &GlobalTransform), With<Orbit>>,
    data: Res<Dataset>,
    mut state: ResMut<CubeState>,
    mut hover: ResMut<Hover>,
    mut gesture: ResMut<Gesture>,
) {
    let (camera, camera_transform) = *camera;
    let dims = data.cube.dims;
    let visible = state.visible();
    let axis = match *gesture {
        Gesture::Select(drag) => drag.axis,
        _ => (camera_transform.forward().as_vec3() * UNITY_FROM_LOCAL)
            .abs()
            .max_position(),
    };
    let (lo, hi) = region_bounds(visible, dims);
    let normal = Vec3::AXES[axis];
    let middle = normal * ((lo[axis] + hi[axis]) / 2.0);

    let voxel = pointer(&window, camera, camera_transform)
        .and_then(|p| {
            p.ray
                .plane_intersection_point(middle, InfinitePlane3d::new(normal))
        })
        .map(|p| {
            std::array::from_fn::<i64, 3, _>(|i| ((p[i] + 0.5) * dims[i] as f32).floor() as i64)
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
    if shift
        && gesture.can_start(&buttons, &egui)
        && let Some(v) = voxel
    {
        let depth = state.selection.unwrap_or(visible);
        *gesture = Gesture::Select(SelectDrag {
            axis,
            start: clamp_to(v, visible),
            depth: [depth.min[axis], depth.max[axis]],
        });
    }
    if let (Gesture::Select(SelectDrag { axis, start, depth }), Some(v)) = (*gesture, voxel) {
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
}

/// Esc clears the selection, C or Enter crops to it and R resets the crop.
pub fn selection_keys(
    keys: Res<ButtonInput<KeyCode>>,
    egui: Res<EguiWantsInput>,
    mut state: ResMut<CubeState>,
    mut gesture: ResMut<Gesture>,
    mut crop_requests: MessageWriter<CropRequest>,
) {
    if egui.wants_keyboard_input() {
        return;
    }
    if keys.just_pressed(KeyCode::Escape) && state.selection.is_some() {
        state.selection = None;
        if matches!(*gesture, Gesture::Select(_) | Gesture::Grip(_)) {
            *gesture = Gesture::Idle;
        }
    }
    if keys.any_just_pressed([KeyCode::KeyC, KeyCode::Enter]) {
        crop_requests.write(CropRequest::ToSelection);
    }
    // As the Reset button, which is disabled with no crop.
    if keys.just_pressed(KeyCode::KeyR) && state.crop.is_some() {
        crop_requests.write(CropRequest::Reset);
    }
}

/// Cropping keeps the selection, as iDaVIE's CropToFeature does, so a reset still shows where the crop was.
/// The crop's texture is built and uploaded in the background, and swapped in when it's ready.
pub fn apply_crop(
    mut requests: MessageReader<CropRequest>,
    data: Res<Dataset>,
    mut state: ResMut<CubeState>,
    mut pending: ResMut<PendingCrop>,
    mut images: ResMut<Assets<Image>>,
    mut uploads: ResMut<Uploads>,
    mut orbit: Single<&mut Orbit>,
) {
    let mut framed = false;
    for request in requests.read() {
        match request {
            CropRequest::ToSelection => {
                let Some(region) = state.selection else {
                    continue;
                };
                let shown = state.crop.as_ref().map(|crop| crop.region);
                let on_its_way = pending.0.as_ref().map(CropStage::region);
                if shown == Some(region) || on_its_way == Some(region) {
                    continue;
                }
                let (cube, config) = (data.cube.clone(), data.config.clone());
                let task = AsyncComputeTaskPool::get()
                    .spawn(async move { region_bytes(&cube, &config, region) });
                pending.0 = Some(CropStage::Building(region, task));
            }
            CropRequest::Reset => {
                pending.0 = None;
                state.crop = None;
                framed = true;
            }
        }
    }
    pending.0 = match pending.0.take() {
        Some(CropStage::Building(region, mut task)) => Some(match block_on(poll_once(&mut task)) {
            Some(bytes) => CropStage::Uploading(bytes.upload(&mut images, &mut uploads)),
            None => CropStage::Building(region, task),
        }),
        Some(CropStage::Uploading(texture)) if !uploads.is_filling(texture.handle.id()) => {
            state.crop = Some(texture);
            framed = true;
            None
        }
        pending => pending,
    };
    if framed {
        let visible = box_transform(state.visible(), data.cube.dims);
        orbit.target = visible.translation;
        orbit.distance = DEFAULT_DISTANCE * visible.scale.max_element();
    }
}
