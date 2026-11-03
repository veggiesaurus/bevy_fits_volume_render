//! Shift-drag region selection, and cropping the volume to it.

use bevy::{prelude::*, render::renderer::RenderDevice};
use bevy_egui::input::EguiWantsInput;

use crate::{
    CubeState, Dataset, UNITY_FROM_LOCAL, VolumeTexture,
    camera::{DEFAULT_DISTANCE, Orbit, pointer},
    fits::Region,
    grips::GripDrag,
    outline::box_transform,
    region_bounds,
    source::{DataSource, Request},
    upload::Uploads,
};

/// The voxel under the cursor, and its value once the data source has sent it.
#[derive(Resource, Default)]
pub struct Hover {
    pub voxel: Option<[usize; 3]>,
    /// The latest value sent back, and the voxel it's for.
    pub value: Option<([usize; 3], f32)>,
    asked: Option<(u32, [usize; 3])>,
}

impl Hover {
    fn ask(&mut self, source: &DataSource, voxel: [usize; 3]) {
        if self.asked.is_some_and(|(_, asked)| asked == voxel) {
            return;
        }
        let id = self.asked.map_or(0, |(id, _)| id.wrapping_add(1));
        source.send(Request::PixelValue { id, voxel });
        self.asked = Some((id, voxel));
    }

    /// A value for a voxel the cursor has since left is dropped.
    pub fn receive(&mut self, id: u32, value: f32) {
        if let Some((asked, voxel)) = self.asked
            && asked == id
        {
            self.value = Some((voxel, value));
        }
    }
}

/// A crop on its way: its texture built in the background, then uploaded over a few frames.
#[derive(Resource, Default)]
pub struct PendingCrop(pub Option<VolumeTexture>);

impl PendingCrop {
    fn cancel(&mut self, source: &DataSource) {
        if let Some(crop) = self.0.take() {
            source.cancel(crop.handle.id());
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
    let dims = data.info.dims;
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
    hover.voxel = voxel
        .filter(|v| inside(v) && !egui.wants_pointer_input())
        .map(|v| clamp_to(v, visible));
    if let Some(v) = hover.voxel {
        hover.ask(&data.source, v);
    }

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
    device: Res<RenderDevice>,
) {
    let mut framed = false;
    for request in requests.read() {
        match request {
            CropRequest::ToSelection => {
                let Some(region) = state.selection else {
                    continue;
                };
                if pending.0.as_ref().is_some_and(|crop| crop.region == region) {
                    continue;
                }
                // Cropping back to the region on screen drops a different crop that's on its way.
                pending.cancel(&data.source);
                if state.crop.as_ref().is_none_or(|crop| crop.region != region) {
                    pending.0 = Some(data.texture(region, &device, &mut images, &mut uploads));
                }
            }
            CropRequest::Reset => {
                pending.cancel(&data.source);
                state.crop = None;
                framed = true;
            }
        }
    }
    if (pending.0.as_ref()).is_some_and(|crop| !uploads.is_filling(crop.handle.id())) {
        state.crop = pending.0.take();
        framed = true;
    }
    if framed {
        let visible = box_transform(state.visible(), data.info.dims);
        orbit.target = visible.translation;
        orbit.distance = DEFAULT_DISTANCE * visible.scale.max_element();
    }
}
