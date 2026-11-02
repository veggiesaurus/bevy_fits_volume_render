//! Grips on the selection box's faces and corners for resizing it.

use bevy::prelude::*;
use bevy_egui::input::EguiWantsInput;

use crate::{CubeState, Dataset, Region, UNITY_FROM_LOCAL, camera::Orbit, region_bounds};

/// A grip on the selection box, as -1 (low face), 0 (middle) or +1 (high face) per axis: one non-zero
/// component is a face grip and three a corner grip.
pub fn grip_ids() -> impl Iterator<Item = IVec3> {
    (-1..=1)
        .flat_map(|x| (-1..=1).flat_map(move |y| (-1..=1).map(move |z| IVec3::new(x, y, z))))
        .filter(|g| matches!(g.abs().element_sum(), 1 | 3))
}

pub fn is_face(grip: IVec3) -> bool {
    grip.abs().element_sum() == 1
}

/// Grips and their face arrows are sized as fractions of the distance to the camera, so they keep a
/// constant size on screen.
pub const GRIP_SIZE: f32 = 0.01;

pub const ARROW_LENGTH: f32 = 0.06;

/// The tip of a face grip's arrow, which starts at the face centre and points outward.
pub fn arrow_tip(grip: IVec3, position: Vec3, eye: Vec3) -> Vec3 {
    position + grip.as_vec3() * (eye - position).length() * ARROW_LENGTH
}

/// A grip's position in iDaVIE (Unity) object space, on the box `lo`..`hi`.
pub fn grip_position(grip: IVec3, (lo, hi): (Vec3, Vec3)) -> Vec3 {
    let middle = (lo + hi) / 2.0;
    Vec3::select(
        grip.cmplt(IVec3::ZERO),
        lo,
        Vec3::select(grip.cmpgt(IVec3::ZERO), hi, middle),
    )
}

/// Where the pointer ray (object space) meets a grip's constraint: the axis line through a face grip,
/// or the plane facing the camera for a corner grip.
pub fn grip_hit(grip: IVec3, anchor: Vec3, origin: Vec3, dir: Vec3, forward: Vec3) -> Option<Vec3> {
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
pub struct Grips {
    pub hovered: Option<IVec3>,
    pub drag: Option<GripDrag>,
}

#[derive(Clone, Copy)]
pub struct GripDrag {
    pub grip: IVec3,
    pub start: Region,
    pub start_hit: Vec3,
}

/// Autodesk-style resizing: drag a face or corner grip of the selection box. Faces snap to voxels and
/// stay inside the visible region, and a face can't pass the opposite one.
pub fn drag_grips(
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
    let ray = camera
        .viewport_to_world(camera_transform, cursor)
        .ok()
        .map(|r| (r.origin * UNITY_FROM_LOCAL, *r.direction * UNITY_FROM_LOCAL));
    let forward = camera_transform.forward().as_vec3() * UNITY_FROM_LOCAL;
    let eye = camera_transform.translation() * UNITY_FROM_LOCAL;

    if let Some(GripDrag {
        grip,
        start,
        start_hit,
    }) = grips.drag
    {
        let anchor = grip_position(grip, region_bounds(start, dims));
        let Some(hit) = ray.and_then(|(origin, dir)| grip_hit(grip, anchor, origin, dir, forward))
        else {
            return;
        };
        let target = anchor + hit - start_hit;
        let visible = state.visible(&data);
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
        && let Some((origin, dir)) = ray
        && let Some(start_hit) = grip_hit(grip, grip_position(grip, bounds), origin, dir, forward)
    {
        grips.drag = Some(GripDrag {
            grip,
            start: selection,
            start_hit,
        });
    }
}

pub fn draw_grips(
    mut gizmos: Gizmos,
    data: Res<Dataset>,
    state: Res<CubeState>,
    grips: Res<Grips>,
    egui: Res<EguiWantsInput>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &Transform), With<Orbit>>,
) {
    let (Some(selection), Ok((camera, transform))) = (state.selection, cameras.single()) else {
        return;
    };
    let cursor = windows
        .single()
        .ok()
        .and_then(Window::cursor_position)
        .filter(|_| !egui.wants_pointer_input());
    // The local transform is already this frame's orbit; GlobalTransform may not be propagated yet.
    let view = GlobalTransform::from(*transform);
    let screen = |p: Vec3| camera.world_to_viewport(&view, p).ok();
    let bounds = region_bounds(selection, data.cube.dims);
    let active = grips.drag.map(|d| d.grip).or(grips.hovered);
    let eye = transform.translation * UNITY_FROM_LOCAL;
    for grip in grip_ids() {
        let at = grip_position(grip, bounds);
        let p = at * UNITY_FROM_LOCAL;
        let size = (transform.translation - p).length() * GRIP_SIZE;
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

pub fn distance_to_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-6)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

#[cfg(test)]
mod tests {
    use super::*;

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
