//! Grips on the selection box's faces and corners for resizing it.

use bevy::prelude::*;
use bevy_egui::input::EguiWantsInput;

use crate::{
    CubeState, Dataset, UNITY_FROM_LOCAL,
    camera::{Orbit, Pointer, pointer},
    fits::Region,
    region_bounds,
    selection::Gesture,
    shape::BoxShape,
};

// Fractions of the distance to the camera, so grips and their face arrows keep a constant size on screen.
const GRIP_SIZE: f32 = 0.01;
const ARROW_LENGTH: f32 = 0.06;

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

/// The tip of a face grip's arrow, which starts at the face centre and points outward, in the world.
fn arrow_tip(grip: IVec3, position: Vec3, eye: Vec3) -> Vec3 {
    position + grip.as_vec3() * UNITY_FROM_LOCAL * (eye - position).length() * ARROW_LENGTH
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

/// Where the pointer ray meets a grip's constraint: the axis line through a face grip, or the plane facing the
/// camera for a corner grip. In the world, since the z scale would skew both in object space.
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

/// `start` with the faces `grip` drags moved to the voxel boundaries nearest `target`, kept inside `within`
/// and from passing the opposite faces.
fn resize(start: Region, grip: IVec3, target: Vec3, dims: [usize; 3], within: Region) -> Region {
    let Region { mut min, mut max } = start;
    for i in (0..3).filter(|&i| grip[i] != 0) {
        // Voxel boundary nearest the target: object coordinate o is boundary (o + 0.5) * dim.
        let boundary = ((target[i] + 0.5) * dims[i] as f32).round() as i64;
        if grip[i] > 0 {
            max[i] = (boundary - 1).clamp(min[i] as i64, within.max[i] as i64) as usize;
        } else {
            min[i] = boundary.clamp(within.min[i] as i64, max[i] as i64) as usize;
        }
    }
    Region { min, max }
}

#[derive(Resource, Default)]
pub struct Grips {
    hovered: Option<IVec3>,
}

#[derive(Clone, Copy, PartialEq)]
pub struct GripDrag {
    grip: IVec3,
    start: Region,
    start_hit: Vec3,
}

/// Autodesk-style resizing: drag a face or corner grip of the selection box. Faces snap to voxels and
/// stay inside the visible region, and a face can't pass the opposite one.
pub fn drag_grips(
    buttons: Res<ButtonInput<MouseButton>>,
    egui: Res<EguiWantsInput>,
    window: Single<&Window>,
    camera: Single<(&Camera, &GlobalTransform), With<Orbit>>,
    data: Res<Dataset>,
    shape: Res<BoxShape>,
    mut state: ResMut<CubeState>,
    mut grips: ResMut<Grips>,
    mut gesture: ResMut<Gesture>,
) {
    // A drag's start is in the world of the shape it began in.
    if shape.is_changed() && matches!(*gesture, Gesture::Grip(_)) {
        *gesture = Gesture::Idle;
    }
    let Some(selection) = state.selection else {
        grips.hovered = None;
        return;
    };
    let (camera, camera_transform) = *camera;
    let Some(Pointer { cursor, ray }) = pointer(&window, camera, camera_transform) else {
        return;
    };
    let dims = data.info.dims;
    let screen = |p: Vec3| camera.world_to_viewport(camera_transform, p).ok();
    let forward = camera_transform.forward().as_vec3();
    let eye = camera_transform.translation();
    let world = |grip, bounds| shape.to_world(grip_position(grip, bounds));

    if let Gesture::Grip(GripDrag {
        grip,
        start,
        start_hit,
    }) = *gesture
    {
        let anchor = world(grip, region_bounds(start, dims));
        let Some(hit) = grip_hit(grip, anchor, ray.origin, *ray.direction, forward) else {
            return;
        };
        let target = shape.to_object(anchor + hit - start_hit);
        let selection = resize(start, grip, target, dims, state.visible);
        if state.selection != Some(selection) {
            state.selection = Some(selection);
        }
        return;
    }

    let bounds = region_bounds(selection, dims);
    grips.hovered = if egui.wants_pointer_input() {
        None
    } else {
        grip_ids()
            .filter_map(|g| {
                let at = world(g, bounds);
                let p = screen(at)?;
                // A face grip is its whole arrow; a corner grip is a point.
                let distance = if is_face(g) {
                    let tip = screen(arrow_tip(g, at, eye))?;
                    cursor.distance(Segment2d::new(p, tip).closest_point(cursor))
                } else {
                    p.distance(cursor)
                };
                Some((g, distance))
            })
            .filter(|(_, d)| *d < 10.0)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(g, _)| g)
    };
    if gesture.can_start(&buttons, &egui)
        && let Some(grip) = grips.hovered
        && let Some(start_hit) = grip_hit(
            grip,
            world(grip, bounds),
            ray.origin,
            *ray.direction,
            forward,
        )
    {
        *gesture = Gesture::Grip(GripDrag {
            grip,
            start: selection,
            start_hit,
        });
    }
}

pub fn draw_grips(
    mut gizmos: Gizmos,
    data: Res<Dataset>,
    shape: Res<BoxShape>,
    state: Res<CubeState>,
    grips: Res<Grips>,
    gesture: Res<Gesture>,
    egui: Res<EguiWantsInput>,
    window: Single<&Window>,
    camera: Single<(&Camera, &Transform), With<Orbit>>,
) {
    let Some(selection) = state.selection else {
        return;
    };
    let (camera, transform) = *camera;
    // The local transform is already this frame's orbit; GlobalTransform may not be propagated yet.
    let view = GlobalTransform::from(*transform);
    let cursor = pointer(&window, camera, &view)
        .map(|p| p.cursor)
        .filter(|_| !egui.wants_pointer_input());
    let screen = |p: Vec3| camera.world_to_viewport(&view, p).ok();
    let bounds = region_bounds(selection, data.info.dims);
    let active = match *gesture {
        Gesture::Grip(drag) => Some(drag.grip),
        _ => grips.hovered,
    };
    let eye = transform.translation;
    for grip in grip_ids() {
        let p = shape.to_world(grip_position(grip, bounds));
        let size = (eye - p).length() * GRIP_SIZE;
        let color = if Some(grip) == active {
            Color::srgb(1.0, 0.85, 0.0)
        } else {
            Color::srgb(0.3, 0.6, 1.0)
        };
        if is_face(grip) {
            // An arrow along the direction the face moves.
            let tip = arrow_tip(grip, p, eye);
            // Arrows fade in as the pointer comes within 65 pixels of them, and are solid within 20.
            let near = match (cursor, screen(p), screen(tip)) {
                (Some(c), Some(a), Some(b)) => c.distance(Segment2d::new(a, b).closest_point(c)),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dragged_face_snaps_and_stops_at_the_visible_region_and_the_opposite_face() {
        let (dims, visible) = (
            [10; 3],
            Region {
                min: [2; 3],
                max: [7; 3],
            },
        );
        let start = Region {
            min: [3; 3],
            max: [5; 3],
        };
        assert_eq!(resize(start, IVec3::X, Vec3::ZERO, dims, visible).max[0], 4);
        assert_eq!(
            resize(start, IVec3::X, Vec3::splat(0.5), dims, visible).max[0],
            7
        );
        assert_eq!(
            resize(start, IVec3::NEG_X, Vec3::splat(0.5), dims, visible).min[0],
            5
        );
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
