//! The orbiting camera: drag to orbit, right or middle drag to pan, scroll to zoom.

use bevy::{
    core_pipeline::prepass::DepthPrepass,
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit},
    prelude::*,
    render::view::{DebandDither, Msaa, Tonemapping},
};
use bevy_egui::input::EguiWantsInput;

use crate::{UNITY_FROM_LOCAL, selection::Gesture};

pub const DEFAULT_DISTANCE: f32 = 2.0;

/// The near plane as a fraction of the orbit distance: iDaVIE's 0.05 at the default distance, and it
/// follows the camera in when a crop or zoom brings it close.
const NEAR_FRACTION: f32 = 0.025;

#[derive(Component)]
pub struct Orbit {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

impl Default for Orbit {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            distance: DEFAULT_DISTANCE,
        }
    }
}

/// The cursor's window position, and the ray under it in iDaVIE (Unity) object space.
pub struct Pointer {
    pub cursor: Vec2,
    pub ray: Ray3d,
}

pub fn pointer(window: &Window, camera: &Camera, view: &GlobalTransform) -> Option<Pointer> {
    let cursor = window.cursor_position()?;
    let ray = camera.viewport_to_world(view, cursor).ok()?;
    Some(Pointer {
        cursor,
        // Mirroring keeps the direction a unit vector.
        ray: Ray3d::new(
            ray.origin * UNITY_FROM_LOCAL,
            Dir3::new_unchecked(*ray.direction * UNITY_FROM_LOCAL),
        ),
    })
}

pub fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        // Unity's camera field of view is vertical, like Bevy's.
        Projection::Perspective(PerspectiveProjection {
            fov: 60f32.to_radians(),
            near: DEFAULT_DISTANCE * NEAR_FRACTION,
            ..default()
        }),
        // Unity's linear colour space pipeline: no tone curve, sRGB encode on output.
        Tonemapping::None,
        DebandDither::Disabled,
        Msaa::Off,
        // The volume shader clamps its rays against this, as iDaVIE's does against _CameraDepthTexture.
        DepthPrepass,
        Orbit::default(),
    ));
}

pub fn orbit_camera(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    egui: Res<EguiWantsInput>,
    mut gesture: ResMut<Gesture>,
    camera: Single<(&mut Orbit, &mut Transform, &mut Projection)>,
) {
    let (mut orbit, mut transform, mut projection) = camera.into_inner();
    let shift = keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
    if !shift && gesture.can_start(&buttons, &egui) {
        *gesture = Gesture::Orbit;
    }
    // A tap can press and release in one frame, which shouldn't turn that frame's pointer motion into an orbit.
    if matches!(*gesture, Gesture::Orbit) && buttons.pressed(MouseButton::Left) {
        orbit.yaw -= motion.delta.x * 0.005;
        orbit.pitch = (orbit.pitch - motion.delta.y * 0.005).clamp(-1.55, 1.55);
    }
    if !egui.wants_pointer_input() {
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
    transform.set_if_neq(
        Transform::from_translation(orbit.target + offset).looking_at(orbit.target, Vec3::Y),
    );
    // Written only when it moves, so the projection isn't marked changed every frame.
    let near = orbit.distance * NEAR_FRACTION;
    if let Projection::Perspective(perspective) = projection.bypass_change_detection()
        && perspective.near != near
    {
        perspective.near = near;
        projection.set_changed();
    }
}
