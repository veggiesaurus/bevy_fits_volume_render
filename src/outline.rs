//! The box outlines around the visible region and the selection, drawn as opaque line meshes.

use bevy::{asset::RenderAssetUsages, mesh::PrimitiveTopology, prelude::*};

use crate::{CubeState, Dataset, Region, UNITY_FROM_LOCAL, camera::Orbit, fits, region_bounds};

#[derive(Component)]
pub struct SelectionOutline;

#[derive(Component)]
pub struct CubeOutline;

/// A box outline drawn as line meshes. `axes` tints the edges at the first voxel's corner as iDaVIE does, and
/// `written` caches the hidden-edge mask and tint last written to its colours.
#[derive(Component, Default)]
pub struct Outline {
    pub axes: bool,
    written: Option<(u16, bool)>,
}

impl Outline {
    pub fn with_axes() -> Self {
        Self {
            axes: true,
            ..default()
        }
    }
}

#[derive(Resource)]
pub struct OutlineMaterials {
    pub white: Handle<StandardMaterial>,
    pub yellow: Handle<StandardMaterial>,
}

/// The 12 edges of a unit cube centred on the origin, in `edge_ends` order.
pub fn box_lines() -> Mesh {
    let positions: Vec<Vec3> = (0..12)
        .flat_map(|e| {
            let (start, end) = edge_ends(e);
            [start, end].map(|p| Vec3::from_array(p) - 0.5)
        })
        .collect();
    // Kept in the main world too, so shade_hidden_edges can rewrite the colours.
    Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, vec![[1.0f32; 4]; 24])
}

/// Ends of box edge `e` in unit-box coordinates: it runs along axis `e / 4`, and the low bits of `e`
/// put it on the low or high side of the other two axes.
pub fn edge_ends(e: usize) -> ([f32; 3], [f32; 3]) {
    let axis = e / 4;
    let mut start = [0.0; 3];
    start[(axis + 1) % 3] = (e & 1) as f32;
    start[(axis + 2) % 3] = ((e >> 1) & 1) as f32;
    let mut end = start;
    end[axis] = 1.0;
    (start, end)
}

/// Bit e is set when edge e (see `edge_ends`) is hidden from `eye`, in the unit cube's own coordinates:
/// neither face it borders faces the eye.
pub fn hidden_edges(eye: Vec3) -> u16 {
    (0..12).fold(0, |mask, e| {
        let (start, _) = edge_ends(e);
        let seen = (0..3).filter(|&axis| axis != e / 4).any(|axis| {
            if start[axis] == 1.0 {
                eye[axis] > 0.5
            } else {
                eye[axis] < -0.5
            }
        });
        if seen { mask } else { mask | 1 << e }
    })
}

/// iDaVIE's X, Y and Z colours on the three edges that meet at the first voxel's corner, white elsewhere. Unity
/// mirrors z, so that corner is at (0, 0, 1) in `edge_ends` coordinates.
fn edge_colour(e: usize, axes: bool) -> LinearRgba {
    let axis = e / 4;
    let (start, _) = edge_ends(e);
    if axes && (0..3).all(|i| i == axis || start[i] == [0.0, 0.0, 1.0][i]) {
        let mut rgb = [0.3; 3];
        rgb[axis] = 1.0;
        Color::srgb_from_array(rgb).into()
    } else {
        LinearRgba::WHITE
    }
}

/// Draws an outline's hidden edges dark so the box reads as solid on a flat screen.
pub fn shade_hidden_edges(
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
        let written = (hidden_edges(eye), outline.axes);
        if outline.written == Some(written) {
            continue;
        }
        if let Some(mut mesh) = meshes.get_mut(&mesh.0) {
            let colours: Vec<[f32; 4]> = (0..24)
                .map(|v| {
                    let colour = edge_colour(v / 2, outline.axes);
                    // Linear 0.1 is about 35% grey once encoded to sRGB.
                    let shade = if written.0 >> (v / 2) & 1 == 1 {
                        0.1
                    } else {
                        1.0
                    };
                    (colour * shade).with_alpha(1.0).to_f32_array()
                })
                .collect();
            mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
            outline.written = Some(written);
        }
    }
}

pub fn box_transform(region: Region, dims: [usize; 3]) -> Transform {
    let (lo, hi) = region_bounds(region, dims);
    Transform::from_translation((lo + hi) / 2.0 * UNITY_FROM_LOCAL).with_scale(hi - lo)
}

/// The cube outline follows the visible (cropped) region; the selection outline follows the selection.
pub fn update_outlines(
    data: Res<Dataset>,
    state: Res<CubeState>,
    outline_materials: Res<OutlineMaterials>,
    mut cube: Query<&mut Transform, (With<CubeOutline>, Without<SelectionOutline>)>,
    mut outline: Query<
        (
            &mut Transform,
            &mut Visibility,
            &mut MeshMaterial3d<StandardMaterial>,
            &mut Outline,
        ),
        With<SelectionOutline>,
    >,
) {
    if !state.is_changed() {
        return;
    }
    for mut transform in &mut cube {
        *transform = box_transform(state.visible(&data), data.cube.dims);
    }
    let Ok((mut transform, mut visibility, mut material, mut outline)) = outline.single_mut()
    else {
        return;
    };
    let Some(selection) = state.selection else {
        *visibility = Visibility::Hidden;
        return;
    };
    *transform = box_transform(selection, data.cube.dims);
    // iDaVIE colours the region white, with tinted axes, when cropping to it would load at full resolution, and
    // yellow otherwise.
    let full_res =
        fits::find_downsample_factors(data.config.gpu_memory_limit_mb, selection.extent())
            == [1; 3];
    outline.axes = full_res;
    material.0 = if full_res {
        outline_materials.white.clone()
    } else {
        outline_materials.yellow.clone()
    };
    *visibility = Visibility::Visible;
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
        assert!((0..12).all(|e| corner >> e & 1 == 0 || edge_ends(e).0 == [0.0; 3]));
        assert_eq!(hidden_edges(Vec3::ZERO), 0xfff);
    }

    #[test]
    fn axis_tints_meet_at_the_first_voxel() {
        let tinted: Vec<usize> = (0..12)
            .filter(|&e| edge_colour(e, true) != LinearRgba::WHITE)
            .collect();
        assert_eq!(tinted, [2, 5, 8]);
        for e in tinted {
            let (start, end) = edge_ends(e);
            assert!(start == [0.0, 0.0, 1.0] || end == [0.0, 0.0, 1.0]);
        }
    }
}
