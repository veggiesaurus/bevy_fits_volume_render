//! The box the cube is drawn as: its voxels' proportions in x and y, the larger of the two 1 wide, and z stretched by
//! the z scale.

use std::ops::RangeInclusive;

use bevy::prelude::*;

use crate::{
    CubeState, Dataset, Settings, UNITY_FROM_LOCAL, camera::Orbit, config::ZScale, fits::Region,
    region_bounds, volume::VolumeMaterial,
};

impl ZScale {
    /// The scale relative to cubic voxels. The Cube preset makes the whole cube as deep as its wider side.
    pub fn value(self, dims: [usize; 3]) -> f32 {
        match self {
            Self::Cube => dims[0].max(dims[1]) as f32 / dims[2] as f32,
            Self::Voxels => 1.0,
            Self::Factor(factor) => factor,
        }
    }
}

/// From a tenth of the smaller preset to ten times the larger.
pub fn z_scale_range(dims: [usize; 3]) -> RangeInclusive<f32> {
    let cube = ZScale::Cube.value(dims);
    cube.min(1.0) / 10.0..=cube.max(1.0) * 10.0
}

/// Bevy's world from iDaVIE's (Unity) object space, where the whole cube is the unit box: z mirrored, and each axis
/// scaled to the box's size.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct BoxShape(Vec3);

impl BoxShape {
    pub fn new(dims: [usize; 3], z_scale: f32) -> Self {
        let [x, y, z] = dims.map(|d| d as f32);
        Self(Vec3::new(x, y, z * z_scale) / x.max(y) * UNITY_FROM_LOCAL)
    }

    /// The whole cube's size in the world.
    pub fn size(self) -> Vec3 {
        self.0.abs()
    }

    pub fn to_world(self, object: Vec3) -> Vec3 {
        object * self.0
    }

    pub fn to_object(self, world: Vec3) -> Vec3 {
        world / self.0
    }

    pub fn ray_to_object(self, ray: Ray3d) -> Option<Ray3d> {
        let direction = Dir3::new(self.to_object(*ray.direction)).ok()?;
        Some(Ray3d::new(self.to_object(ray.origin), direction))
    }

    /// The unit box centred on the origin to a voxel region's box in the world.
    pub fn transform(self, region: Region, dims: [usize; 3]) -> Transform {
        let (lo, hi) = region_bounds(region, dims);
        Transform::from_translation(self.to_world((lo + hi) / 2.0))
            .with_scale((hi - lo) * self.size())
    }
}

/// The volume follows the z scale, and so does the orbit: it stays on the same place in the cube, and keeps its
/// distance in proportion to the one that frames the visible region.
pub fn reshape(
    data: Res<Dataset>,
    settings: Res<Settings>,
    state: Res<CubeState>,
    mut shape: ResMut<BoxShape>,
    mut volume: Query<&mut Transform, With<MeshMaterial3d<VolumeMaterial>>>,
    mut orbit: Single<&mut Orbit>,
) {
    let dims = data.info.dims;
    let new = BoxShape::new(dims, settings.z_scale);
    if new == *shape {
        return;
    }
    let target = new.to_world(shape.to_object(orbit.target));
    let zoom = orbit.distance / orbit.framed;
    orbit.frame(new, state.visible, dims);
    orbit.target = target;
    orbit.distance *= zoom;
    for mut transform in &mut volume {
        transform.scale = new.size();
    }
    *shape = new;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_and_picking_through_a_flattened_box() {
        // M83 CO: the cube preset is as deep as it's wide, and cubic voxels a sixth of that.
        let dims = [1238, 1236, 196];
        let cube = BoxShape::new(dims, ZScale::Cube.value(dims));
        assert!(
            cube.size()
                .abs_diff_eq(Vec3::new(1.0, 1236.0 / 1238.0, 1.0), 1e-6)
        );
        let voxels = BoxShape::new(dims, 1.0);
        assert!(
            voxels
                .size()
                .abs_diff_eq(Vec3::new(1.0, 1236.0 / 1238.0, 196.0 / 1238.0), 1e-6)
        );
        assert!(
            z_scale_range(dims).contains(&0.25)
                && z_scale_range(dims).contains(&ZScale::Cube.value(dims))
        );

        // An oblique ray at a quarter of the cubic voxel scale lands on the voxel it's aimed at, mirrored z and all.
        let shape = BoxShape::new(dims, 0.25);
        let voxel = [300, 900, 150];
        let centre = Vec3::from_array(std::array::from_fn(|i| {
            (voxel[i] as f32 + 0.5) / dims[i] as f32 - 0.5
        }));
        let eye = Vec3::new(0.4, -0.7, 2.0);
        let world = Ray3d::new(eye, Dir3::new(shape.to_world(centre) - eye).unwrap());
        let ray = shape.ray_to_object(world).unwrap();
        let plane = InfinitePlane3d::new(Vec3::Z);
        let hit = ray
            .plane_intersection_point(Vec3::Z * centre.z, plane)
            .unwrap();
        let picked: [usize; 3] =
            std::array::from_fn(|i| ((hit[i] + 0.5) * dims[i] as f32).floor() as usize);
        assert_eq!(picked, voxel);
    }
}
