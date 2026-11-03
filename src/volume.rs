use bevy::{
    asset::{RenderAssetUsages, embedded_asset},
    image::{CompressedImageFormats, ImageSampler, ImageType},
    mesh::MeshVertexBufferLayoutRef,
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    render::render_resource::{
        AsBindGroup, CompareFunction, Extent3d, Face, RenderPipelineDescriptor, ShaderType,
        SpecializedMeshPipelineError, TextureDimension, TextureFormat,
    },
    shader::ShaderRef,
    tasks::{ComputeTaskPool, ParallelSliceMut, TaskPool},
};

use crate::config::COLOR_MAPS;

const SHADER: &str = "embedded://bevy_fits_volume_render/volume.wesl";

pub struct VolumePlugin;

impl Plugin for VolumePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "volume.wesl");
        app.add_plugins(MaterialPlugin::<VolumeMaterial>::default());
    }
}

/// Field order must match `VolumeParams` in volume.wesl.
#[derive(Clone, Copy, Debug, PartialEq, ShaderType)]
pub struct VolumeParams {
    pub slice_min: Vec3,
    pub threshold_min: f32,
    pub slice_max: Vec3,
    pub threshold_max: f32,
    pub highlight_min: Vec3,
    pub scale_min: f32,
    pub highlight_max: Vec3,
    pub scale_max: f32,
    pub jitter: f32,
    pub color_map_index: u32,
    pub num_color_maps: u32,
    pub scale_type: u32,
    pub scale_alpha: f32,
    pub scale_gamma: f32,
    pub scale_bias: f32,
    pub scale_contrast: f32,
    pub max_steps: u32,
    pub highlight_saturate_factor: f32,
    pub projection_mode: u32,
    pub empty_space_skipping: u32,
}

impl Default for VolumeParams {
    /// iDaVIE's runtime values: VolumeDataSetRenderer, overridden by Assets/Prefabs/CubePrefab.prefab.
    fn default() -> Self {
        Self {
            slice_min: Vec3::splat(-0.5),
            threshold_min: 0.0,
            slice_max: Vec3::splat(0.5),
            threshold_max: 1.0,
            highlight_min: Vec3::ZERO,
            scale_min: 0.0,
            highlight_max: Vec3::ZERO,
            scale_max: 1.0,
            jitter: 1.0,
            color_map_index: 0,
            num_color_maps: COLOR_MAPS.len() as u32,
            scale_type: 0,
            scale_alpha: 1000.0,
            scale_gamma: 1.0,
            scale_bias: 0.0,
            scale_contrast: 1.0,
            max_steps: 384,
            highlight_saturate_factor: 1.0,
            projection_mode: 0,
            empty_space_skipping: 0,
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct VolumeMaterial {
    #[uniform(0)]
    pub params: VolumeParams,
    // Read with textureLoad: R32Float is filterable on some devices and not others, which a sampler binding can't span.
    #[texture(1, dimension = "3d", sample_type = "float", filterable = false)]
    pub data_cube: Handle<Image>,
    #[texture(3)]
    #[sampler(4)]
    pub color_map: Handle<Image>,
    #[texture(5, dimension = "3d", sample_type = "float", filterable = false)]
    pub brick_range: Handle<Image>,
}

impl Material for VolumeMaterial {
    fn vertex_shader() -> ShaderRef {
        SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        SHADER.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }

    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
    }

    // Unity pass state: Cull Front, ZWrite Off, ZTest Always.
    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.vertex.buffers = vec![
            layout
                .0
                .get_layout(&[Mesh::ATTRIBUTE_POSITION.at_shader_location(0)])?,
        ];
        descriptor.primitive.cull_mode = Some(Face::Front);
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            depth.depth_write_enabled = Some(false);
            depth.depth_compare = Some(CompareFunction::Always);
        }
        Ok(())
    }
}

pub fn cube_image(dims: [usize; 3], data: &[f32]) -> Image {
    let size = Extent3d {
        width: dims[0] as u32,
        height: dims[1] as u32,
        depth_or_array_layers: dims[2] as u32,
    };
    let bytes = bytemuck::cast_slice(data).to_vec();
    Image::new(
        size,
        TextureDimension::D3,
        bytes,
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    )
}

/// Texels per brick edge in `brick_image`, as BRICK in volume.wesl.
pub const BRICK: usize = 8;

/// The min and max of each BRICK³ block of texels, ignoring NaN (so min > max when a block is all NaN),
/// which lets the shader skip blocks that can't change a ray's result.
pub fn brick_image(dims: [usize; 3], data: &[f32]) -> Image {
    let bricks = dims.map(|d| d.div_ceil(BRICK));
    let layer = bricks[0] * bricks[1];
    let mut ranges = vec![[f32::INFINITY, f32::NEG_INFINITY]; layer * bricks[2]];
    ranges.par_chunk_map_mut(
        ComputeTaskPool::get_or_init(TaskPool::default),
        layer,
        |bz, out| {
            for z in bz * BRICK..((bz + 1) * BRICK).min(dims[2]) {
                for y in 0..dims[1] {
                    let row = &data[(z * dims[1] + y) * dims[0]..][..dims[0]];
                    let out = &mut out[y / BRICK * bricks[0]..][..bricks[0]];
                    for (range, texels) in out.iter_mut().zip(row.chunks(BRICK)) {
                        for &v in texels {
                            range[0] = range[0].min(v);
                            range[1] = range[1].max(v);
                        }
                    }
                }
            }
        },
    );
    let size = Extent3d {
        width: bricks[0] as u32,
        height: bricks[1] as u32,
        depth_or_array_layers: bricks[2] as u32,
    };
    Image::new(
        size,
        TextureDimension::D3,
        bytemuck::cast_slice(&ranges).to_vec(),
        TextureFormat::Rg32Float,
        RenderAssetUsages::RENDER_WORLD,
    )
}

/// iDaVIE's 80-row colour map atlas (Assets/Textures/allmaps.png): sRGB, point filtered, clamped.
pub fn color_map_image() -> Image {
    Image::from_buffer(
        include_bytes!("../assets/allmaps.png"),
        ImageType::Extension("png"),
        CompressedImageFormats::NONE,
        true,
        ImageSampler::nearest(),
        RenderAssetUsages::RENDER_WORLD,
    )
    .expect("embedded colour map decodes")
}
