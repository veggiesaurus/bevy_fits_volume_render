use bevy::{
    asset::{RenderAssetUsages, embedded_asset},
    image::{
        CompressedImageFormats, ImageAddressMode, ImageSampler, ImageSamplerDescriptor, ImageType,
    },
    mesh::MeshVertexBufferLayoutRef,
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    render::{
        render_resource::{
            AsBindGroup, CompareFunction, Extent3d, Face, RenderPipelineDescriptor, ShaderType,
            SpecializedMeshPipelineError, TextureDimension, TextureFormat, TextureViewDescriptor,
            TextureViewDimension,
        },
        storage::ShaderBuffer,
    },
    shader::ShaderRef,
};

use crate::{config::COLOR_MAPS, fits};

const SHADER: &str = "embedded://bevy_fits_volume_render/volume.wesl";

pub struct VolumePlugin;

impl Plugin for VolumePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "volume.wesl");
        app.add_plugins(MaterialPlugin::<VolumeMaterial>::default())
            .add_systems(Startup, insert_no_grid);
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
    pub step_length: f32,
    pub step_count: f32,
    pub cell_size: Vec3,
    pub layers: f32,
    pub grid_cells: UVec3,
    pub has_nan: u32,
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
            step_length: 0.0,
            step_count: 0.0,
            cell_size: Vec3::ONE,
            layers: 1.0,
            grid_cells: UVec3::ONE,
            has_nan: 0,
        }
    }
}

impl VolumeParams {
    /// iDaVIE's step, sqrt(max_length) / max_steps, with the step count lifted so the longest rays (sqrt(3) on
    /// the full cube) reach the far side rather than stopping at max_steps. Capped at about two steps per texel
    /// along the texture's diagonal, so a small texture (a crop, or a heavily downsampled cube) costs less.
    /// Then the texture's grid, or `NoGrid`'s one cell over all of it while the grid is still to come.
    pub fn set_texture(&mut self, dims: [usize; 3], grid: Option<&Grid>) {
        let texels = Vec3::from_array(dims.map(|d| d as f32));
        let max_length = (self.slice_max - self.slice_min).length();
        self.step_count =
            (max_length.sqrt() * self.max_steps as f32).min((2.0 * texels.length()).ceil());
        self.step_length = max_length / self.step_count;
        self.layers = texels.z;
        let edge = fits::CUBELET as f32;
        (self.cell_size, self.grid_cells, self.has_nan) = match grid {
            Some(grid) => (
                Vec3::new(edge / texels.x, edge / texels.y, edge),
                UVec3::from_array(dims.map(|d| d.div_ceil(fits::CUBELET) as u32)),
                grid.has_nan as u32,
            ),
            None => (Vec3::new(1.0, 1.0, texels.z), UVec3::ONE, 1),
        };
    }
}

/// A texture's `fits::cell_summary` on the GPU, a cell of 64³ texels to an entry: 1 if it has a NaN, then its max,
/// so a Maximum ray can step over the cell or leave out the NaN test.
#[derive(Clone)]
pub struct Grid {
    pub buffer: Handle<ShaderBuffer>,
    pub has_nan: bool,
}

impl Grid {
    pub fn new(cells: &[fits::Cell], buffers: &mut Assets<ShaderBuffer>) -> Self {
        let entries: Vec<[u32; 2]> = (cells.iter())
            .map(|cell| [cell.has_nan as u32, cell.max.to_bits()])
            .collect();
        Self {
            buffer: buffers.add(ShaderBuffer::new(entries, RenderAssetUsages::RENDER_WORLD)),
            has_nan: cells.iter().any(|cell| cell.has_nan),
        }
    }
}

/// One cell that never skips, with a NaN and a max of +inf, for a texture whose grid is still to come.
#[derive(Resource)]
pub struct NoGrid(pub Handle<ShaderBuffer>);

fn insert_no_grid(mut commands: Commands, mut buffers: ResMut<Assets<ShaderBuffer>>) {
    let never_skips = fits::Cell {
        max: f32::INFINITY,
        has_nan: true,
    };
    commands.insert_resource(NoGrid(Grid::new(&[never_skips], &mut buffers).buffer));
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct VolumeMaterial {
    #[uniform(0)]
    pub params: VolumeParams,
    // R32Float is filterable on some devices and not others, so it's declared unfilterable everywhere.
    #[texture(1, dimension = "2d_array", sample_type = "float", filterable = false)]
    #[sampler(2, sampler_type = "non_filtering")]
    pub data_cube: Handle<Image>,
    #[texture(3)]
    #[sampler(4)]
    pub color_map: Handle<Image>,
    #[storage(5, read_only)]
    pub grid: Handle<ShaderBuffer>,
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

/// An R32Float 2D array texture with a layer per channel and no data yet, which `Uploads` fills. A 2D array
/// sample issues faster than a 3D one.
pub fn empty_cube_image(dims: [usize; 3]) -> Image {
    let size = Extent3d {
        width: dims[0] as u32,
        height: dims[1] as u32,
        depth_or_array_layers: dims[2] as u32,
    };
    let mut image = Image::new_uninit(
        size,
        TextureDimension::D2,
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    // Without this a single channel would be a plain 2D texture.
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    // Point filtering with Repeat wrapping, which is what iDaVIE's `new Texture3D` gets by default; the shader
    // wraps z.
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        ..ImageSamplerDescriptor::nearest()
    });
    image
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
