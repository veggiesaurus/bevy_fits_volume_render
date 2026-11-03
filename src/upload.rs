//! Fills volume textures: cubelets written at their offsets, a slab of z planes per frame, and a finer texture's
//! placeholder upsampled from a coarser one by a compute pass. Bevy uploads an image in one go, which stalls a frame
//! for a big texture and holds the bytes twice (the image and a staging copy of all of it).

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use bevy::{
    core_pipeline::schedule::camera_driver,
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::{
            BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, Extent3d,
            Origin3d, PipelineCache, ShaderStages, ShaderType, StorageTextureAccess,
            TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect, TextureFormat,
            TextureSampleType, UniformBuffer,
            binding_types::{texture_2d_array, texture_storage_2d_array, uniform_buffer},
        },
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue},
        texture::GpuImage,
    },
};

/// Bytes written per frame: a few milliseconds of upload on an M2.
const SLAB_BYTES: usize = 32 << 20;
const UPSAMPLE_SHADER: &str = "embedded://bevy_fits_volume_render/upsample.wgsl";

#[derive(Clone)]
struct Upload {
    image: AssetId<Image>,
    origin: [usize; 3],
    dims: [usize; 3],
    voxels: Arc<Vec<f32>>,
    /// Held back until set: a placeholder the voxels would otherwise be written under.
    after: Option<Arc<AtomicBool>>,
    planes_written: Arc<AtomicUsize>,
}

impl Upload {
    fn finished(&self) -> bool {
        self.planes_written.load(Ordering::Relaxed) == self.dims[2]
    }
}

/// Voxels still to be written into R32Float textures, oldest first.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct Uploads(Vec<Upload>);

impl Uploads {
    /// Writes `dims` voxels into `image` from `origin`, once `after` is set if there is one.
    pub fn push_at(
        &mut self,
        image: AssetId<Image>,
        origin: [usize; 3],
        dims: [usize; 3],
        voxels: Vec<f32>,
        after: Option<Arc<AtomicBool>>,
    ) {
        debug_assert_eq!(voxels.len(), dims.iter().product::<usize>());
        self.0.push(Upload {
            image,
            origin,
            dims,
            voxels: Arc::new(voxels),
            after,
            planes_written: Arc::default(),
        });
    }

    pub fn is_filling(&self, image: AssetId<Image>) -> bool {
        self.0.iter().any(|upload| upload.image == image)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Fills `dst` with the nearest texel of `src` once both are on the GPU, then sets `done`. Texel `i` of `dst` takes
/// `floor(i * scale + offset)` of `src`, clamped to it.
#[derive(Clone)]
pub struct Upsample {
    pub src: Handle<Image>,
    pub dst: AssetId<Image>,
    pub scale: Vec3,
    pub offset: Vec3,
    pub src_dims: [usize; 3],
    pub dst_dims: [usize; 3],
    pub done: Arc<AtomicBool>,
}

#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct Upsamples(pub Vec<Upsample>);

pub struct UploadPlugin;

impl Plugin for UploadPlugin {
    fn build(&self, app: &mut App) {
        bevy::asset::embedded_asset!(app, "upsample.wgsl");
        app.init_resource::<Uploads>()
            .init_resource::<Upsamples>()
            .add_plugins((
                ExtractResourcePlugin::<Uploads>::default(),
                ExtractResourcePlugin::<Upsamples>::default(),
            ))
            .add_systems(Last, retire_uploads);
        app.sub_app_mut(RenderApp)
            .add_systems(RenderStartup, init_upsample)
            .add_systems(Render, write_slabs.in_set(RenderSystems::PrepareResources))
            .add_systems(RenderGraph, upsample.before(camera_driver));
    }
}

/// Drops finished uploads, and those whose image is gone (a crop reset or replaced before it arrived).
pub fn retire_uploads(mut uploads: ResMut<Uploads>, images: Res<Assets<Image>>) {
    let done = |upload: &Upload| upload.finished() || !images.contains(upload.image);
    if uploads.0.iter().any(done) {
        uploads.0.retain(|upload| !done(upload));
    }
}

fn write_slabs(
    uploads: Res<Uploads>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    queue: Res<RenderQueue>,
) {
    let mut budget = SLAB_BYTES;
    for upload in &uploads.0 {
        let held = (upload.after.as_ref()).is_some_and(|after| !after.load(Ordering::Relaxed));
        let Some(gpu_image) = gpu_images.get(upload.image).filter(|_| !held) else {
            continue;
        };
        let [width, height, depth] = upload.dims;
        let plane = width * height * 4;
        let start = upload.planes_written.load(Ordering::Relaxed);
        let end = (start + (budget / plane).max(1)).min(depth);
        if start == end {
            continue;
        }
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &gpu_image.texture,
                mip_level: 0,
                origin: Origin3d {
                    x: upload.origin[0] as u32,
                    y: upload.origin[1] as u32,
                    z: (upload.origin[2] + start) as u32,
                },
                aspect: TextureAspect::All,
            },
            &bytemuck::cast_slice(&upload.voxels)[start * plane..end * plane],
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width as u32 * 4),
                rows_per_image: Some(height as u32),
            },
            Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: (end - start) as u32,
            },
        );
        upload.planes_written.store(end, Ordering::Relaxed);
        budget = budget.saturating_sub((end - start) * plane);
        if budget == 0 {
            break;
        }
    }
}

/// Field order must match `Params` in upsample.wgsl.
#[derive(ShaderType)]
struct UpsampleParams {
    scale: Vec3,
    offset: Vec3,
    src_max: UVec3,
    dims: UVec3,
}

#[derive(Resource)]
struct UpsamplePipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

fn init_upsample(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "upsample",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d_array(TextureSampleType::Float { filterable: false }),
                texture_storage_2d_array(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<UpsampleParams>(false),
            ),
        ),
    );
    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("upsample".into()),
        layout: vec![layout.clone()],
        shader: asset_server.load(UPSAMPLE_SHADER),
        entry_point: Some("upsample".into()),
        ..default()
    });
    commands.insert_resource(UpsamplePipeline { layout, pipeline });
}

/// In the frame's encoder, so the voxels uploaded before it are in `src`, and those held back for `dst` land on top
/// from the next frame on.
fn upsample(
    mut context: RenderContext,
    jobs: Res<Upsamples>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    pipeline: Res<UpsamplePipeline>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let Some(compute) = cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };
    for job in jobs
        .0
        .iter()
        .filter(|job| !job.done.load(Ordering::Relaxed))
    {
        let (Some(src), Some(dst)) = (gpu_images.get(&job.src), gpu_images.get(job.dst)) else {
            continue;
        };
        let to_u32 = |d: [usize; 3]| UVec3::from_array(d.map(|d| d as u32));
        let mut params = UniformBuffer::from(UpsampleParams {
            scale: job.scale,
            offset: job.offset,
            src_max: to_u32(job.src_dims) - 1,
            dims: to_u32(job.dst_dims),
        });
        params.write_buffer(&device, &queue);
        let bind_group = device.create_bind_group(
            "upsample",
            &cache.get_bind_group_layout(&pipeline.layout),
            &BindGroupEntries::sequential((&src.texture_view, &dst.texture_view, &params)),
        );
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(compute);
        pass.set_bind_group(0, &bind_group, &[]);
        let [x, y, z] = to_u32(job.dst_dims).to_array();
        pass.dispatch_workgroups(x.div_ceil(8), y.div_ceil(8), z.div_ceil(4));
        job.done.store(true, Ordering::Relaxed);
    }
}
