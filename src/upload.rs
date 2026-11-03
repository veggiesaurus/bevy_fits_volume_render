//! Fills volume textures a slab of z planes per frame. Bevy uploads an image in one go, which stalls a frame for
//! a big texture and holds the bytes twice (the image and a staging copy of all of it).

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use bevy::{
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::{
            Extent3d, Origin3d, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
        },
        renderer::RenderQueue,
        texture::GpuImage,
    },
};

/// Bytes written per frame: a few milliseconds of upload on an M2.
const SLAB_BYTES: usize = 32 << 20;

#[derive(Clone)]
struct Upload {
    image: AssetId<Image>,
    dims: [usize; 3],
    /// None until the data source has sent them.
    voxels: Option<Arc<Vec<f32>>>,
    planes_written: Arc<AtomicUsize>,
}

impl Upload {
    fn finished(&self) -> bool {
        self.planes_written.load(Ordering::Relaxed) == self.dims[2]
    }
}

/// R32Float textures created empty and still to be filled, oldest first.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct Uploads(Vec<Upload>);

impl Uploads {
    /// `image` counts as filling from now, though its voxels come later through `fill`.
    pub fn expect(&mut self, image: AssetId<Image>, dims: [usize; 3]) {
        self.0.push(Upload {
            image,
            dims,
            voxels: None,
            planes_written: Arc::default(),
        });
    }

    /// Dropped if the image went before its voxels arrived.
    pub fn fill(&mut self, image: AssetId<Image>, voxels: Vec<f32>) {
        if let Some(upload) = self.0.iter_mut().find(|upload| upload.image == image) {
            debug_assert_eq!(voxels.len(), upload.dims.iter().product::<usize>());
            upload.voxels = Some(Arc::new(voxels));
        }
    }

    pub fn is_filling(&self, image: AssetId<Image>) -> bool {
        self.0.iter().any(|upload| upload.image == image)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

pub struct UploadPlugin;

impl Plugin for UploadPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Uploads>()
            .add_plugins(ExtractResourcePlugin::<Uploads>::default())
            .add_systems(Last, retire_uploads);
        app.sub_app_mut(RenderApp)
            .add_systems(Render, write_slabs.in_set(RenderSystems::PrepareResources));
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
        let (Some(gpu_image), Some(voxels)) = (gpu_images.get(upload.image), &upload.voxels) else {
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
                    x: 0,
                    y: 0,
                    z: start as u32,
                },
                aspect: TextureAspect::All,
            },
            &bytemuck::cast_slice(voxels)[start * plane..end * plane],
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
