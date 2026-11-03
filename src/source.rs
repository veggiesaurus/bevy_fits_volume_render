//! The viewer's only way to the voxels. Requests go out and their events come back on a later frame, so nothing
//! else reads the cube. The local source holds the voxels, reads a pixel value at once and builds textures on the
//! async compute pool.

use std::sync::{Arc, Mutex};

use bevy::{
    platform::time::Instant,
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};

use crate::fits::{self, Cell, Region};

pub enum Request {
    /// A region downsampled for `image`, until textures are built from cubelets.
    Texture {
        image: AssetId<Image>,
        region: Region,
        factors: [usize; 3],
        max_mode: bool,
    },
    PixelValue {
        id: u32,
        voxel: [usize; 3],
    },
}

pub enum Event {
    Texture {
        image: AssetId<Image>,
        data: Vec<f32>,
        cells: Vec<Cell>,
    },
    PixelValue {
        id: u32,
        value: f32,
    },
}

pub struct DataSource {
    data: Arc<Vec<f32>>,
    dims: [usize; 3],
    /// Dropping a task cancels it, if it hasn't started.
    textures: Mutex<Vec<(AssetId<Image>, Task<Event>)>>,
    answered: Mutex<Vec<Event>>,
}

impl DataSource {
    pub fn local(data: Vec<f32>, dims: [usize; 3]) -> Self {
        Self {
            data: Arc::new(data),
            dims,
            textures: default(),
            answered: default(),
        }
    }

    pub fn send(&self, request: Request) {
        match request {
            Request::Texture {
                image,
                region,
                factors,
                max_mode,
            } => {
                let (cube, cube_dims) = (self.data.clone(), self.dims);
                let task = AsyncComputeTaskPool::get().spawn(async move {
                    let start = Instant::now();
                    let (dims, data) =
                        fits::crop_and_downsample(&cube, cube_dims, region, factors, max_mode);
                    let cells = fits::cell_summary(&data, dims, fits::CUBELET);
                    info!(
                        "Texture {dims:?} for {region:?}, downsampled by {factors:?}, in {:.2} s",
                        start.elapsed().as_secs_f32()
                    );
                    Event::Texture { image, data, cells }
                });
                self.textures.lock().unwrap().push((image, task));
            }
            Request::PixelValue { id, voxel } => {
                let [width, height, _] = self.dims;
                let value = self.data[voxel[0] + width * (voxel[1] + height * voxel[2])];
                self.answered
                    .lock()
                    .unwrap()
                    .push(Event::PixelValue { id, value });
            }
        }
    }

    /// Drops the request for `image`'s texture.
    pub fn cancel(&self, image: AssetId<Image>) {
        self.textures.lock().unwrap().retain(|(id, _)| *id != image);
    }

    /// The events that have arrived since the last call. A texture task that panicked panics here.
    pub fn drain(&self) -> Vec<Event> {
        let mut events = std::mem::take(&mut *self.answered.lock().unwrap());
        self.textures
            .lock()
            .unwrap()
            .retain_mut(|(_, task)| match block_on(poll_once(task)) {
                Some(event) => {
                    events.push(event);
                    false
                }
                None => true,
            });
        events
    }

    /// Whether any request is still to be answered, so the app keeps updating until it is.
    pub fn busy(&self) -> bool {
        !self.textures.lock().unwrap().is_empty() || !self.answered.lock().unwrap().is_empty()
    }
}
