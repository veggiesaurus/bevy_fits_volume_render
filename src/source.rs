//! The viewer's only way to the voxels. Requests go out and their events come back on a later frame, so nothing
//! else reads the cube. The local source holds the cube, reads a pixel value at once and cuts cubelets on the async
//! compute pool: from level 0, from the overview level, or from one mode's coarse levels, built when a cubelet of them
//! is first asked for and dropped for the other mode's. Until they're built the coarsest is cut from level 0. It wakes
//! the app when there's something to collect.

use std::{
    collections::{HashSet, VecDeque},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

use bevy::{
    platform::time::Instant,
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};

use crate::{
    fits::{self, Cube, Cubelet, Level},
    wake,
};

/// Cuts at a time, so a cubelet asked for first is cut first and the browser's single thread gets frames between.
const IN_FLIGHT: usize = 32;

/// Which pyramid: max for Maximum, mean for Average.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    Max,
    Mean,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CubeletId {
    pub mode: Mode,
    pub level: u8,
    pub x: u16,
    pub y: u16,
    pub z: u16,
}

impl CubeletId {
    /// Level 0 is the same in both pyramids, so it always goes as Max.
    pub fn new(mode: Mode, level: usize, at: [usize; 3]) -> Self {
        Self {
            mode: if level == 0 { Mode::Max } else { mode },
            level: level as u8,
            x: at[0] as u16,
            y: at[1] as u16,
            z: at[2] as u16,
        }
    }

    pub fn at(&self) -> [usize; 3] {
        [self.x, self.y, self.z].map(usize::from)
    }
}

pub enum Request {
    /// In the order wanted.
    AddCubelets {
        ids: Vec<CubeletId>,
    },
    RemoveCubelets {
        ids: Vec<CubeletId>,
    },
    PixelValue {
        id: u32,
        voxel: [usize; 3],
    },
}

pub enum Event {
    Cubelet { id: CubeletId, cubelet: Cubelet },
    PixelValue { id: u32, value: f32 },
}

pub struct DataSource {
    cube: Arc<Cube>,
    state: Mutex<State>,
    answered: Arc<Mutex<Vec<Event>>>,
}

#[derive(Default)]
struct State {
    /// The chain's factors, then the overview's when it isn't one of them.
    levels: Vec<[usize; 3]>,
    chain: usize,
    /// One mode's at most, a third of the cube or so.
    pyramid: Option<Pyramid>,
    queue: VecDeque<CubeletId>,
    /// Dropping a task cancels it, if it hasn't started.
    cutting: Vec<(CubeletId, Task<()>)>,
}

impl State {
    /// Whether `id` can only be cut from its mode's pyramid: a chain level between level 0 and the coarsest.
    fn needs_pyramid(&self, id: CubeletId) -> bool {
        id.level != 0 && usize::from(id.level) + 1 < self.chain
    }
}

/// A mode's levels after level 0, set once built. Dropping it stops the build at the next level.
struct Pyramid {
    mode: Mode,
    levels: Arc<OnceLock<Vec<Level>>>,
    /// Until it's finished and polled.
    build: Option<Task<()>>,
    cancel: Arc<AtomicBool>,
}

impl Drop for Pyramid {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl DataSource {
    pub fn local(cube: Cube) -> Self {
        Self {
            cube: Arc::new(cube),
            state: default(),
            answered: default(),
        }
    }

    /// The level list the viewer asks by, the first `chain` of them the power-of-two chain.
    pub fn open(&self, levels: Vec<[usize; 3]>, chain: usize) {
        let mut state = self.state.lock().unwrap();
        (state.levels, state.chain) = (levels, chain);
    }

    pub fn send(&self, request: Request) {
        let mut state = self.state.lock().unwrap();
        match request {
            Request::AddCubelets { ids } => {
                if let Some(id) = ids.iter().find(|id| state.needs_pyramid(**id)) {
                    self.build(&mut state, id.mode);
                }
                state.queue.extend(ids);
                self.pump(&mut state);
            }
            Request::RemoveCubelets { ids } => {
                let ids: HashSet<CubeletId> = ids.into_iter().collect();
                state.queue.retain(|id| !ids.contains(id));
                state.cutting.retain(|(id, _)| !ids.contains(id));
            }
            Request::PixelValue { id, voxel } => {
                let [width, height, _] = self.cube.info.dims;
                let value = self.cube.data[voxel[0] + width * (voxel[1] + height * voxel[2])];
                self.answered
                    .lock()
                    .unwrap()
                    .push(Event::PixelValue { id, value });
                wake();
            }
        }
    }

    /// The events that have arrived since the last call.
    pub fn drain(&self) -> Vec<Event> {
        self.pump(&mut self.state.lock().unwrap());
        std::mem::take(&mut *self.answered.lock().unwrap())
    }

    /// Starts cutting what can be cut, in the order asked. Finished tasks are polled first, so that a panic in one
    /// comes out here.
    fn pump(&self, state: &mut State) {
        state
            .cutting
            .retain_mut(|(_, task)| block_on(poll_once(task)).is_none());
        let pyramid = state.pyramid.as_mut();
        if let Some(build) = pyramid.and_then(|p| p.build.take_if(|build| build.is_finished())) {
            block_on(build);
        }
        let mut i = 0;
        while i < state.queue.len() && state.cutting.len() < IN_FLIGHT {
            let id = state.queue[i];
            let level = usize::from(id.level);
            let built = (state.pyramid.as_ref())
                .filter(|p| p.mode == id.mode && level != 0 && level < state.chain)
                .filter(|p| p.levels.get().is_some())
                .map(|p| p.levels.clone());
            if built.is_none() && state.needs_pyramid(id) {
                i += 1;
                continue;
            }
            state.queue.remove(i);
            let (cube, answered) = (self.cube.clone(), self.answered.clone());
            let factors = state.levels[level];
            let task = AsyncComputeTaskPool::get().spawn(async move {
                let cubelet = match &built {
                    Some(levels) => levels.get().unwrap()[level - 1].cubelet(id.at()),
                    None => fits::cubelet(&cube, factors, id.at(), id.mode == Mode::Max),
                };
                if let Some(cubelet) = cubelet {
                    answered
                        .lock()
                        .unwrap()
                        .push(Event::Cubelet { id, cubelet });
                    wake();
                }
            });
            state.cutting.push((id, task));
        }
    }

    /// The coarse levels of `mode`'s pyramid in the background, in place of the other mode's, which goes first so
    /// that the two are never in memory together.
    fn build(&self, state: &mut State, mode: Mode) {
        if state.pyramid.as_ref().is_some_and(|p| p.mode == mode) {
            return;
        }
        state.pyramid = None;
        let (levels, cancel) = (Arc::<OnceLock<_>>::default(), Arc::<AtomicBool>::default());
        let (cube, factors) = (self.cube.clone(), state.levels[1..state.chain].to_vec());
        let (built, cancelled) = (levels.clone(), cancel.clone());
        let build = AsyncComputeTaskPool::get().spawn(async move {
            let start = Instant::now();
            if let Some(levels) = Level::pyramid(&cube, &factors, mode == Mode::Max, &cancelled) {
                info!(
                    "{mode:?} levels {factors:?} in {:.2} s",
                    start.elapsed().as_secs_f32()
                );
                let _ = built.set(levels);
                wake();
            }
        });
        state.pyramid = Some(Pyramid {
            mode,
            levels,
            build: Some(build),
            cancel,
        });
    }
}

#[cfg(test)]
mod tests {
    use bevy::tasks::TaskPool;

    use super::*;

    /// The ids of the next `n` cubelets to arrive.
    fn arrivals(source: &DataSource, n: usize) -> Vec<CubeletId> {
        let start = Instant::now();
        let mut ids = Vec::new();
        while ids.len() < n {
            assert!(start.elapsed().as_secs() < 10, "{ids:?} of {n}");
            ids.extend(source.drain().into_iter().filter_map(|event| match event {
                Event::Cubelet { id, .. } => Some(id),
                Event::PixelValue { .. } => None,
            }));
            std::thread::yield_now();
        }
        ids
    }

    #[test]
    fn one_pyramid_at_a_time_built_when_first_needed() {
        AsyncComputeTaskPool::get_or_init(TaskPool::default);
        let dims = [256, 256, 4];
        let mut bytes: Vec<u8> = ["SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3"]
            .into_iter()
            .zip(["T", "-32", "3", "256", "256", "4"])
            .flat_map(|(key, value)| format!("{key:<8}= {value:>20}{:50}", "").into_bytes())
            .chain(format!("{:<80}", "END").into_bytes())
            .collect();
        bytes.resize(2880, b' ');
        bytes.extend((0..256 * 256 * 4).flat_map(|i| (i as f32).to_be_bytes()));
        bytes.resize(bytes.len().div_ceil(2880) * 2880, 0);
        let source = DataSource::local(fits::read_fits(&bytes[..], None).unwrap());
        let chain = fits::levels(dims, 384, 2048).chain;
        assert_eq!(chain, [[1; 3], [2, 2, 1], [4, 4, 1]]);
        source.open(chain, 3);
        let built = || {
            source
                .state
                .lock()
                .unwrap()
                .pyramid
                .as_ref()
                .map(|p| p.mode)
        };
        let ask = |mode, level, at| {
            let ids = vec![CubeletId::new(mode, level, at)];
            source.send(Request::AddCubelets { ids: ids.clone() });
            assert_eq!(arrivals(&source, 1), ids);
        };

        // Level 0 and the coarsest come straight from the cube.
        ask(Mode::Mean, 2, [0; 3]);
        ask(Mode::Mean, 0, [3, 3, 0]);
        assert_eq!(built(), None);
        // A level between them waits on its mode's pyramid, which replaces the other mode's.
        ask(Mode::Max, 1, [1, 0, 0]);
        assert_eq!(built(), Some(Mode::Max));
        ask(Mode::Mean, 1, [1, 1, 0]);
        assert_eq!(built(), Some(Mode::Mean));
    }
}
