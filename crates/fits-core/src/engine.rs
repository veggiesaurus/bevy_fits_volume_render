//! The local data source as state without threads. It holds the cube, answers at once what needs no work and hands
//! out the rest as jobs, run wherever its owner likes: on the viewer's task pools natively, one at a time in the
//! browser's Web Worker. A mode's coarse levels are built a level per job the first time a cubelet of them is asked
//! for, in place of the other mode's, and until they're built the coarsest is cut from level 0.

use std::{
    collections::{HashSet, VecDeque},
    fmt,
    sync::{Arc, RwLock, Weak},
};

use crate::{Cube, Cubelet, Level, cubelet};

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

#[derive(Debug)]
pub enum Request {
    /// The level list the viewer asks by, the first `chain` of them the power-of-two chain, in place of any before
    /// along with what was asked for by them.
    Levels {
        factors: Vec<[usize; 3]>,
        chain: usize,
    },
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

#[derive(Debug)]
pub enum Event {
    Cubelet { id: CubeletId, cubelet: Cubelet },
    PixelValue { id: u32, value: f32 },
}

pub struct Engine {
    cube: Arc<Cube>,
    /// The chain's factors, then the overview's when it isn't one of them.
    levels: Vec<[usize; 3]>,
    chain: usize,
    /// One mode's at most, a third of the cube or so.
    pyramid: Option<Pyramid>,
    /// The pyramid before while a job still has it, which the next level waits for so that the two are never in
    /// memory together.
    retired: Weak<RwLock<Vec<Level>>>,
    queue: VecDeque<CubeletId>,
}

/// A mode's levels after level 0, finest first, each built by a job of its own from the one before, so that what
/// comes in between is taken and a build that's no longer wanted stops at the next level. Shared with the job building
/// the next level, and once they're all built with the cuts from them.
struct Pyramid {
    mode: Mode,
    levels: Arc<RwLock<Vec<Level>>>,
}

/// Work for any thread, which holds what it needs of the cube.
pub struct Job(Work);

enum Work {
    Cut {
        id: CubeletId,
        cube: Arc<Cube>,
        factors: [usize; 3],
        built: Option<Arc<RwLock<Vec<Level>>>>,
    },
    Level {
        mode: Mode,
        cube: Arc<Cube>,
        factors: [usize; 3],
        levels: Arc<RwLock<Vec<Level>>>,
    },
}

impl Job {
    /// The cubelet it cuts, so that it can be dropped when that's no longer wanted.
    pub fn cubelet(&self) -> Option<CubeletId> {
        match self.0 {
            Work::Cut { id, .. } => Some(id),
            Work::Level { .. } => None,
        }
    }

    /// What it has for the viewer, if anything. A level job's level goes into its pyramid, and the next can only be
    /// handed out once the job is dropped.
    pub fn run(&self) -> Option<Event> {
        match &self.0 {
            Work::Cut {
                id,
                cube,
                factors,
                built,
            } => {
                let cubelet = match built {
                    Some(levels) => {
                        levels.read().unwrap()[usize::from(id.level) - 1].cubelet(id.at())
                    }
                    None => cubelet(cube, *factors, id.at(), id.mode == Mode::Max),
                };
                cubelet.map(|cubelet| Event::Cubelet { id: *id, cubelet })
            }
            Work::Level {
                mode,
                cube,
                factors,
                levels,
            } => {
                let level = match levels.read().unwrap().last() {
                    Some(finer) => finer.chain(*factors),
                    None => Level::new(cube, *factors, *mode == Mode::Max),
                };
                let mut levels = levels.write().unwrap();
                if let Some(finer) = levels.last_mut() {
                    finer.counts = Vec::new();
                }
                levels.push(level);
                None
            }
        }
    }
}

/// For a log line, when a job takes a while.
impl fmt::Display for Job {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.0 {
            Work::Cut { id, factors, .. } => {
                write!(f, "{:?} cubelet {:?} of {factors:?}", id.mode, id.at())
            }
            Work::Level { mode, factors, .. } => write!(f, "{mode:?} level {factors:?}"),
        }
    }
}

impl Engine {
    pub fn new(cube: Cube) -> Self {
        Self {
            cube: Arc::new(cube),
            levels: Vec::new(),
            chain: 0,
            pyramid: None,
            retired: Weak::new(),
            queue: VecDeque::new(),
        }
    }

    /// Takes a request, returning the answer to one that needs no work.
    pub fn request(&mut self, request: Request) -> Option<Event> {
        match request {
            Request::Levels { factors, chain } => {
                (self.levels, self.chain) = (factors, chain);
                self.queue.clear();
                self.retire();
            }
            Request::AddCubelets { ids } => {
                let levels = self.levels.len();
                let ids = ids.into_iter().filter(|id| usize::from(id.level) < levels);
                for id in ids {
                    if self.needs_pyramid(id) {
                        self.build(id.mode);
                    }
                    self.queue.push_back(id);
                }
            }
            Request::RemoveCubelets { ids } => {
                let ids: HashSet<CubeletId> = ids.into_iter().collect();
                self.queue.retain(|id| !ids.contains(id));
            }
            Request::PixelValue { id, voxel } => {
                let [width, height, depth] = self.cube.info.dims;
                if voxel[0] < width && voxel[1] < height && voxel[2] < depth {
                    let value = self.cube.data[voxel[0] + width * (voxel[1] + height * voxel[2])];
                    return Some(Event::PixelValue { id, value });
                }
            }
        }
        None
    }

    /// The next job, in the order asked: the first cubelet that can be cut now, or the next level of the pyramid
    /// that one before it waits on.
    pub fn next_job(&mut self) -> Option<Job> {
        for i in 0..self.queue.len() {
            let id = self.queue[i];
            let built = self.built(id);
            if built.is_some() || !self.needs_pyramid(id) {
                self.queue.remove(i);
                return Some(Job(Work::Cut {
                    id,
                    cube: self.cube.clone(),
                    factors: self.levels[usize::from(id.level)],
                    built,
                }));
            }
            if let Some(job) = self.next_level(id.mode) {
                return Some(job);
            }
        }
        None
    }

    /// The job building `mode`'s next level, unless one is out already or a job still has the pyramid before.
    fn next_level(&self, mode: Mode) -> Option<Job> {
        let pyramid = self.pyramid.as_ref().filter(|p| p.mode == mode)?;
        // Until they're all built, a job holding the levels is the one building the next of them.
        if Arc::strong_count(&pyramid.levels) > 1 || self.retired.strong_count() > 0 {
            return None;
        }
        let built = pyramid.levels.read().unwrap().len();
        Some(Job(Work::Level {
            mode,
            cube: self.cube.clone(),
            factors: *self.levels[1..self.chain].get(built)?,
            levels: pyramid.levels.clone(),
        }))
    }

    /// Whether `id` can only be cut from its mode's pyramid: a chain level between level 0 and the coarsest.
    fn needs_pyramid(&self, id: CubeletId) -> bool {
        id.level != 0 && usize::from(id.level) + 1 < self.chain
    }

    /// The levels a chain level after level 0 is cut from, once they're all built and they're its mode's.
    fn built(&self, id: CubeletId) -> Option<Arc<RwLock<Vec<Level>>>> {
        let level = usize::from(id.level);
        (self.pyramid.as_ref())
            .filter(|p| p.mode == id.mode && level != 0 && level < self.chain)
            .filter(|p| p.levels.read().unwrap().len() + 1 == self.chain)
            .map(|p| p.levels.clone())
    }

    /// `mode`'s coarse levels, in place of the other mode's.
    fn build(&mut self, mode: Mode) {
        if self.pyramid.as_ref().is_some_and(|p| p.mode == mode) {
            return;
        }
        self.retire();
        self.pyramid = Some(Pyramid {
            mode,
            levels: Arc::default(),
        });
    }

    /// Drops the pyramid, which is kept track of while a job still has it. Only one pyramid at a time can have a job
    /// out, since the next level waits for the one before's.
    fn retire(&mut self) {
        if let Some(old) = self.pyramid.take()
            && Arc::strong_count(&old.levels) > 1
        {
            self.retired = Arc::downgrade(&old.levels);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{levels, tests::cube};

    #[test]
    fn one_pyramid_at_a_time_built_a_level_per_job_when_first_needed() {
        let dims = [256, 256, 4];
        let data = (0..256 * 256 * 4).map(|i| i as f32).collect();
        let mut engine = Engine::new(cube(dims, data));
        let chain = levels(dims, 384, 2048).chain;
        assert_eq!(chain, [[1; 3], [2, 2, 1], [4, 4, 1]]);
        engine.request(Request::Levels {
            factors: chain,
            chain: 3,
        });
        let value = engine.request(Request::PixelValue {
            id: 7,
            voxel: [3, 2, 1],
        });
        assert!(
            matches!(value, Some(Event::PixelValue { id: 7, value }) if value == 3.0 + 256.0 * (2.0 + 256.0))
        );

        let mut ask = |mode, level, at| {
            let ids = vec![CubeletId::new(mode, level, at)];
            engine.request(Request::AddCubelets { ids: ids.clone() });
            let mut arrived = Vec::new();
            while let Some(job) = engine.next_job() {
                if let Some(Event::Cubelet { id, .. }) = job.run() {
                    arrived.push(id);
                }
            }
            assert_eq!(arrived, ids);
            engine.pyramid.as_ref().map(|p| p.mode)
        };
        // Level 0 and the coarsest come straight from the cube.
        assert_eq!(ask(Mode::Mean, 2, [0; 3]), None);
        assert_eq!(ask(Mode::Mean, 0, [3, 3, 0]), None);
        // A level between them waits on its mode's pyramid, which replaces the other mode's.
        assert_eq!(ask(Mode::Max, 1, [1, 0, 0]), Some(Mode::Max));
        assert_eq!(ask(Mode::Mean, 1, [1, 1, 0]), Some(Mode::Mean));

        // A level at a time, and the other mode's first waits for the job still building one of this mode's.
        let add = |mode| Request::AddCubelets {
            ids: vec![CubeletId::new(mode, 1, [0; 3])],
        };
        engine.request(add(Mode::Max));
        let max_level = engine.next_job().unwrap();
        assert!(max_level.cubelet().is_none());
        assert!(engine.next_job().is_none());
        engine.request(add(Mode::Mean));
        assert!(engine.next_job().is_none());
        drop(max_level);
        assert!(engine.next_job().unwrap().cubelet().is_none());

        // Levels for a file with fewer of them drop what was asked for by the ones before.
        engine.request(Request::Levels {
            factors: vec![[1; 3]],
            chain: 1,
        });
        assert!(engine.next_job().is_none());
    }
}
