//! The local data source as state without threads. It holds the cube, answers at once what needs no work and hands
//! out the rest as jobs, run wherever its owner likes: on the viewer's task pools natively, one at a time in the
//! browser's Web Worker. A mode's coarse levels are built a level per job the first time a cubelet of them is asked
//! for, in place of the other mode's, and until they're built the coarsest is cut from level 0. Processed cubelets,
//! and a texture's noise and mask, come from the pyramid's level once it's built and otherwise from level 0, after a
//! job has measured the level's normaliser when they need one.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt,
    sync::{Arc, OnceLock, RwLock, Weak},
};

use crate::{
    CUBELET, Cube, Cubelet, Level, LevelOf, Noise, Normaliser, Processing, Region, Shown, cubelet,
    downsampled_dims,
};

/// Which pyramid: max for Maximum, mean for Average.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    Max,
    Mean,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CubeletId {
    pub mode: Mode,
    pub processing: Processing,
    pub level: u8,
    pub x: u16,
    pub y: u16,
    pub z: u16,
}

impl Mode {
    /// A level's noise over level 0's: 1/√N for a mean level whose voxels each average N, as for independent noise,
    /// and 1 for a max level, which is no quieter. A tolerance set in units of the noise scales the same way.
    pub fn noise_ratio(self, factors: [usize; 3]) -> f64 {
        match self {
            Mode::Max => 1.0,
            Mode::Mean => 1.0 / (factors.iter().product::<usize>() as f64).sqrt(),
        }
    }
}

impl CubeletId {
    /// Level 0 is the same in both pyramids, so it always goes as Max.
    pub fn new(mode: Mode, processing: Processing, level: usize, at: [usize; 3]) -> Self {
        Self {
            mode: if level == 0 { Mode::Max } else { mode },
            processing,
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

/// Cubelets in the order they were asked for, each once, so that asking again for one still to come adds nothing.
#[derive(Debug, Default)]
pub struct Queue {
    order: VecDeque<CubeletId>,
    queued: HashSet<CubeletId>,
}

impl Queue {
    /// Whether `id` wasn't queued already.
    pub fn push_back(&mut self, id: CubeletId) -> bool {
        let new = self.queued.insert(id);
        if new {
            self.order.push_back(id);
        }
        new
    }

    pub fn pop_front(&mut self) -> Option<CubeletId> {
        self.remove(0)
    }

    fn remove(&mut self, at: usize) -> Option<CubeletId> {
        let id = self.order.remove(at)?;
        self.queued.remove(&id);
        Some(id)
    }

    pub fn remove_all(&mut self, ids: &HashSet<CubeletId>) {
        self.order.retain(|id| !ids.contains(id));
        self.queued.retain(|id| !ids.contains(id));
    }

    pub fn clear(&mut self) {
        self.order.clear();
        self.queued.clear();
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
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
    /// The noise of `texels` of a level as shown.
    Measure {
        id: u32,
        mode: Mode,
        processing: Processing,
        level: u8,
        texels: Region,
    },
    /// The mask over `texels` of a level as shown, in place of one still to come.
    Mask {
        id: u32,
        mode: Mode,
        processing: Processing,
        level: u8,
        texels: Region,
        threshold: f32,
    },
}

#[derive(Debug)]
pub enum Event {
    Cubelet {
        id: CubeletId,
        cubelet: Cubelet,
    },
    PixelValue {
        id: u32,
        value: f32,
    },
    Measured {
        id: u32,
        noise: Option<Noise>,
    },
    /// A mask of `voxels` texels is made, after its parts.
    Mask {
        id: u32,
        voxels: usize,
    },
    /// A cubelet's share of a mask, a byte a texel, at `origin` in the texels it's over.
    MaskPart {
        id: u32,
        origin: [usize; 3],
        dims: [usize; 3],
        mask: Vec<u8>,
    },
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
    queue: Queue,
    extras: VecDeque<Extra>,
    normalisers: HashMap<(Mode, u8), NormaliserCell>,
}

/// A mode's levels after level 0, finest first, each built by a job of its own from the one before, so that what
/// comes in between is taken and a build that's no longer wanted stops at the next level. Shared with the job building
/// the next level, and once they're all built with the cuts from them.
struct Pyramid {
    mode: Mode,
    levels: Arc<RwLock<Vec<Level>>>,
}

/// A measure or, with a threshold, a mask.
#[derive(Debug)]
struct Extra {
    id: u32,
    mode: Mode,
    processing: Processing,
    level: u8,
    texels: Region,
    threshold: Option<f32>,
}

/// A level's normaliser once its job has measured it, None for a level with no noise to go by.
type NormaliserCell = Arc<OnceLock<Option<Normaliser>>>;

enum Need {
    Ready(Option<NormaliserCell>),
    Build(Job),
    Wait,
}

/// A pyramid's levels after level 0, once they're all built.
type Built = Option<Arc<RwLock<Vec<Level>>>>;

/// Work for any thread, which holds what it needs of the cube.
pub struct Job(Work);

enum Work {
    Cut {
        id: CubeletId,
        cube: Arc<Cube>,
        factors: [usize; 3],
        built: Built,
        normaliser: Option<NormaliserCell>,
    },
    Level {
        mode: Mode,
        cube: Arc<Cube>,
        factors: [usize; 3],
        levels: Arc<RwLock<Vec<Level>>>,
    },
    Normaliser {
        mode: Mode,
        level: u8,
        cube: Arc<Cube>,
        factors: [usize; 3],
        built: Built,
        done: NormaliserCell,
    },
    Extra {
        extra: Extra,
        cube: Arc<Cube>,
        factors: [usize; 3],
        built: Built,
        normaliser: Option<NormaliserCell>,
    },
}

/// `f` of `level` in `built`, if it's there, while it's read.
fn with_built<R>(built: &Built, level: u8, f: impl FnOnce(Option<&Level>) -> R) -> R {
    let levels = built.as_ref().map(|levels| levels.read().unwrap());
    f(levels
        .as_ref()
        .map(|levels| &levels[usize::from(level) - 1]))
}

/// A level as shown, normalised with its normaliser where asked, and read from `built` when it's that.
fn shown<'a>(
    cube: &'a Cube,
    factors: [usize; 3],
    mode: Mode,
    processing: Processing,
    normaliser: &'a Option<NormaliserCell>,
    built: Option<&'a Level>,
) -> Shown<'a> {
    let held = built.map(|level| &level.data[..]);
    Shown {
        level: LevelOf::new(cube, factors, mode == Mode::Max, held),
        processing,
        normaliser: normaliser.as_ref().and_then(|n| n.get()?.as_ref()),
    }
}

/// A mask's share of each cubelet of the level that `texels` take in, from `mask` over `grown` around them, and
/// then that it's made, last so that its parts are in by then.
fn mask_events(id: u32, mask: &[u8], grown: Region, texels: Region) -> Vec<Event> {
    let dims = grown.extent();
    let (lo, hi) = (
        texels.min.map(|t| t / CUBELET),
        texels.max.map(|t| t / CUBELET),
    );
    let (mut events, mut voxels) = (Vec::new(), 0);
    for z in lo[2]..=hi[2] {
        for y in lo[1]..=hi[1] {
            for x in lo[0]..=hi[0] {
                let at = [x, y, z];
                let part = Region {
                    min: std::array::from_fn(|i| (at[i] * CUBELET).max(texels.min[i])),
                    max: std::array::from_fn(|i| ((at[i] + 1) * CUBELET - 1).min(texels.max[i])),
                };
                let mut bytes = Vec::with_capacity(part.extent().iter().product());
                for z in part.min[2]..=part.max[2] {
                    for y in part.min[1]..=part.max[1] {
                        let row = ((z - grown.min[2]) * dims[1] + y - grown.min[1]) * dims[0];
                        let start = row + part.min[0] - grown.min[0];
                        bytes.extend_from_slice(&mask[start..=start + part.max[0] - part.min[0]]);
                    }
                }
                voxels += bytes.iter().filter(|&&m| m != 0).count();
                events.push(Event::MaskPart {
                    id,
                    origin: std::array::from_fn(|i| part.min[i] - texels.min[i]),
                    dims: part.extent(),
                    mask: bytes,
                });
            }
        }
    }
    events.push(Event::Mask { id, voxels });
    events
}

impl Job {
    /// The cubelet it cuts, so that it can be dropped when that's no longer wanted.
    pub fn cubelet(&self) -> Option<CubeletId> {
        match self.0 {
            Work::Cut { id, .. } => Some(id),
            _ => None,
        }
    }

    /// What it has for the viewer. A level job's level goes into its pyramid, and the next can only be handed out once
    /// the job is dropped; a normaliser job's normaliser is kept for what waits on it.
    pub fn run(&self) -> Vec<Event> {
        match &self.0 {
            Work::Cut {
                id,
                cube,
                factors,
                built,
                normaliser,
            } => {
                let cubelet = with_built(built, id.level, |level| match level {
                    _ if !id.processing.is_none() => {
                        let shown =
                            shown(cube, *factors, id.mode, id.processing, normaliser, level);
                        shown.cubelet(id.at())
                    }
                    Some(level) => level.cubelet(id.at()),
                    None => cubelet(cube, *factors, id.at(), id.mode == Mode::Max),
                });
                let cubelet = cubelet.map(|cubelet| Event::Cubelet { id: *id, cubelet });
                cubelet.into_iter().collect()
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
                Vec::new()
            }
            Work::Normaliser {
                mode,
                level,
                cube,
                factors,
                built,
                done,
            } => {
                let normaliser = with_built(built, *level, |level| {
                    let held = level.map(|level| &level.data[..]);
                    Normaliser::measure(&LevelOf::new(cube, *factors, *mode == Mode::Max, held))
                });
                let _ = done.set(normaliser);
                Vec::new()
            }
            Work::Extra {
                extra,
                cube,
                factors,
                built,
                normaliser,
            } => with_built(built, extra.level, |level| {
                let shown = shown(
                    cube,
                    *factors,
                    extra.mode,
                    extra.processing,
                    normaliser,
                    level,
                );
                let id = extra.id;
                match extra.threshold {
                    Some(threshold) => {
                        let (mask, grown) = shown.mask(extra.texels, threshold);
                        mask_events(id, &mask, grown, extra.texels)
                    }
                    None => vec![Event::Measured {
                        id,
                        noise: shown.measure(extra.texels),
                    }],
                }
            }),
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
            Work::Normaliser { mode, factors, .. } => {
                write!(f, "{mode:?} normaliser of {factors:?}")
            }
            Work::Extra { extra, factors, .. } => {
                let what = if extra.threshold.is_some() {
                    "mask"
                } else {
                    "noise"
                };
                let over = extra.texels.extent();
                write!(f, "{:?} {what} of {factors:?} over {over:?}", extra.mode)
            }
        }
    }
}

impl Engine {
    /// A cube of its own, or one it shares, as a server's sessions do.
    pub fn new(cube: impl Into<Arc<Cube>>) -> Self {
        Self {
            cube: cube.into(),
            levels: Vec::new(),
            chain: 0,
            pyramid: None,
            retired: Weak::new(),
            queue: Queue::default(),
            extras: VecDeque::new(),
            normalisers: HashMap::new(),
        }
    }

    /// Takes a request, returning the answer to one that needs no work.
    pub fn request(&mut self, request: Request) -> Option<Event> {
        match request {
            Request::Levels { factors, chain } => {
                (self.levels, self.chain) = (factors, chain);
                self.queue.clear();
                self.extras.clear();
                self.normalisers.clear();
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
                self.queue.remove_all(&ids.into_iter().collect());
            }
            Request::PixelValue { id, voxel } => {
                let [width, height, depth] = self.cube.info.dims;
                if voxel[0] < width && voxel[1] < height && voxel[2] < depth {
                    let value = self.cube.data[voxel[0] + width * (voxel[1] + height * voxel[2])];
                    return Some(Event::PixelValue { id, value });
                }
            }
            Request::Measure {
                id,
                mode,
                processing,
                level,
                texels,
            } => {
                self.extras.retain(|extra| extra.threshold.is_some());
                self.add_extra(Extra {
                    id,
                    mode,
                    processing,
                    level,
                    texels,
                    threshold: None,
                });
            }
            Request::Mask {
                id,
                mode,
                processing,
                level,
                texels,
                threshold,
            } => {
                self.extras.retain(|extra| extra.threshold.is_none());
                self.add_extra(Extra {
                    id,
                    mode,
                    processing,
                    level,
                    texels,
                    threshold: Some(threshold),
                });
            }
        }
        None
    }

    fn add_extra(&mut self, extra: Extra) {
        let level = self.levels.get(usize::from(extra.level));
        let inside = level.is_some_and(|&f| {
            let dims = downsampled_dims(Region::whole(self.cube.info.dims), f);
            (0..3).all(|i| {
                extra.texels.min[i] <= extra.texels.max[i] && extra.texels.max[i] < dims[i]
            })
        });
        if inside {
            self.extras.push_back(extra);
        }
    }

    /// The next job: a measure or a mask, which the view waits on, or else the first cubelet that can be cut now,
    /// in the order asked; or the job one of them waits on, the next level of a pyramid or a normaliser.
    pub fn next_job(&mut self) -> Option<Job> {
        self.next(true)
    }

    /// `next_job`, but never a cut, for when there's nowhere to put another cubelet yet.
    pub fn next_job_but_a_cut(&mut self) -> Option<Job> {
        self.next(false)
    }

    fn next(&mut self, cuts: bool) -> Option<Job> {
        for i in 0..self.extras.len() {
            let Extra {
                mode,
                processing,
                level,
                ..
            } = self.extras[i];
            match self.normaliser(mode, level, processing) {
                Need::Ready(normaliser) => {
                    let extra = self.extras.remove(i).unwrap();
                    return Some(Job(Work::Extra {
                        extra,
                        cube: self.cube.clone(),
                        factors: self.levels[usize::from(level)],
                        built: self.built(mode, level),
                        normaliser,
                    }));
                }
                Need::Build(job) => return Some(job),
                Need::Wait => {}
            }
        }
        for i in 0..self.queue.len() {
            let id = self.queue.order[i];
            let factors = self.levels[usize::from(id.level)];
            if !id.processing.is_none() {
                match self.normaliser(id.mode, id.level, id.processing) {
                    Need::Ready(_) if !cuts => continue,
                    Need::Ready(normaliser) => {
                        self.queue.remove(i);
                        return Some(Job(Work::Cut {
                            id,
                            cube: self.cube.clone(),
                            factors,
                            built: self.built(id.mode, id.level),
                            normaliser,
                        }));
                    }
                    Need::Build(job) => return Some(job),
                    Need::Wait => continue,
                }
            }
            let built = self.built(id.mode, id.level);
            if built.is_some() || !self.needs_pyramid(id) {
                if !cuts {
                    continue;
                }
                self.queue.remove(i);
                return Some(Job(Work::Cut {
                    id,
                    cube: self.cube.clone(),
                    factors,
                    built,
                    normaliser: None,
                }));
            }
            if let Some(job) = self.next_level(id.mode) {
                return Some(job);
            }
        }
        None
    }

    /// The normaliser of `mode`'s `level`, when `processing` needs one: ready, or else the job measuring it unless
    /// that's out already.
    fn normaliser(&mut self, mode: Mode, level: u8, processing: Processing) -> Need {
        if !processing.normalise {
            return Need::Ready(None);
        }
        let mode = if level == 0 { Mode::Max } else { mode };
        let cell = self.normalisers.entry((mode, level)).or_default().clone();
        if cell.get().is_some() {
            return Need::Ready(Some(cell));
        }
        // The map's and this one: any more is the job measuring it.
        if Arc::strong_count(&cell) > 2 {
            return Need::Wait;
        }
        Need::Build(Job(Work::Normaliser {
            mode,
            level,
            cube: self.cube.clone(),
            factors: self.levels[usize::from(level)],
            built: self.built(mode, level),
            done: cell,
        }))
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

    /// Whether `id` can only be cut from its mode's pyramid: an unprocessed chain level between level 0 and the
    /// coarsest.
    fn needs_pyramid(&self, id: CubeletId) -> bool {
        id.processing.is_none() && id.level != 0 && usize::from(id.level) + 1 < self.chain
    }

    /// The levels `mode`'s chain `level` after level 0 is read from, once they're all built and they're its mode's.
    fn built(&self, mode: Mode, level: u8) -> Built {
        let level = usize::from(level);
        (self.pyramid.as_ref())
            .filter(|p| p.mode == mode && level != 0 && level < self.chain)
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
            let ids = vec![CubeletId::new(mode, Processing::default(), level, at)];
            engine.request(Request::AddCubelets { ids: ids.clone() });
            let mut arrived = Vec::new();
            while let Some(job) = engine.next_job() {
                for event in job.run() {
                    if let Event::Cubelet { id, .. } = event {
                        arrived.push(id);
                    }
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
            ids: vec![CubeletId::new(mode, Processing::default(), 1, [0; 3])],
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

    #[test]
    fn with_no_room_for_a_cubelet_the_level_cuts_wait_on_still_starts() {
        let dims = [256, 256, 4];
        let mut engine = Engine::new(cube(dims, vec![0.0; 256 * 256 * 4]));
        engine.request(Request::Levels {
            factors: levels(dims, 384, 2048).chain,
            chain: 3,
        });
        let ids =
            [0, 1].map(|level| CubeletId::new(Mode::Max, Processing::default(), level, [0; 3]));
        engine.request(Request::AddCubelets { ids: ids.to_vec() });
        let level = engine.next_job_but_a_cut().unwrap();
        assert!(level.cubelet().is_none() && engine.next_job_but_a_cut().is_none());
        assert_eq!(engine.next_job().unwrap().cubelet(), Some(ids[0]));
    }

    #[test]
    fn a_cubelet_asked_for_again_before_its_cut_is_cut_once() {
        let mut engine = Engine::new(cube([130, 4, 1], vec![0.0; 130 * 4]));
        engine.request(Request::Levels {
            factors: vec![[1; 3]],
            chain: 1,
        });
        let id = CubeletId::new(Mode::Max, Processing::default(), 0, [1, 0, 0]);
        for _ in 0..3 {
            engine.request(Request::AddCubelets { ids: vec![id, id] });
        }
        let mut cut = 0;
        while let Some(job) = engine.next_job() {
            cut += job.run().len();
        }
        assert_eq!(cut, 1);
        // Once it's out, it can be asked for again.
        engine.request(Request::AddCubelets { ids: vec![id] });
        assert!(engine.next_job().is_some());
    }

    #[test]
    fn normalised_work_waits_for_the_levels_normaliser_and_a_mask_comes_in_parts() {
        let dims = [100, 70, 8];
        let data = (0..100 * 70 * 8)
            .map(|i| ((i * 7919) % 101) as f32)
            .collect();
        let mut engine = Engine::new(cube(dims, data));
        engine.request(Request::Levels {
            factors: vec![[1; 3], [2, 2, 1]],
            chain: 2,
        });
        let processing = Processing {
            normalise: true,
            smoothing: crate::Smoothing::Spectral,
        };
        let id = CubeletId::new(Mode::Mean, processing, 1, [0; 3]);
        engine.request(Request::AddCubelets { ids: vec![id] });
        let texels = Region {
            min: [10, 5, 0],
            max: [49, 34, 7],
        };
        engine.request(Request::Mask {
            id: 3,
            mode: Mode::Mean,
            processing,
            level: 1,
            texels,
            threshold: 5.0,
        });

        // The normaliser first, which both wait on until it's measured.
        let normaliser = engine.next_job().unwrap();
        assert!(normaliser.cubelet().is_none() && engine.next_job().is_none());
        assert!(normaliser.run().is_empty());
        let mut events = Vec::new();
        while let Some(job) = engine.next_job() {
            events.extend(job.run());
        }
        let parts: usize = (events.iter())
            .filter_map(|event| match event {
                Event::MaskPart { dims, mask, .. } => {
                    assert_eq!(mask.len(), dims.iter().product());
                    Some(mask.len())
                }
                _ => None,
            })
            .sum();
        let made = events
            .iter()
            .position(|e| matches!(e, Event::Mask { id: 3, .. }));
        assert_eq!(
            made,
            events
                .iter()
                .rposition(|e| matches!(e, Event::MaskPart { .. }))
                .map(|i| i + 1)
        );
        assert_eq!(parts, texels.extent().iter().product());
        assert!((events.iter()).any(|e| matches!(e, Event::Cubelet { id: got, .. } if *got == id)));
    }
}
