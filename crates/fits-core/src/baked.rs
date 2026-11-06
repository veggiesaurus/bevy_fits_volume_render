//! A baked cube: the cubelets of a cube's max and mean pyramids, compressed once, in one file for static hosting, so
//! that a viewer reads only the cubelets it shows, by range request. The file is an index of what's in it, then a
//! pack's records, the same as the client keeps on disk. `Source` is the viewer's side, which reads through the
//! client's cache, and worker.js makes the range requests it asks for.

use std::collections::{HashMap, HashSet};

use crate::{
    CUBELET, Cubelet, FileInfo, Voxels,
    cache::{self, Kept, Lru, Memory, RECORD, Store},
    codec,
    engine::{CubeletId, Event, Mode, Request},
    wire::{self, Reader, Wire, Writer},
};

/// The first thing in a baked file, which goes up when its layout changes.
const FORMAT: &str = "fits-cubelets 1";

/// What a baked file holds, before its records.
#[derive(Debug)]
pub struct Index {
    pub info: FileInfo,
    /// The viewer's default budget's, which a viewer with another takes as they are.
    pub levels: Vec<[usize; 3]>,
    /// What the cubelets are compressed to within, in level 0's units, as a fits-server's are; None when exact.
    pub tolerance: Option<f64>,
    /// Whether level 0 is exact even so, so that cursor values from it are.
    pub exact_level_0: bool,
    /// What the cube is, for the client's cache.
    pub content_id: String,
    /// Each cubelet's record, where it starts in the file and how long it is.
    pub cubelets: Vec<(CubeletId, u64, u32)>,
}

impl Wire for Index {
    fn write(&self, w: &mut Writer) {
        FORMAT.to_string().write(w);
        self.info.write(w);
        self.levels.write(w);
        self.tolerance.write(w);
        self.exact_level_0.write(w);
        self.content_id.write(w);
        (self.cubelets.len() as u32).write(w);
        for (id, at, len) in &self.cubelets {
            id.write(w);
            at.write(w);
            len.write(w);
        }
    }

    fn read(r: &mut Reader) -> Option<Self> {
        if String::read(r)? != FORMAT {
            return None;
        }
        let (info, levels, tolerance) = (Wire::read(r)?, Wire::read(r)?, Wire::read(r)?);
        let (exact_level_0, content_id) = (Wire::read(r)?, Wire::read(r)?);
        let n = u32::read(r)?;
        let mut cubelets = Vec::new();
        for _ in 0..n {
            cubelets.push((Wire::read(r)?, Wire::read(r)?, Wire::read(r)?));
        }
        Some(Self {
            info,
            levels,
            tolerance,
            exact_level_0,
            content_id,
            cubelets,
        })
    }
}

/// The longest an index can be. One holds a histogram of up to 65535 bins, a header's cards and every cubelet's place,
/// which comes to about a fifth of a megabyte for each example cube.
pub const MAX_INDEX: usize = 4 << 20;

/// The longest a record can be: an exact cubelet, or a compressed one and its NaN runs, after its header.
const MAX_RECORD: u32 = 4 << 20;

impl Index {
    /// The index at the start of `bytes`, from a file of `size` bytes when that's known. A baked file can be anyone's,
    /// so one that doesn't add up is an error: a cube or header no FITS file has, levels unlike the viewer's, or
    /// records that aren't each in place, one after the other, to the end of the file.
    pub fn read(bytes: &[u8], size: Option<u64>) -> Result<Self, String> {
        let len = (bytes.first_chunk().map(|&head| wire::header_len(head)))
            .filter(|&len| len <= MAX_INDEX.min(bytes.len()))
            .ok_or("not a baked cube, or not all of its index")?;
        let index: Self = wire::decode(&bytes[..len], |_| None)
            .map_err(|_| "not a baked cube of this version")?;
        index.check(len as u64, size)?;
        Ok(index)
    }

    fn check(&self, start: u64, size: Option<u64>) -> Result<(), String> {
        let info = &self.info;
        let dims = info.dims;
        let voxels = dims.iter().try_fold(1u64, |n, &d| n.checked_mul(d as u64));
        let most = usize::from(u16::MAX) * CUBELET;
        let Some(voxels) =
            voxels.filter(|&n| n <= 1 << 40 && dims.iter().all(|&d| (1..=most).contains(&d)))
        else {
            return Err(format!("a cube of {dims:?}"));
        };
        // Cards are read a byte column at a time, so they're checked before anything reads them.
        if info.cards.iter().any(|c| c.len() != 80 || !c.is_ascii()) {
            return Err("a header no FITS file has".into());
        }
        let naxis: usize = crate::num(&info.cards, "NAXIS").ok().flatten().unwrap_or(3);
        if info.axes.iter().any(|&a| a >= 999) || naxis > 999 {
            return Err("a header no FITS file has".into());
        }
        let bins = &info.histogram.0;
        if bins.is_empty() || bins.len() > crate::MAX_BINS || bins.windows(2).any(|w| w[0] > w[1]) {
            return Err("a histogram that isn't one".into());
        }
        let s = info.stats;
        let nothing_finite = s.non_finite as u64 == voxels;
        if !nothing_finite && !(s.min.is_finite() && s.max.is_finite() && s.min <= s.max) {
            return Err("statistics that aren't any".into());
        }
        let sigma = info.sigma.is_none_or(|s| s.is_finite() && s > 0.0);
        if !sigma || self.tolerance.is_some_and(|t| !(t.is_finite() && t >= 0.0)) {
            return Err("a noise or tolerance that isn't one".into());
        }
        // The overview's factors go up together on x and y, so on a thin axis they can be more than its length.
        let chain = default_levels(dims).chain;
        let longest = dims.into_iter().max().unwrap_or(1);
        let overview = self.levels.get(chain.len());
        let overview = overview.is_none_or(|f| f.iter().all(|f| (1..=longest).contains(f)));
        if !self.levels.starts_with(&chain) || self.levels.len() > chain.len() + 1 || !overview {
            return Err("levels unlike the viewer's".into());
        }
        let grids: Vec<_> = (self.levels.iter())
            .map(|&f| {
                crate::downsampled_dims(crate::Region::whole(dims), f).map(|d| d.div_ceil(CUBELET))
            })
            .collect();
        let mut end = start;
        for &(id, at, len) in &self.cubelets {
            let grid = grids.get(usize::from(id.level));
            let placed =
                grid.is_some_and(|g| (0..3).all(|i| id.at()[i] < g[i])) && id.processing.is_none();
            if !placed || at != end || !(RECORD as u32 + 1..=MAX_RECORD).contains(&len) {
                return Err(format!("the record for {id:?} isn't where it could be"));
            }
            end = end
                .checked_add(len.into())
                .ok_or("records past the end of any file")?;
        }
        if size.is_some_and(|size| size != end) {
            return Err("a file that doesn't end where its index says".into());
        }
        Ok(())
    }
}

/// The levels a cube is baked at: the viewer's at its default budget, which a viewer with another takes as they are.
fn default_levels(dims: [usize; 3]) -> crate::Levels {
    crate::levels(dims, 384, 2048)
}

/// `cube`'s file: every cubelet of its max and mean pyramids at the viewer's default budget, within `sigmas` of its
/// noise, or exact without a noise, and level 0 exact if asked. The cubelets are the engine's, as a fits-server's are.
#[cfg(not(target_arch = "wasm32"))]
pub fn bake(
    cube: crate::Cube,
    content_id: String,
    sigmas: f64,
    exact_level_0: bool,
) -> Result<Vec<u8>, String> {
    use std::sync::Mutex;

    use prost::Message;

    use crate::engine::Engine;

    let dims = cube.info.dims;
    let levels = default_levels(dims);
    let chain = levels.chain.len();
    let factors = levels.list().0;
    let tolerance = cube.info.sigma.map(|sigma| sigmas * sigma);
    let info = cube.info.clone();
    let mut engine = Engine::new(cube);
    engine.request(Request::Levels {
        factors: factors.clone(),
        chain,
    });
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let records = Mutex::new(Vec::new());
    let mut asked = 0;
    // A mode at a time, since the engine builds one pyramid in place of the other's, and level 0 once.
    for mode in [Mode::Max, Mode::Mean] {
        let mut ids = Vec::new();
        for (level, &f) in factors
            .iter()
            .enumerate()
            .skip(usize::from(mode == Mode::Mean))
        {
            let grid =
                crate::downsampled_dims(crate::Region::whole(dims), f).map(|d| d.div_ceil(CUBELET));
            for z in 0..grid[2] {
                for y in 0..grid[1] {
                    for x in 0..grid[0] {
                        ids.push(CubeletId::new(mode, Default::default(), level, [x, y, z]));
                    }
                }
            }
        }
        asked += ids.len();
        engine.request(Request::AddCubelets { ids });
        loop {
            let jobs: Vec<_> = std::iter::from_fn(|| engine.next_job())
                .take(4 * threads)
                .collect();
            if jobs.is_empty() {
                break;
            }
            let jobs = Mutex::new(jobs);
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        while let Some(job) = jobs.lock().unwrap().pop() {
                            let events = job.run();
                            drop(job);
                            for event in events {
                                let Event::Cubelet { id, cubelet } = event else {
                                    continue;
                                };
                                let exact = exact_level_0 && id.level == 0;
                                let f = factors[usize::from(id.level)];
                                let within = tolerance.filter(|_| !exact);
                                let payload = codec::encode(
                                    &cubelet,
                                    within.map(|t| t * id.mode.noise_ratio(f)),
                                );
                                let record = cache::record(
                                    id,
                                    within.unwrap_or(0.0),
                                    &payload.encode_to_vec(),
                                );
                                records.lock().unwrap().push((id, record));
                            }
                        }
                    });
                }
            });
        }
    }
    let mut records = records.into_inner().unwrap();
    if records.len() != asked {
        return Err(format!("{} of {asked} cubelets were cut", records.len()));
    }
    // In a fixed order, so that the same cube always bakes to the same file.
    records.sort_by_key(|(id, _)| (id.mode == Mode::Mean, id.level, id.z, id.y, id.x));
    // What's in the file, as well as what it's of, so that a cube baked again differently is a different cube.
    let baked = cache::fnv(
        &records
            .iter()
            .flat_map(|(_, record)| record)
            .copied()
            .collect::<Vec<_>>(),
    );
    let mut index = Index {
        info,
        levels: factors,
        tolerance,
        exact_level_0,
        content_id: format!("{content_id}:{baked:016x}"),
        cubelets: Vec::new(),
    };
    // The index's length doesn't depend on the offsets in it, so the records start after one with none yet.
    let mut at = 0;
    for (id, record) in &records {
        index.cubelets.push((*id, at, record.len() as u32));
        at += record.len() as u64;
    }
    let start = wire::header(&index).len() as u64;
    if start as usize > MAX_INDEX {
        return Err(format!(
            "an index of {start} bytes, more than a viewer reads"
        ));
    }
    index
        .cubelets
        .iter_mut()
        .for_each(|(_, at, _)| *at += start);
    let mut bytes = wire::header(&index);
    records.iter().for_each(|(_, record)| bytes.extend(record));
    Ok(bytes)
}

/// A range of the baked file to read for the source, to be handed back to `Source::fetched` with its token.
#[derive(Debug)]
pub struct Fetch {
    pub token: u32,
    pub start: u64,
    pub end: u64,
}

/// A cubelet being read, and whether the viewer wants it, or only the cursor.
struct Read {
    id: CubeletId,
    shown: bool,
}

/// A baked cube as the viewer's data source, whose cubelets come from the cache or else by range request. Cursor
/// values come from level 0's cubelets, so they're within the bake's tolerance unless level 0 is exact.
pub struct Source {
    records: HashMap<CubeletId, (u64, u32)>,
    kept: Kept,
    memory: Memory,
    reads: HashMap<u32, Read>,
    /// The cursor's latest value, while its cubelet is being read.
    value: Option<(u32, [usize; 3])>,
    /// The level 0 cubelet the cursor was last over, decoded.
    last: Option<(CubeletId, Cubelet)>,
    next_token: u32,
    fetches: Vec<Fetch>,
}

impl Source {
    /// The cube that `index` is of, from the file at `url`, which names its pack in the cache.
    pub fn new(index: &Index, url: &str) -> Self {
        let identity = cache::identity(&cache::origin(url), &index.content_id, &index.levels);
        Self {
            records: (index.cubelets.iter())
                .map(|&(id, at, len)| (id, (at, len)))
                .collect(),
            kept: Kept::new(identity, index.tolerance.unwrap_or(0.0)),
            memory: Lru::new(0),
            reads: HashMap::new(),
            value: None,
            last: None,
            next_token: 0,
            fetches: Vec::new(),
        }
    }

    /// The name of the cube's pack in the cache.
    pub fn pack(&self) -> &str {
        &self.kept.name
    }

    /// The pack, or None when it can't be kept, in which case cubelets are kept in memory instead.
    // ponytail: the browser's default cache_memory_mb, rather than the page's config, which the worker doesn't have.
    pub fn attach(&mut self, store: Option<(Box<dyn Store>, u64)>) {
        self.memory = Lru::new(if store.is_some() { 0 } else { 256 << 20 });
        self.kept.attach(store);
    }

    /// The answer to `request` when it needs no range read. The reads it does need go to `fetches`.
    pub fn request(&mut self, request: Request) -> Option<Event> {
        match request {
            Request::AddCubelets { ids } => {
                for id in ids {
                    if !self.kept.hit(&mut self.memory, id) {
                        self.read(id, true);
                    }
                }
                None
            }
            Request::RemoveCubelets { ids } => {
                let removed: HashSet<_> = ids.iter().collect();
                self.kept.hits.retain(|id| !removed.contains(id));
                for read in self
                    .reads
                    .values_mut()
                    .filter(|read| removed.contains(&read.id))
                {
                    read.shown = false;
                }
                None
            }
            Request::PixelValue { id, voxel } => {
                let cubelet = at(voxel);
                if !self.last.as_ref().is_some_and(|(last, _)| *last == cubelet) {
                    self.last =
                        (self.kept.read(&mut self.memory, cubelet).ok()).map(|c| (cubelet, c));
                }
                match &self.last {
                    Some((_, cubelet)) => Some(Event::PixelValue {
                        id,
                        value: value(cubelet, voxel),
                    }),
                    None => {
                        self.value = Some((id, voxel));
                        self.read(cubelet, false);
                        None
                    }
                }
            }
            // The viewer takes the bake's levels, and measures nothing of a cube it doesn't hold.
            Request::Levels { .. } | Request::Mask { .. } => None,
            Request::Measure { id, .. } => Some(Event::Measured { id, noise: None }),
        }
    }

    /// Reads `id`, unless it's being read already.
    fn read(&mut self, id: CubeletId, shown: bool) {
        if let Some(read) = self.reads.values_mut().find(|read| read.id == id) {
            read.shown |= shown;
            return;
        }
        // A cubelet the bake doesn't have, such as a processed one, never comes.
        let Some(&(start, len)) = self.records.get(&id) else {
            return;
        };
        self.next_token = self.next_token.wrapping_add(1);
        let token = self.next_token;
        self.reads.insert(token, Read { id, shown });
        let end = start + u64::from(len);
        self.fetches.push(Fetch { token, start, end });
    }

    /// The range reads asked for since this was last called.
    pub fn fetches(&mut self) -> Vec<Fetch> {
        std::mem::take(&mut self.fetches)
    }

    /// Whether the read for `token` is still wanted, by the viewer or the cursor. One that isn't is forgotten, as it
    /// won't be made.
    pub fn wanted(&mut self, token: u32) -> bool {
        let cursor = self.value.map(|(_, voxel)| at(voxel));
        let wanted =
            (self.reads.get(&token)).is_some_and(|read| read.shown || Some(read.id) == cursor);
        if !wanted {
            self.reads.remove(&token);
        }
        wanted
    }

    /// The read for `token` failed, and the viewer won't have its cubelet unless it asks again.
    pub fn failed(&mut self, token: u32) {
        self.reads.remove(&token);
    }

    /// What the read for `token` brought, which is kept even if it's no longer wanted. An error is a file that doesn't
    /// match its index.
    pub fn fetched(&mut self, token: u32, bytes: &[u8]) -> Result<Vec<Event>, String> {
        let Some(read) = self.reads.remove(&token) else {
            return Ok(Vec::new());
        };
        let (head, payload) = bytes
            .split_first_chunk::<RECORD>()
            .ok_or("a short record")?;
        let (id, within, len) = cache::read_record(head);
        if id != read.id || len != payload.len() {
            return Err(format!(
                "the record for {:?} isn't where the index says",
                read.id
            ));
        }
        let cubelet = codec::decode_payload(payload)?;
        self.kept.keep(&mut self.memory, id, within, payload);
        let mut events = Vec::new();
        if let Some((value_id, voxel)) = self.value.filter(|&(_, voxel)| at(voxel) == id) {
            self.value = None;
            let value = value(&cubelet, voxel);
            events.push(Event::PixelValue {
                id: value_id,
                value,
            });
        }
        if read.shown {
            events.push(Event::Cubelet { id, cubelet });
        } else {
            self.last = Some((id, cubelet));
        }
        Ok(events)
    }

    /// Whether `cached` has cubelets to decode.
    pub fn has_cached(&self) -> bool {
        !self.kept.hits.is_empty()
    }

    /// Empties the cache of the cube's cubelets, in memory and in its pack.
    pub fn clear(&mut self) {
        self.memory.clear();
        self.kept.clear();
        self.last = None;
    }

    /// The next of the cubelets asked for that are in the cache, decoded. One that has gone or doesn't decode is read
    /// again, in `fetches`.
    pub fn cached(&mut self) -> Option<Event> {
        while let Some((id, cubelet)) = self.kept.next(&mut self.memory) {
            match cubelet {
                Ok(cubelet) => return Some(Event::Cubelet { id, cubelet }),
                Err(_) => self.read(id, true),
            }
        }
        None
    }
}

/// The level 0 cubelet `voxel` is in.
fn at(voxel: [usize; 3]) -> CubeletId {
    CubeletId::new(Mode::Max, Default::default(), 0, voxel.map(|v| v / CUBELET))
}

fn value(cubelet: &Cubelet, voxel: [usize; 3]) -> f32 {
    let [x, y, z] = voxel.map(|v| v % CUBELET);
    let [nx, ny, _] = cubelet.dims;
    match &cubelet.voxels {
        Voxels::AllNan => f32::NAN,
        Voxels::Constant(v) => *v,
        Voxels::Data(v) => v.get(x + nx * (y + ny * z)).copied().unwrap_or(f32::NAN),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cube, read_fits};

    /// A float32 cube whose voxels are `value` of their place, as read from a FITS file.
    fn cube(dims: [usize; 3], value: impl Fn([usize; 3]) -> f32) -> Cube {
        let cards = [
            ("SIMPLE", "T".to_string()),
            ("BITPIX", "-32".into()),
            ("NAXIS", "3".into()),
        ];
        let axes = (0..3).map(|i| (["NAXIS1", "NAXIS2", "NAXIS3"][i], dims[i].to_string()));
        let mut bytes: Vec<u8> = (cards.into_iter().chain(axes))
            .flat_map(|(key, value)| format!("{key:<8}= {value:>20}{:50}", "").into_bytes())
            .chain(format!("{:<80}", "END").into_bytes())
            .collect();
        bytes.resize(2880, b' ');
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    bytes.extend(value([x, y, z]).to_be_bytes());
                }
            }
        }
        bytes.resize(bytes.len().div_ceil(2880) * 2880, 0);
        read_fits(&bytes[..], None).unwrap()
    }

    #[test]
    fn a_baked_cube_comes_through_its_index_and_then_from_the_cache() {
        let dims = [150, 70, 80];
        let value = |[x, y, z]: [usize; 3]| match (x, y) {
            (..10, ..20) => f32::NAN,
            _ => ((x * 7 + y * 13 + z * 29) % 97) as f32 * 0.1 - 2.0,
        };
        let local = cube(dims, value);
        let pack =
            std::env::temp_dir().join(format!("fits-core-baked-test-{}", std::process::id()));
        for exact_level_0 in [false, true] {
            let file = bake(cube(dims, value), "cube".into(), 1.0, exact_level_0).unwrap();
            let index = Index::read(&file, Some(file.len() as u64)).unwrap();
            // A baked file can be anyone's, so one that doesn't add up is refused, not trusted.
            let header = wire::header_len(*file.first_chunk().unwrap());
            let chain = default_levels(dims).chain.len();
            let refused = |change: &dyn Fn(&mut Index), why: &str| {
                let mut index = Index::read(&file, None).unwrap();
                change(&mut index);
                let mut bytes = wire::header(&index);
                bytes.extend_from_slice(&file[header..]);
                let e = Index::read(&bytes, Some(bytes.len() as u64)).unwrap_err();
                assert!(e.contains(why), "{why}: {e}");
            };
            let overview = |f: [usize; 3]| {
                move |i: &mut Index| {
                    i.levels.truncate(chain);
                    i.levels.push(f);
                }
            };
            refused(&overview([0, 1, 1]), "levels");
            refused(&overview([151, 1, 1]), "levels");
            refused(&|i| i.info.cards[0].push('x'), "header");
            refused(&|i| i.info.stats.min = f32::NAN, "statistics");
            refused(&|i| i.info.dims = [0, 70, 80], "cube");
            refused(&|i| i.info.dims = [usize::MAX, 1, 1], "cube");
            refused(&|i| i.info.axes = [usize::MAX, 1, 2], "header");
            refused(&|i| i.info.histogram.0.clear(), "histogram");
            refused(&|i| i.tolerance = Some(f64::NAN), "tolerance");
            refused(&|i| i.cubelets[1].1 += 1, "record");
            refused(&|i| i.cubelets[0].2 = u32::MAX, "record");
            refused(&|i| i.cubelets[0].0.level = 99, "record");
            assert!(Index::read(&file, Some(file.len() as u64 - 1)).is_err());
            // An index longer than any there is, or than the bytes it came in, isn't decoded at all.
            let mut long = file.clone();
            long[..4].copy_from_slice(&u32::MAX.to_le_bytes());
            assert!(Index::read(&long, None).is_err());
            assert!(Index::read(&file[..header - 1], None).is_err());
            assert_eq!(index.levels, crate::levels(dims, 384, 2048).list().0);
            let tolerance = index.tolerance.unwrap();
            // Range reads, as worker.js makes them, and what they bring.
            let read = |source: &mut Source| -> Vec<Event> {
                (source.fetches().into_iter())
                    .flat_map(|f| {
                        source
                            .fetched(f.token, &file[f.start as usize..f.end as usize])
                            .unwrap()
                    })
                    .collect()
            };
            let open = || {
                let mut source = Source::new(&index, "https://host/cubes/cube.fits.cubelets");
                let file = (std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true))
                .truncate(false)
                .open(&pack)
                .unwrap();
                source.attach(Some((Box::new(file), u64::MAX)));
                source
            };
            let _ = std::fs::remove_file(&pack);
            let ids = vec![
                CubeletId::new(Mode::Max, Default::default(), 0, [1, 0, 0]),
                CubeletId::new(Mode::Mean, Default::default(), 1, [0, 0, 0]),
            ];
            let mut source = open();
            assert!(
                source
                    .request(Request::AddCubelets { ids: ids.clone() })
                    .is_none()
            );
            let mut arrived = read(&mut source);
            // A cursor value comes from level 0's cubelet, which is exact when level 0 is.
            // That's read once, and from the cache after.
            let voxel = [20, 30, 40];
            let ask = Request::PixelValue { id: 3, voxel };
            assert!(source.request(ask).is_none());
            let Some(Event::PixelValue { id: 3, value: got }) = read(&mut source).pop() else {
                panic!("no value")
            };
            let again = source.request(Request::PixelValue { id: 4, voxel });
            assert!(matches!(again, Some(Event::PixelValue { id: 4, value }) if value == got));
            let off = (got - value(voxel)).abs();
            assert!(
                if exact_level_0 {
                    off == 0.0
                } else {
                    off <= tolerance as f32
                },
                "{off}"
            );

            // Opened again, it's all in the cache, with nothing to read.
            let mut source = open();
            source.request(Request::AddCubelets { ids: ids.clone() });
            assert!(source.fetches().is_empty());
            arrived.extend(std::iter::from_fn(|| source.cached()));
            assert_eq!(arrived.len(), 4);
            for event in arrived {
                let Event::Cubelet { id, cubelet } = event else {
                    panic!("{event:?}")
                };
                let factors = index.levels[usize::from(id.level)];
                let want = crate::cubelet(&local, factors, id.at(), id.mode == Mode::Max).unwrap();
                let within = if exact_level_0 && id.level == 0 {
                    0.0
                } else {
                    tolerance * id.mode.noise_ratio(factors)
                };
                let got = match cubelet.voxels {
                    Voxels::Data(v) => v,
                    v => panic!("{v:?}"),
                };
                let Voxels::Data(want) = want.voxels else {
                    panic!()
                };
                for (g, w) in got.iter().zip(&want) {
                    assert!(
                        if w.is_nan() {
                            g.is_nan()
                        } else {
                            (g - w).abs() <= within as f32
                        },
                        "{id:?}"
                    );
                }
            }
        }
        std::fs::remove_file(&pack).unwrap();
    }
}
