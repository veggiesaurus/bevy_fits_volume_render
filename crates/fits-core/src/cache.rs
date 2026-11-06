//! Compressed cubelets kept for later: an LRU in memory, which the server and the client both have, and on the
//! client a pack per cube on disk, so that a cube opened again comes from there rather than the server or static
//! hosting. Natively a pack is a file in a folder of them; in the browser it's an OPFS file, which worker.js opens and
//! makes room for. A pack's records are also what a baked cube's file holds.

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    hash::Hash,
    sync::Arc,
};

use crate::{
    CUBELET, Cubelet, codec,
    engine::{CubeletId, Mode},
};

/// Entries under a cap on their bytes, least recently used out first.
pub struct Lru<K, V> {
    entries: HashMap<K, (V, usize, u64)>,
    used: BTreeMap<u64, K>,
    tick: u64,
    bytes: usize,
    cap: usize,
}

impl<K: Hash + Eq + Clone, V> Lru<K, V> {
    pub fn new(cap: usize) -> Self {
        Self {
            entries: HashMap::new(),
            used: BTreeMap::new(),
            tick: 0,
            bytes: 0,
            cap,
        }
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        self.tick += 1;
        let (value, _, used) = self.entries.get_mut(key)?;
        let key = self.used.remove(used).unwrap();
        *used = self.tick;
        self.used.insert(self.tick, key);
        Some(value)
    }

    /// `bytes` is what `value` counts for against the cap, and the entry itself a little more.
    pub fn insert(&mut self, key: K, value: V, bytes: usize) {
        let bytes = bytes + 64;
        self.tick += 1;
        self.bytes += bytes;
        if let Some((_, old, used)) = (self.entries).insert(key.clone(), (value, bytes, self.tick))
        {
            self.bytes -= old;
            self.used.remove(&used);
        }
        self.used.insert(self.tick, key);
        while self.bytes > self.cap
            && let Some((_, key)) = self.used.pop_first()
        {
            let (_, old, _) = self.entries.remove(&key).unwrap();
            self.bytes -= old;
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.used.clear();
        self.bytes = 0;
    }
}

/// The file a pack is kept in.
pub trait Store {
    fn size(&mut self) -> u64;
    fn read_at(&mut self, at: u64, buf: &mut [u8]) -> bool;
    fn write_at(&mut self, at: u64, bytes: &[u8]) -> bool;
    fn truncate(&mut self, size: u64) -> bool;
}

/// A record's header: its payload's length, then the cubelet's mode, level and place, then the tolerance it was
/// compressed to.
pub const RECORD: usize = 4 + 1 + 1 + 6 + 8;

/// One cube's cubelets on disk, for a cube whose server gives it a `content_id`: what the pack is of, then a record
/// per cubelet, each appended as it arrives. A cubelet sent again replaces the one before in the index, and the old record
/// stays where it is.
// ponytail: replaced records are dead space until the pack is evicted or cleared; compact if that grows.
pub struct Pack {
    store: Box<dyn Store>,
    /// Where each cubelet's payload starts, its length, and its tolerance.
    index: HashMap<CubeletId, (u64, usize, f64)>,
    start: u64,
    end: u64,
    /// How long the pack may grow, to keep every pack under the cache's cap.
    room: u64,
}

impl Pack {
    /// The pack in `store` of `identity`, emptied if it's of anything else. Its header is written again either way, so
    /// that the file's modified time says when it was last used. A pack that fails to read or write is read as far as
    /// it can be and left as it is.
    pub fn open(mut store: Box<dyn Store>, identity: &str, room: u64) -> Self {
        let header = [&(identity.len() as u32).to_le_bytes(), identity.as_bytes()].concat();
        let start = header.len() as u64;
        let mut found = vec![0; header.len()];
        let ours = store.read_at(0, &mut found) && found == header;
        let size = if ours { store.size() } else { 0 };
        let mut usable = if ours {
            size >= start
        } else {
            store.truncate(0)
        };
        usable &= store.write_at(0, &header);
        let mut pack = Self {
            store,
            index: HashMap::new(),
            start,
            end: start,
            room,
        };
        let mut head = [0; RECORD];
        while usable && pack.end + RECORD as u64 <= size {
            usable = pack.store.read_at(pack.end, &mut head);
            let (id, tolerance, len) = read_record(&head);
            let payload = pack.end + RECORD as u64;
            if !usable || len == 0 || payload + len as u64 > size {
                break;
            }
            pack.index.insert(id, (payload, len, tolerance));
            pack.end = payload + len as u64;
        }
        // Past the last whole record is what a write cut short left.
        if usable && pack.end < size {
            usable = pack.store.truncate(pack.end);
        }
        if !usable {
            pack.room = 0;
        }
        pack
    }

    pub fn has(&self, id: CubeletId, tolerance: f64) -> bool {
        self.index.get(&id).is_some_and(|e| e.2 <= tolerance)
    }

    /// The cubelet, if it's here within `tolerance`, with the tolerance it's within.
    pub fn get(&mut self, id: CubeletId, tolerance: f64) -> Option<(f64, Vec<u8>)> {
        let &(at, len, within) = self.index.get(&id).filter(|e| e.2 <= tolerance)?;
        let mut payload = vec![0; len];
        self.store
            .read_at(at, &mut payload)
            .then_some((within, payload))
    }

    /// Appends the cubelet, unless it's here already within `tolerance`.
    pub fn put(&mut self, id: CubeletId, tolerance: f64, payload: &[u8]) {
        let end = self.end + (RECORD + payload.len()) as u64;
        if end > self.room || self.has(id, tolerance) {
            return;
        }
        if self
            .store
            .write_at(self.end, &record(id, tolerance, payload))
        {
            let at = self.end + RECORD as u64;
            self.index.insert(id, (at, payload.len(), tolerance));
            self.end = end;
        } else {
            // Out of quota or disk, so it stops here.
            self.room = self.end;
        }
    }

    /// A cubelet that didn't decode, so that the one sent in its place is kept.
    pub fn forget(&mut self, id: CubeletId) {
        self.index.remove(&id);
    }

    pub fn clear(&mut self) {
        self.store.truncate(self.start);
        self.index.clear();
        self.end = self.start;
    }
}

/// A cubelet's record: its header, then its CUBELET_DATA.
pub fn record(id: CubeletId, tolerance: f64, payload: &[u8]) -> Vec<u8> {
    let mut record = Vec::with_capacity(RECORD + payload.len());
    record.extend((payload.len() as u32).to_le_bytes());
    record.extend([u8::from(id.mode == Mode::Mean), id.level]);
    record.extend([id.x, id.y, id.z].iter().flat_map(|v| v.to_le_bytes()));
    record.extend(tolerance.to_le_bytes());
    record.extend_from_slice(payload);
    record
}

/// A record's cubelet, the tolerance it's within and its payload's length, from its header.
pub fn read_record(head: &[u8; RECORD]) -> (CubeletId, f64, usize) {
    let u16_at = |i: usize| usize::from(u16::from_le_bytes([head[i], head[i + 1]]));
    let mode = if head[4] == 1 { Mode::Mean } else { Mode::Max };
    let at = [u16_at(6), u16_at(8), u16_at(10)];
    let id = CubeletId::new(mode, Default::default(), head[5].into(), at);
    let tolerance = f64::from_le_bytes(head[12..20].try_into().unwrap());
    let len = u32::from_le_bytes(head[..4].try_into().unwrap());
    (id, tolerance, len as usize)
}

/// What a pack is of: the server, which is named so that one server's cubelets are never taken for another's, the
/// file's identity from that server, and the level list its cubelets are cut by.
pub fn identity(server: &str, content_id: &str, levels: &[[usize; 3]]) -> String {
    // The number goes up when a record's layout changes, which leaves the packs before to be evicted.
    format!("cubelets 1, {CUBELET}³, {server}, {content_id}, {levels:?}")
}

/// A server's URL as the host, port and path it's at, which tell servers apart: without the scheme, since `ws://` and
/// `wss://` share a port, and without the query, the fragment or a user, which can carry a token.
pub fn origin(url: &str) -> String {
    let url = &url[..url.find(['?', '#']).unwrap_or(url.len())];
    let (scheme, rest) = match url.find("://") {
        Some(i) if !url[..i].contains('/') => (url[..i].to_ascii_lowercase(), &url[i + 3..]),
        _ => (String::new(), url),
    };
    let (authority, path) = rest.split_at(rest.find('/').unwrap_or(rest.len()));
    let host = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    let has_port = (host.rsplit_once(':'))
        .is_some_and(|(_, port)| !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()));
    let port = match scheme.as_str() {
        _ if has_port => "",
        "wss" | "https" => ":443",
        "ws" | "http" => ":80",
        _ => "",
    };
    let path = path.trim_end_matches('/');
    format!("{}{port}{path}", host.to_ascii_lowercase())
}

/// The name of the file a pack is kept in, a hash of its identity.
pub fn pack_name(identity: &str) -> String {
    format!("{:016x}", fnv(identity.as_bytes()))
}

/// FNV-1a.
pub fn fnv(bytes: &[u8]) -> u64 {
    (bytes.iter()).fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

/// A cubelet's CUBELET_DATA, which is what's kept.
pub type Payload = Arc<[u8]>;

/// Cubelets in memory by what they're of and where, with the tolerance they're within, shared by every file.
pub type Memory = Lru<(Arc<str>, CubeletId), (f64, Payload)>;

/// One file's cubelets in the cache: in memory, and in its pack once one is attached. A cubelet kept at a tighter
/// tolerance serves a looser request.
pub struct Kept {
    identity: Arc<str>,
    /// The pack's name.
    pub name: String,
    /// What the file's cubelets are asked for within, in level 0's units: 0 for exact.
    tolerance: f64,
    pack: Option<Pack>,
    /// The cache was cleared before the pack was attached, so it's emptied then.
    cleared: bool,
    /// Cubelets asked for that are in the cache, for `next`.
    pub hits: VecDeque<CubeletId>,
}

impl Kept {
    pub fn new(identity: String, tolerance: f64) -> Self {
        Self {
            name: pack_name(&identity),
            identity: identity.into(),
            tolerance,
            pack: None,
            cleared: false,
            hits: VecDeque::new(),
        }
    }

    /// The pack, from the store it's in and how long it may grow, or None when it can't be kept.
    pub fn attach(&mut self, store: Option<(Box<dyn Store>, u64)>) {
        self.pack = store.map(|(store, room)| Pack::open(store, &self.identity, room));
        if let (Some(pack), true) = (&mut self.pack, self.cleared) {
            pack.clear();
        }
    }

    /// Empties the pack, now or once it's attached. Memory is the caller's to clear.
    pub fn clear(&mut self) {
        match &mut self.pack {
            Some(pack) => pack.clear(),
            None => self.cleared = true,
        }
    }

    /// Whether `id` is in the cache within the tolerance asked for, in which case it's queued for `next`.
    pub fn hit(&mut self, memory: &mut Memory, id: CubeletId) -> bool {
        let key = (self.identity.clone(), id);
        let in_memory = memory.get(&key).is_some_and(|e| e.0 <= self.tolerance);
        let hit = in_memory
            || self
                .pack
                .as_ref()
                .is_some_and(|p| p.has(id, self.tolerance));
        if hit {
            self.hits.push_back(id);
        }
        hit
    }

    /// The next of the cubelets queued by `hit`, read and decoded, which is slow enough to go one at a time. One that
    /// has gone from the cache since, or doesn't decode, is an error, for the caller to ask for again.
    pub fn next(&mut self, memory: &mut Memory) -> Option<(CubeletId, Result<Cubelet, String>)> {
        let id = self.hits.pop_front()?;
        Some((id, self.read(memory, id)))
    }

    /// A cubelet from the cache, decoded. One the pack can't read or decode is forgotten, so that the one sent again
    /// is kept in its place.
    pub fn read(&mut self, memory: &mut Memory, id: CubeletId) -> Result<Cubelet, String> {
        let payload = self.find(memory, id).ok_or("not in the cache");
        let cubelet = payload
            .map_err(String::from)
            .and_then(|p| codec::decode_payload(&p));
        if cubelet.is_err()
            && let Some(pack) = &mut self.pack
        {
            pack.forget(id);
        }
        cubelet
    }

    /// From memory or else from the pack, and then kept in both: the pack may have been made since the cubelet came.
    fn find(&mut self, memory: &mut Memory, id: CubeletId) -> Option<Payload> {
        let key = (self.identity.clone(), id);
        if let Some(&(within, ref payload)) = memory.get(&key).filter(|e| e.0 <= self.tolerance) {
            if let Some(pack) = &mut self.pack {
                pack.put(id, within, payload);
            }
            return Some(payload.clone());
        }
        let (within, payload) = self.pack.as_mut()?.get(id, self.tolerance)?;
        let payload: Payload = payload.into();
        memory.insert(key, (within, payload.clone()), payload.len());
        Some(payload)
    }

    /// A cubelet that came, as `payload`, within `tolerance`.
    pub fn keep(&mut self, memory: &mut Memory, id: CubeletId, tolerance: f64, payload: &[u8]) {
        if let Some(pack) = &mut self.pack {
            pack.put(id, tolerance, payload);
        }
        let key = (self.identity.clone(), id);
        memory.insert(key, (tolerance, payload.into()), payload.len());
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::Folder;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::{
        fs::{self, File, OpenOptions},
        io::{Read, Seek, SeekFrom, Write},
        path::PathBuf,
    };

    use super::Store;

    impl Store for File {
        fn size(&mut self) -> u64 {
            self.metadata().map_or(0, |m| m.len())
        }

        fn read_at(&mut self, at: u64, buf: &mut [u8]) -> bool {
            self.seek(SeekFrom::Start(at)).is_ok() && self.read_exact(buf).is_ok()
        }

        fn write_at(&mut self, at: u64, bytes: &[u8]) -> bool {
            self.seek(SeekFrom::Start(at)).is_ok() && self.write_all(bytes).is_ok()
        }

        fn truncate(&mut self, size: u64) -> bool {
            self.set_len(size).is_ok()
        }
    }

    /// The native viewer's packs, a file each in a folder of their own, under a cap on them all.
    pub struct Folder {
        pub dir: PathBuf,
        pub cap: u64,
        /// What a pack opened is given room to grow by, evicting others least recently used first.
        pub reserve: u64,
    }

    impl Folder {
        pub fn new(dir: PathBuf) -> Self {
            Self {
                dir,
                cap: 4 << 30,
                reserve: 1 << 30,
            }
        }

        /// The file for pack `name`, made if it isn't there, and how long it may grow. None when another viewer has
        /// it open.
        // ponytail: a fixed reserve, and a pack that outgrows it stops being written to; size it by the cube, or evict
        // as the pack grows, if that bites.
        pub fn open(&self, name: &str) -> Option<(File, u64)> {
            fs::create_dir_all(&self.dir).ok()?;
            let mut packs: Vec<_> = (fs::read_dir(&self.dir).ok()?.flatten())
                .filter_map(|entry| {
                    let meta = entry.metadata().ok()?;
                    Some((meta.modified().ok()?, meta.len(), entry.file_name()))
                })
                .collect();
            packs.sort();
            let size = |name: &str| (packs.iter()).find(|p| p.2 == name).map_or(0, |p| p.1);
            let current = size(name);
            let mut others = packs.iter().map(|p| p.1).sum::<u64>() - current;
            for (_, len, other) in &packs {
                if others + current + self.reserve <= self.cap {
                    break;
                }
                if other != name && fs::remove_file(self.dir.join(other)).is_ok() {
                    others -= len;
                }
            }
            let file = (OpenOptions::new().read(true).write(true).create(true))
                .truncate(false)
                .open(self.dir.join(name))
                .ok()?;
            file.try_lock().ok()?;
            Some((file, self.cap.saturating_sub(others)))
        }

        /// Every pack but `except`, which is open.
        pub fn clear(&self, except: Option<&str>) {
            // Listed first, since removing them while reading the folder can skip some.
            let names: Vec<_> = (fs::read_dir(&self.dir).into_iter().flatten().flatten())
                .map(|entry| entry.file_name())
                .filter(|name| except.is_none_or(|except| name != except))
                .collect();
            for name in names {
                let _ = fs::remove_file(self.dir.join(name));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{self, OpenOptions},
        io::Write,
        time::{Duration, SystemTime},
    };

    use super::*;

    #[test]
    fn the_least_recently_used_go_first_and_a_pack_serves_what_its_tolerance_covers() {
        let mut lru = Lru::new(3 * (100 + 64));
        lru.insert('a', (), 100);
        lru.insert('b', (), 100);
        lru.insert('c', (), 100);
        assert!(lru.get(&'a').is_some());
        lru.insert('d', (), 100);
        assert!(lru.get(&'b').is_none(), "b was the least recently used");
        assert!(['a', 'c', 'd'].iter().all(|k| lru.get(k).is_some()));

        let dir = std::env::temp_dir().join(format!("fits-core-cache-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let folder = Folder {
            dir: dir.clone(),
            cap: 2000,
            reserve: 500,
        };
        let id = |level, x| CubeletId::new(Mode::Mean, Default::default(), level, [x, 1, 2]);
        let open = |name: &str, identity: &str| {
            let (file, room) = folder.open(name).unwrap();
            Pack::open(Box::new(file), identity, room)
        };
        // Packs are kept by server, which a token in the URL doesn't change.
        let server = origin("ws://LocalHost:3002/?token=secret");
        assert_eq!(
            server,
            origin("wss://user:pass@localhost:3002#token=secret")
        );
        assert_eq!(server, "localhost:3002");
        assert_eq!(origin("ws://example.org"), origin("ws://example.org:80/"));
        assert_ne!(origin("ws://example.org"), origin("wss://example.org"));
        assert_ne!(
            origin("wss://hub.org/user/alice"),
            origin("wss://hub.org/user/bob")
        );
        assert_eq!(origin("host:3002/?next=ws://other:1"), "host:3002");
        let cube = |server: &str| identity(server, "cubes/a.fits:100:7:0", &[[1; 3], [2, 2, 1]]);
        let mut pack = open("a", &cube(&server));
        pack.put(id(1, 0), 0.5, &[1; 100]);
        pack.put(id(0, 0), 0.5, &[2; 100]);
        // Level 0 is the same in either mode.
        let max0 = CubeletId::new(Mode::Max, Default::default(), 0, [0, 1, 2]);
        assert_eq!(pack.get(max0, 0.5).map(|c| c.1[0]), Some(2));
        // A tighter cubelet serves a looser request, and a looser one doesn't serve a tighter.
        assert_eq!(pack.get(id(1, 0), 1.0), Some((0.5, vec![1; 100])));
        assert!(pack.get(id(1, 0), 0.25).is_none());
        pack.put(id(1, 0), 0.25, &[3; 50]);
        assert_eq!(pack.get(id(1, 0), 0.25), Some((0.25, vec![3; 50])));
        pack.put(id(1, 0), 1.0, &[7; 10]);
        assert_eq!(pack.get(id(1, 0), 1.0).map(|c| c.1), Some(vec![3; 50]));

        // Reopened, it has what it had, less a record cut short.
        drop(pack);
        let path = dir.join("a");
        let size = fs::metadata(&path).unwrap().len();
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(&[9; RECORD + 10]).unwrap();
        drop(file);
        let mut pack = open("a", &cube(&server));
        assert_eq!(fs::metadata(&path).unwrap().len(), size);
        assert_eq!(pack.get(id(1, 0), 0.5).map(|c| c.1), Some(vec![3; 50]));
        assert!(pack.get(id(0, 0), 0.5).is_some());
        // Of the same file on another server, it starts empty.
        drop(pack);
        let mut pack = open("a", &cube("elsewhere:3002"));
        assert!(pack.get(id(0, 0), 0.5).is_none());

        // A pack that would leave no room for the one opened goes, the least recently used first, and the one opened
        // only grows as far as the cap allows.
        pack.put(id(1, 1), 0.5, &[4; 900]);
        drop(pack);
        let an_hour_ago = SystemTime::now() - Duration::from_secs(3600);
        let a = OpenOptions::new().write(true).open(dir.join("a")).unwrap();
        a.set_modified(an_hour_ago).unwrap();
        let mut pack = open("b", "cube b");
        pack.put(id(1, 1), 0.5, &[5; 600]);
        drop(pack);
        let mut pack = open("c", "cube c");
        assert!(!dir.join("a").exists() && dir.join("b").exists());
        assert!(folder.open("c").is_none(), "another viewer has it");
        pack.put(id(1, 1), 0.5, &[6; 1400]);
        assert!(pack.get(id(1, 1), 0.5).is_none(), "past the cap");
        pack.put(id(1, 1), 0.5, &[6; 500]);
        assert!(pack.get(id(1, 1), 0.5).is_some());

        pack.clear();
        assert!(pack.get(id(1, 1), 0.5).is_none());
        folder.clear(Some("c"));
        assert!(!dir.join("b").exists() && dir.join("c").exists());
        fs::remove_dir_all(&dir).unwrap();
    }
}
