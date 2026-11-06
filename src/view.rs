//! The view on cubelets: the level a region is shown at, the dense textures that hold it, filled cubelet by cubelet
//! over a placeholder, and the order the cubelets are asked for in.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use bevy::{prelude::*, render::storage::ShaderBuffer};

use crate::{
    Dataset, Settings,
    fits::{self, CUBELET, Cell, Cubelet, Processing, Region, Voxels},
    source::{CubeletId, DataSource, Mode, Request},
    upload::{Uploads, Upsample, Upsamples},
    volume::{self, VolumeMaterial, VolumeParams},
};

/// What a cell is before its cubelet comes, when nothing bounds it.
const NEVER_SKIPS: Cell = Cell {
    max: f32::INFINITY,
    has_nan: true,
};

/// The levels a cube is shown at: the power-of-two chain, then the overview level when it isn't one of them.
pub struct Levels {
    pub factors: Vec<[usize; 3]>,
    pub chain: usize,
    overview: usize,
    dims: [usize; 3],
    max_texels: u64,
    max_depth: usize,
}

impl Levels {
    /// `given` is the source's own list, a baked cube's, in place of the one for this budget. It has the same chain,
    /// and its overview is shown only if it fits.
    pub fn new(
        dims: [usize; 3],
        gpu_memory_limit_mb: u64,
        max_depth: usize,
        given: Option<&[[usize; 3]]>,
    ) -> Self {
        let levels = fits::levels(dims, gpu_memory_limit_mb, max_depth);
        let chain = levels.chain.len();
        let (mut factors, mut overview) = levels.list();
        if let Some(given) = given {
            overview =
                (given.iter().position(|&f| f == levels.overview)).unwrap_or(given.len() - 1);
            factors = given.to_vec();
        }
        Self {
            factors,
            chain,
            overview,
            dims,
            max_texels: fits::texel_budget(gpu_memory_limit_mb),
            max_depth: max_depth.max(1),
        }
    }

    /// The last of the chain, which fits in one cubelet.
    pub fn coarsest(&self) -> usize {
        self.chain - 1
    }

    /// The texels of `level` that `region` takes in.
    pub fn texels(&self, region: Region, level: usize) -> Region {
        let f = self.factors[level];
        Region {
            min: std::array::from_fn(|i| region.min[i] / f[i]),
            max: std::array::from_fn(|i| region.max[i] / f[i]),
        }
    }

    /// The level `region` is shown at: the finest of the chain whose texels over it fit the GPU budget and the
    /// device's layers, or for the whole cube the overview when that's finer.
    pub fn for_region(&self, region: Region) -> usize {
        let texels = |level| self.texels(region, level).extent().map(|d| d as u64);
        let fits = |level: &usize| {
            let [x, y, z] = texels(*level);
            x <= 2048 && y <= 2048 && z <= self.max_depth as u64 && x * y * z <= self.max_texels
        };
        let fit = (0..self.chain).find(fits).unwrap_or(self.coarsest());
        let count = |level| texels(level).iter().product::<u64>();
        if region == Region::whole(self.dims)
            && fits(&self.overview)
            && count(self.overview) > count(fit)
        {
            self.overview
        } else {
            fit
        }
    }

    /// Whether a texture of `level` over `texels` shows `region` at least as finely as the level `region` is shown
    /// at, so that cropping to it only moves the clip bounds.
    pub fn shows(&self, level: usize, texels: Region, region: Region) -> bool {
        let (have, want) = (self.factors[level], self.factors[self.for_region(region)]);
        (0..3).all(|i| have[i] <= want[i]) && covers(have, texels, region)
    }
}

/// Whether `texels` of a level of `factors` take in all of `region` of level 0.
fn covers(factors: [usize; 3], texels: Region, region: Region) -> bool {
    (0..3).all(|i| {
        texels.min[i] * factors[i] <= region.min[i]
            && region.max[i] < (texels.max[i] + 1) * factors[i]
    })
}

/// The level 0 voxels of the cubelet at `at` of a level of `factors`.
fn footprint(dims: [usize; 3], factors: [usize; 3], at: [usize; 3]) -> Region {
    fits::cubelet_region(dims, factors, at).expect("a cubelet inside the level")
}

/// The max pyramid's coarsest level, whose every voxel bounds the voxels under it.
pub struct Coarse {
    factors: [usize; 3],
    dims: [usize; 3],
    data: Vec<f32>,
}

impl Coarse {
    fn new(factors: [usize; 3], cubelet: &Cubelet) -> Self {
        let n = cubelet.dims.iter().product();
        let data = match &cubelet.voxels {
            Voxels::AllNan => vec![f32::NAN; n],
            Voxels::Constant(v) => vec![*v; n],
            Voxels::Data(data) => data.clone(),
        };
        Self {
            factors,
            dims: cubelet.dims,
            data,
        }
    }

    /// Over the coarse voxels that `region` of level 0 takes in.
    fn cell(&self, region: Region) -> Cell {
        let texels = Region {
            min: std::array::from_fn(|i| region.min[i] / self.factors[i]),
            max: std::array::from_fn(|i| (region.max[i] / self.factors[i]).min(self.dims[i] - 1)),
        };
        Cell::of(&self.data, self.dims, texels)
    }
}

/// The cubelets of a level of `factors` that a texture over `texels` takes, nearest `centre` (in level 0 voxels)
/// first, less those `coarse` says can't show: all NaN, or, given a Maximum `floor`, all below it.
fn plan(
    dims: [usize; 3],
    factors: [usize; 3],
    texels: Region,
    centre: Vec3,
    coarse: Option<&Coarse>,
    floor: Option<f32>,
) -> Vec<[usize; 3]> {
    let (lo, hi) = (
        texels.min.map(|t| t / CUBELET),
        texels.max.map(|t| t / CUBELET),
    );
    let mut cubelets = Vec::new();
    for z in lo[2]..=hi[2] {
        for y in lo[1]..=hi[1] {
            for x in lo[0]..=hi[0] {
                let at = [x, y, z];
                let voxels = footprint(dims, factors, at);
                let cell = coarse.map(|coarse| coarse.cell(voxels));
                let nan = cell.is_some_and(|c| c.max == f32::NEG_INFINITY && c.has_nan);
                if nan || cell.zip(floor).is_some_and(|(c, floor)| c.max < floor) {
                    continue;
                }
                let middle = Vec3::from_array(std::array::from_fn(|i| {
                    (voxels.min[i] + voxels.max[i] + 1) as f32 / 2.0
                }));
                cubelets.push((at, middle.distance_squared(centre)));
            }
        }
    }
    cubelets.sort_by(|a, b| a.1.total_cmp(&b.1));
    cubelets.into_iter().map(|(at, _)| at).collect()
}

/// Where the cubelet at `at`, of `dims` texels, goes in a texture over `texels` of its level: the part of it inside
/// the texture, and that part's origin in the texture.
fn placement(texels: Region, at: [usize; 3], dims: [usize; 3]) -> Option<(Region, [usize; 3])> {
    let mut part = Region::whole(dims);
    let mut origin = [0; 3];
    for i in 0..3 {
        let start = at[i] * CUBELET;
        let (lo, hi) = (
            start.max(texels.min[i]),
            (start + dims[i] - 1).min(texels.max[i]),
        );
        if lo > hi {
            return None;
        }
        (part.min[i], part.max[i], origin[i]) = (lo - start, hi - start, lo - texels.min[i]);
    }
    Some((part, origin))
}

/// Which pyramid a cubelet comes from, and what's done to it.
type Kind = (Mode, Processing);

#[derive(Clone, Copy, PartialEq)]
enum Have {
    No,
    Asked(Kind),
    Got(Kind),
}

/// A dense texture over `texels` of one level, filled cubelet by cubelet. Its grid has an entry per cubelet it
/// takes in, including those cut by its edges.
pub struct VolumeTexture {
    pub level: usize,
    pub factors: [usize; 3],
    pub texels: Region,
    pub handle: Handle<Image>,
    pub grid: Handle<ShaderBuffer>,
    pub has_nan: bool,
    /// Set once a coarser texture has been upsampled into it, for a texture that starts that way.
    placeholder: Option<Arc<AtomicBool>>,
    /// For one that doesn't, set once every cubelet is in.
    filled: bool,
    cells: Vec<Cell>,
    have: Vec<Have>,
    cells_changed: bool,
    /// The kind, Maximum floor and whether culling was possible when it was last planned.
    planned: Option<(Kind, Option<u32>, bool)>,
}

impl VolumeTexture {
    /// Until a cubelet comes its cell holds what was upsampled there, which `coarse` bounds.
    pub fn new(
        levels: &Levels,
        region: Region,
        level: usize,
        placeholder: bool,
        coarse: Option<&Coarse>,
        images: &mut Assets<Image>,
        buffers: &mut Assets<ShaderBuffer>,
    ) -> Self {
        let texels = levels.texels(region, level);
        let factors = levels.factors[level];
        let mut texture = Self {
            level,
            factors,
            texels,
            handle: images.add(volume::empty_cube_image(texels.extent(), placeholder)),
            grid: Handle::default(),
            has_nan: true,
            placeholder: placeholder.then(Arc::default),
            filled: false,
            cells: Vec::new(),
            have: Vec::new(),
            cells_changed: true,
            planned: None,
        };
        let count = texture.cells().iter().product();
        let bound = |at| match coarse.filter(|_| placeholder) {
            Some(coarse) => {
                // A texel more each side, for samples that round across the cell's faces.
                let voxels = fits::grow(footprint(levels.dims, factors, at), levels.dims, factors);
                Cell {
                    max: coarse.cell(voxels).max,
                    has_nan: true,
                }
            }
            None => NEVER_SKIPS,
        };
        texture.cells = (0..count).map(|i| bound(texture.at(i))).collect();
        texture.have = vec![Have::No; count];
        texture.update_grid(buffers);
        texture
    }

    pub fn dims(&self) -> [usize; 3] {
        self.texels.extent()
    }

    /// The cubelets it takes in along each axis.
    pub fn cells(&self) -> [usize; 3] {
        std::array::from_fn(|i| self.texels.max[i] / CUBELET - self.texels.min[i] / CUBELET + 1)
    }

    /// How far its first cubelet starts before it, in texels.
    pub fn grid_offset(&self) -> [usize; 3] {
        self.texels.min.map(|t| t % CUBELET)
    }

    fn at(&self, i: usize) -> [usize; 3] {
        let [cx, cy, _] = self.cells();
        let first = self.texels.min.map(|t| t / CUBELET);
        [
            first[0] + i % cx,
            first[1] + i / cx % cy,
            first[2] + i / (cx * cy),
        ]
    }

    fn index(&self, at: [usize; 3]) -> Option<usize> {
        let cells = self.cells();
        let first = self.texels.min.map(|t| t / CUBELET);
        let local: [usize; 3] = std::array::from_fn(|i| at[i].wrapping_sub(first[i]));
        (0..3)
            .all(|i| local[i] < cells[i])
            .then(|| local[0] + cells[0] * (local[1] + cells[1] * local[2]))
    }

    /// Its box in object space, which can reach past the cube's far faces.
    pub fn bounds(&self, dims: [usize; 3]) -> (Vec3, Vec3) {
        let at = |t: [usize; 3]| {
            Vec3::from_array(std::array::from_fn(|i| {
                (t[i] * self.factors[i]) as f32 / dims[i] as f32
            })) - 0.5
        };
        (at(self.texels.min), at(self.texels.max.map(|t| t + 1)))
    }

    pub fn covers(&self, region: Region) -> bool {
        covers(self.factors, self.texels, region)
    }

    /// Whether it holds a processed cubelet.
    fn processed(&self) -> bool {
        (self.have.iter())
            .any(|have| matches!(have, Have::Got((_, processing)) if !processing.is_none()))
    }

    pub fn drawable(&self) -> bool {
        self.filled || (self.placeholder.as_ref()).is_some_and(|done| done.load(Ordering::Relaxed))
    }

    fn asked(&self) -> Vec<CubeletId> {
        (self.have.iter().enumerate())
            .filter_map(|(i, have)| match *have {
                Have::Asked((mode, processing)) => {
                    Some(CubeletId::new(mode, processing, self.level, self.at(i)))
                }
                _ => None,
            })
            .collect()
    }

    /// What to ask for and what to stop asking for, after the kind, the floor or the coarse level has changed since
    /// the last time. Only a texture with a placeholder leaves cubelets out, since the rest of it would be empty.
    fn plan(
        &mut self,
        dims: [usize; 3],
        centre: Vec3,
        coarse: Option<&Coarse>,
        (mode, processing): Kind,
        floor: Option<f32>,
    ) -> (Vec<CubeletId>, Vec<CubeletId>) {
        let kind = (if self.level == 0 { Mode::Max } else { mode }, processing);
        let coarse = coarse.filter(|_| self.placeholder.is_some());
        let floor = floor.filter(|_| coarse.is_some());
        let key = (kind, floor.map(f32::to_bits), coarse.is_some());
        if self.planned.replace(key) == Some(key) {
            return default();
        }
        let mut remove = Vec::new();
        for i in 0..self.have.len() {
            if let Have::Asked(asked) = self.have[i]
                && asked != kind
            {
                remove.push(CubeletId::new(asked.0, asked.1, self.level, self.at(i)));
                self.have[i] = Have::No;
            }
        }
        let mut add = Vec::new();
        for at in plan(dims, self.factors, self.texels, centre, coarse, floor) {
            let i = self.index(at).unwrap();
            if !matches!(self.have[i], Have::Asked(k) | Have::Got(k) if k == kind) {
                self.have[i] = Have::Asked(kind);
                add.push(CubeletId::new(kind.0, kind.1, self.level, at));
            }
        }
        (add, remove)
    }

    fn wants(&self, id: CubeletId) -> bool {
        id.level as usize == self.level
            && (self.index(id.at()))
                .is_some_and(|i| self.have[i] == Have::Asked((id.mode, id.processing)))
    }

    fn receive(&mut self, id: CubeletId, cubelet: Cubelet, uploads: &mut Uploads) {
        let Some(i) = self.index(id.at()) else {
            return;
        };
        self.have[i] = Have::Got((id.mode, id.processing));
        self.cells[i] = cubelet.cell;
        self.cells_changed = true;
        let Some((part, origin)) = placement(self.texels, id.at(), cubelet.dims) else {
            return;
        };
        let n = part.extent().iter().product();
        let voxels = match cubelet.voxels {
            Voxels::AllNan => vec![f32::NAN; n],
            Voxels::Constant(v) => vec![v; n],
            Voxels::Data(data) if part == Region::whole(cubelet.dims) => data,
            Voxels::Data(data) => {
                fits::crop_and_downsample(&data, cubelet.dims, part, [1; 3], true).1
            }
        };
        let after = self.placeholder.clone();
        uploads.push_at(self.handle.id(), origin, part.extent(), voxels, after);
    }

    fn update_grid(&mut self, buffers: &mut Assets<ShaderBuffer>) {
        self.grid = volume::grid_buffer(&self.cells, buffers);
        self.has_nan = self.cells.iter().any(|cell| cell.has_nan);
        self.cells_changed = false;
    }

    /// Fills this texture's placeholder from `coarser`, which takes in all of it.
    fn upsample_from(&self, coarser: &VolumeTexture) -> Upsample {
        let factors = |t: &VolumeTexture| Vec3::from_array(t.factors.map(|f| f as f32));
        let corner =
            |t: &VolumeTexture| factors(t) * Vec3::from_array(t.texels.min.map(|m| m as f32));
        let scale = factors(self) / factors(coarser);
        Upsample {
            src: coarser.handle.clone(),
            dst: self.handle.id(),
            scale,
            offset: (corner(self) - corner(coarser)) / factors(coarser) + 0.5 * scale,
            src_dims: coarser.dims(),
            dst_dims: self.dims(),
            done: self.placeholder.clone().unwrap(),
        }
    }

    /// Stops what it asked for and its placeholder, if they're still to come.
    fn cancel(self, source: &DataSource, upsamples: &mut Upsamples) {
        upsamples.0.retain(|job| job.dst != self.handle.id());
        let ids = self.asked();
        if !ids.is_empty() {
            source.send(Request::RemoveCubelets { ids });
        }
    }
}

#[derive(Resource)]
pub struct CubeState {
    pub material: Handle<VolumeMaterial>,
    pub levels: Levels,
    /// The whole cube, kept for an instant reset.
    pub full: VolumeTexture,
    /// A finer texture for a crop.
    pub crop: Option<VolumeTexture>,
    /// A finer texture for what's visible, which takes over once its placeholder is in.
    pub next: Option<VolumeTexture>,
    coarse: Option<Coarse>,
    /// The projection mode last planned for.
    mode: Option<Mode>,
    /// The clip box.
    pub visible: Region,
    pub selection: Option<Region>,
}

impl CubeState {
    pub fn new(
        material: Handle<VolumeMaterial>,
        levels: Levels,
        full: VolumeTexture,
        dims: [usize; 3],
    ) -> Self {
        Self {
            material,
            levels,
            full,
            crop: None,
            next: None,
            coarse: None,
            mode: None,
            visible: Region::whole(dims),
            selection: None,
        }
    }

    /// The texture on screen.
    pub fn shown(&self) -> &VolumeTexture {
        self.crop.as_ref().unwrap_or(&self.full)
    }

    pub fn cropped(&self) -> bool {
        self.visible != Region::whole(self.levels.dims)
    }

    /// Nothing is still to come for the view.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn settled(&self, uploads: &Uploads) -> bool {
        let shown = self.shown();
        self.next.is_none()
            && shown.drawable()
            && !shown.have.iter().any(|have| matches!(have, Have::Asked(_)))
            && uploads.is_empty()
    }

    /// Each of the coarse level's planes' flux over `cut`, and the level 0 channels a plane takes in, once it's here.
    pub fn line_flux(&self, cut: f32) -> Option<(Vec<f32>, usize)> {
        let coarse = self.coarse.as_ref()?;
        let plane = coarse.dims[0] * coarse.dims[1];
        let flux = (coarse.data.chunks(plane))
            .map(|plane| plane.iter().map(|&v| (v - cut).max(0.0)).sum())
            .collect();
        Some((flux, coarse.factors[2]))
    }

    pub fn receive(&mut self, id: CubeletId, cubelet: Cubelet, uploads: &mut Uploads) {
        let level = id.level as usize;
        let plain = id.mode == Mode::Max && id.processing.is_none();
        if level == self.levels.coarsest() && plain && self.coarse.is_none() {
            self.coarse = Some(Coarse::new(self.levels.factors[level], &cubelet));
        }
        let textures = std::iter::once(&mut self.full)
            .chain(&mut self.crop)
            .chain(&mut self.next);
        if let Some(texture) = textures.into_iter().find(|texture| texture.wants(id)) {
            texture.receive(id, cubelet, uploads);
        }
    }
}

/// The shader's floor_value: a Maximum ray under it shows nothing.
pub fn floor_value(params: &VolumeParams) -> f32 {
    let range = match params.scale_max - params.scale_min {
        0.0 => 1.0,
        range => range,
    };
    params.threshold_min * range + params.scale_min
}

/// Keeps the textures in step with what's visible: a finer texture when the shown one is too coarse, started from it
/// upsampled and taking over once that's in; the cubelets they need for the projection mode and threshold; and their
/// grids as cubelets arrive.
pub fn refine(
    data: Res<Dataset>,
    settings: Res<Settings>,
    mut state: ResMut<CubeState>,
    uploads: Res<Uploads>,
    mut upsamples: ResMut<Upsamples>,
    mut images: ResMut<Assets<Image>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    let (dims, source) = (data.info.dims, &data.source);
    let whole = Region::whole(dims);
    let (params, mode, processing) = (&settings.params, settings.mode(), settings.modes.processing);
    // The coarse level only bounds unprocessed data.
    let floor = (mode == Mode::Max && processing.is_none()).then(|| floor_value(params));
    let s = state.bypass_change_detection();
    let mut changed = false;

    if s.next.as_ref().is_some_and(VolumeTexture::drawable) {
        let next = s.next.take().unwrap();
        upsamples.0.retain(|job| job.dst != next.handle.id());
        let old = if next.texels == s.levels.texels(whole, next.level) {
            Some(std::mem::replace(&mut s.full, next))
        } else if next.covers(s.visible) {
            s.crop.replace(next)
        } else {
            Some(next)
        };
        if let Some(old) = old {
            old.cancel(source, &mut upsamples);
        }
        changed = true;
    }
    if (s.crop.as_ref()).is_some_and(|crop| !s.cropped() || !crop.covers(s.visible)) {
        s.crop.take().unwrap().cancel(source, &mut upsamples);
        changed = true;
    }
    let level = s.levels.for_region(s.visible);
    let shown = s.crop.as_ref().unwrap_or(&s.full);
    if s.levels.shows(shown.level, shown.texels, s.visible) {
        if let Some(next) = s.next.take() {
            next.cancel(source, &mut upsamples);
            changed = true;
        }
    } else if shown.drawable()
        && !(s.next.as_ref()).is_some_and(|next| next.level == level && next.covers(s.visible))
    {
        // The placeholder is the shown texture's data, which only the coarse level bounds while it's unprocessed.
        let coarse = (s.coarse.as_ref()).filter(|_| processing.is_none() && !shown.processed());
        let next = VolumeTexture::new(
            &s.levels,
            s.visible,
            level,
            true,
            coarse,
            &mut images,
            &mut buffers,
        );
        upsamples.0.push(next.upsample_from(shown));
        if let Some(old) = s.next.replace(next) {
            old.cancel(source, &mut upsamples);
        }
        changed = true;
    }

    let centre = Vec3::from_array(std::array::from_fn(|i| {
        (s.visible.min[i] + s.visible.max[i] + 1) as f32 / 2.0
    }));
    // The whole cube's after a crop's, so that a reset shows this mode too.
    for texture in s.next.iter_mut().chain(&mut s.crop).chain([&mut s.full]) {
        let kind = (mode, processing);
        let (add, remove) = texture.plan(dims, centre, s.coarse.as_ref(), kind, floor);
        if !remove.is_empty() {
            source.send(Request::RemoveCubelets { ids: remove });
        }
        if !add.is_empty() {
            source.send(Request::AddCubelets { ids: add });
        }
    }
    // The max coarsest cubelet bounds and culls the rest, so it's asked for again on a switch to Maximum when a
    // switch before it came cancelled it.
    let switched = s.mode.replace(mode) != Some(mode);
    let coarsest = CubeletId::new(
        Mode::Max,
        Processing::default(),
        s.levels.coarsest(),
        [0; 3],
    );
    let mut textures = std::iter::once(&s.full).chain(&s.crop).chain(&s.next);
    if switched && mode == Mode::Max && s.coarse.is_none() && !textures.any(|t| t.wants(coarsest)) {
        source.send(Request::AddCubelets {
            ids: vec![coarsest],
        });
    }

    for texture in std::iter::once(&mut s.full)
        .chain(&mut s.crop)
        .chain(&mut s.next)
    {
        if texture.cells_changed {
            texture.update_grid(&mut buffers);
            changed = true;
        }
        let all_in = texture.have.iter().all(|have| matches!(have, Have::Got(_)));
        if !texture.filled && all_in && !uploads.is_filling(texture.handle.id()) {
            texture.filled = true;
            changed = true;
        }
    }
    if changed {
        state.set_changed();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubelets_come_nearest_the_middle_first_without_those_that_cant_show() {
        // Level 0 of a 256 x 256 x 64 cube, four cubelets by four, under a coarsest level of 4 x 4 x 1.
        let dims = [256, 256, 64];
        let levels = Levels::new(dims, 384, 2048, None);
        assert_eq!(levels.factors[levels.coarsest()], [4, 4, 1]);
        // Each cubelet's coarse texels are its x + 4y, apart from one cubelet that's all NaN and one with a NaN.
        let data = (0..64 * 64 * 64)
            .map(|i| match [i % 64 / 16, i / 64 % 64 / 16] {
                [0, 3] => f32::NAN,
                [3, 0] if i % 64 == 63 => f32::NAN,
                [x, y] => (x + 4 * y) as f32,
            })
            .collect();
        let coarse = Coarse {
            factors: [4, 4, 1],
            dims: [64, 64, 64],
            data,
        };
        let (mut images, mut buffers) = (Assets::default(), Assets::default());
        let mut texture = VolumeTexture::new(
            &levels,
            Region::whole(dims),
            0,
            true,
            Some(&coarse),
            &mut images,
            &mut buffers,
        );
        // Off-centre: 16 voxels from the middle of [2, 2] in x and y, then [3, 2] and [2, 3], then [3, 3].
        let centre = Vec3::new(176.0, 176.0, 32.0);
        let at = |ids: Vec<CubeletId>| ids.iter().map(|id| [id.x, id.y]).collect::<Vec<_>>();

        // Maximum at a floor of 9.5 leaves out all but 10, 11, 13, 14 and 15, and never the NaN cubelet's 12.
        let (max, mean) = (
            (Mode::Max, Processing::default()),
            (Mode::Mean, Processing::default()),
        );
        let (add, remove) = texture.plan(dims, centre, Some(&coarse), max, Some(9.5));
        assert_eq!(at(add), [[2, 2], [3, 2], [2, 3], [3, 3], [1, 3]]);
        assert!(remove.is_empty());
        let again = texture.plan(dims, centre, Some(&coarse), max, Some(9.5));
        assert_eq!(again, default());

        // Lowering the floor asks for what it left out, nearest first; [3, 0], with its one NaN, comes in.
        let (add, _) = texture.plan(dims, centre, Some(&coarse), max, Some(1.5));
        let want = [
            [2, 1],
            [1, 2],
            [3, 1],
            [1, 1],
            [2, 0],
            [0, 2],
            [3, 0],
            [0, 1],
        ];
        assert_eq!(at(add), want);

        // Average has no floor and level 0 is the same in both pyramids, so it only adds the two under 1.5.
        let (add, remove) = texture.plan(dims, centre, Some(&coarse), mean, None);
        assert_eq!((at(add), remove), (vec![[1, 0], [0, 0]], vec![]));
    }

    #[test]
    fn a_crop_is_clip_bounds_while_the_shown_level_is_fine_enough() {
        let region = |min: [usize; 3], max: [usize; 3]| Region { min, max };
        // M83 CO opens at 2 x 2 x 1, its overview and the chain's second level.
        let dims = [1238, 1236, 196];
        let levels = Levels::new(dims, 384, 2048, None);
        let whole = Region::whole(dims);
        assert_eq!(
            (levels.for_region(whole), levels.factors[1]),
            (1, [2, 2, 1])
        );
        let full = levels.texels(whole, 1);
        // 900 x 900 still needs 2 x 2 x 1, so it's clip bounds on the whole cube's texture, and so is going back.
        let wide = region([100, 100, 0], [999, 999, 195]);
        assert!(levels.shows(1, full, wide) && levels.shows(1, full, whole));
        // 500 x 500 fits at full resolution, so it needs a texture of its own, which then shows anything inside it.
        let narrow = region([300, 300, 0], [799, 799, 195]);
        assert_eq!(levels.for_region(narrow), 0);
        assert!(!levels.shows(1, full, narrow));
        let crop = levels.texels(narrow, 0);
        assert!(levels.shows(0, crop, region([400, 400, 10], [500, 500, 20])));
        assert!(!levels.shows(0, crop, region([200, 400, 10], [500, 500, 20])));

        // CGPS opens at its 5 x 5 x 1 overview, finer than the chain's 8 x 8 x 1, which still shows a crop that
        // needs 8 x 8 x 1, but not one that needs 4 x 4 x 1.
        let dims = [4096, 4096, 100];
        let levels = Levels::new(dims, 384, 2048, None);
        let whole = Region::whole(dims);
        let overview = levels.for_region(whole);
        assert_eq!(
            (levels.factors[overview], overview),
            ([5, 5, 1], levels.chain)
        );
        let full = levels.texels(whole, overview);
        let at_8 = region([0, 0, 0], [3999, 3999, 99]);
        let at_4 = region([0, 0, 0], [2999, 2999, 99]);
        assert_eq!(levels.factors[levels.for_region(at_8)], [8, 8, 1]);
        assert_eq!(levels.factors[levels.for_region(at_4)], [4, 4, 1]);
        assert!(levels.shows(overview, full, at_8) && !levels.shows(overview, full, at_4));
    }

    #[test]
    fn cubelets_cut_by_the_texture_or_the_cube_go_in_at_their_offsets() {
        // A texture over texels 10..=140, 0..=63 and 60..=69 of a level of 150 x 64 x 70, whose cubelets at x = 2
        // are 22 wide and at z = 1 are 6 deep.
        let texels = Region {
            min: [10, 0, 60],
            max: [140, 63, 69],
        };
        let place = |at, dims| {
            placement(texels, at, dims).map(|(part, origin)| (part.min, part.max, origin))
        };
        assert_eq!(
            place([0, 0, 0], [64, 64, 64]),
            Some(([10, 0, 60], [63, 63, 63], [0, 0, 0]))
        );
        assert_eq!(
            place([1, 0, 1], [64, 64, 6]),
            Some(([0, 0, 0], [63, 63, 5], [54, 0, 4]))
        );
        assert_eq!(
            place([2, 0, 1], [22, 64, 6]),
            Some(([0, 0, 0], [12, 63, 5], [118, 0, 4]))
        );
        assert_eq!(place([3, 0, 0], [64, 64, 64]), None);
    }
}
