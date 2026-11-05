//! The rendering modes beyond Maximum and Average, each its own shader specialisation, with their controls: the σ
//! level, the mask and the processing the cubelets come with. The data source measures the shown texture's noise and
//! makes its mask.

use bevy::{
    asset::{RenderAssetUsages, uuid_handle},
    prelude::*,
    render::render_resource::{
        Extent3d, RenderPipelineDescriptor, ShaderType, TextureDimension, TextureFormat,
        TextureViewDescriptor, TextureViewDimension,
    },
};
use bevy_egui::egui;

use crate::{
    CubeState, Dataset, Settings,
    config::ASINH,
    fits::{Noise, Processing, Smoothing},
    source::{Event, Mode, Request},
    upload::Uploads,
    view,
    volume::{VolumeMaterial, VolumeParams},
};

pub const SIGMA_LEVELS: [f32; 4] = [1.0, 3.0, 5.0, 10.0];
/// SoFiA-2's default `scfind.threshold`.
pub const MASK_SIGMAS: f32 = 5.0;

const MEASURING: &str = "measuring the noise";

/// No mask: one voxel, bound until a mask is made.
pub const NO_MASK: Handle<Image> = uuid_handle!("6f0c1d0e-3b7a-4f53-9a51-2c0b4d8e7a19");

pub struct ModesPlugin;

impl Plugin for ModesPlugin {
    fn build(&self, app: &mut App) {
        let cube_shown = resource_exists::<CubeState>;
        app.init_resource::<Extras>()
            .add_message::<GenerateMask>()
            .add_systems(Startup, insert_no_mask)
            .add_systems(
                Update,
                (
                    (follow_texture, generate_mask)
                        .before(crate::update_material)
                        .run_if(cube_shown),
                    update_material.after(crate::update_material).run_if(
                        cube_shown.and_then(
                            resource_exists_and_changed::<Settings>
                                .or_eager(resource_exists_and_changed::<CubeState>)
                                .or_eager(resource_changed::<Extras>),
                        ),
                    ),
                ),
            );
    }
}

/// A texture as it's shown: which it is, the pyramid its cubelets come from and their processing.
type Shown = (AssetId<Image>, Mode, Processing);

/// The texture shown, as it's shown. Level 0 is the same in both pyramids.
fn shown(settings: &Settings, state: &CubeState) -> Shown {
    let texture = state.shown();
    let mode = if texture.level == 0 {
        Mode::Max
    } else {
        settings.mode()
    };
    (texture.handle.id(), mode, settings.modes.processing)
}

/// What the data source has measured or made for the texture shown, its noise and its mask, and the span of the
/// channel colours. One per dataset.
#[derive(Resource, Default)]
pub struct Extras {
    next_id: u32,
    asked: Option<(Shown, u32)>,
    /// The last noise measured, kept through a re-measure so that the levels don't jump about in between.
    noise: Option<Noise>,
    mask: Option<Mask>,
    /// The level 0 channels the channel colours run from and to, once the coarse level is in.
    channels: Option<(f32, f32)>,
}

struct Mask {
    shown: Shown,
    id: u32,
    sigmas: f32,
    image: Handle<Image>,
    /// Once it's made.
    voxels: Option<usize>,
    /// Once its parts are all written into `image`.
    ready: bool,
}

impl Extras {
    pub fn noise(&self) -> Option<Noise> {
        self.noise
    }

    /// The mask, for the texture as it's shown: the threshold it was made at, and how many texels are in it once it's
    /// ready.
    pub fn mask(&self) -> Option<(f32, Option<usize>)> {
        (self.mask.as_ref()).map(|mask| (mask.sigmas, mask.voxels.filter(|_| mask.ready)))
    }

    fn ready_mask(&self) -> Option<&Handle<Image>> {
        (self.mask.as_ref())
            .filter(|mask| mask.ready)
            .map(|mask| &mask.image)
    }
}

/// A mask asked for, at this threshold in σ.
#[derive(Message)]
pub struct GenerateMask(pub f32);

/// Asks for the noise of the texture shown each time it changes or is shown otherwise, once it's the one the view
/// wants rather than a coarser one on the way to it; drops a mask made for another or for this one as it was shown
/// before, and readies one once its parts are written; and finds the channel colours' span once the coarse level is
/// in.
fn follow_texture(
    settings: Res<Settings>,
    state: Res<CubeState>,
    data: Res<Dataset>,
    uploads: Res<Uploads>,
    mut extras: ResMut<Extras>,
) {
    if extras.channels.is_none()
        && let Some(channels) = channels(&state, &data)
    {
        extras.channels = Some(channels);
    }
    let (texture, shown) = (state.shown(), shown(&settings, &state));
    // A server measures nothing, so the noise is the cube's at open scaled to the level shown, with no baseline.
    if data.source.is_remote() {
        let noise = data.info.sigma.map(|sigma| Noise {
            median: 0.0,
            sigma: (sigma * shown.1.noise_ratio(texture.factors)) as f32,
        });
        if extras.noise != noise {
            extras.noise = noise;
        }
        return;
    }
    if extras.mask.as_ref().is_some_and(|mask| mask.shown != shown) {
        extras.mask = None;
    }
    let written =
        |mask: &Mask| mask.voxels.is_some() && !mask.ready && !uploads.is_filling(mask.image.id());
    // Through a shared reference first, so that Extras only changes when the mask does.
    if extras.mask.as_ref().is_some_and(written) {
        extras.mask.as_mut().unwrap().ready = true;
    }
    let wanted =
        state.next.is_none() && (state.levels).shows(texture.level, texture.texels, state.visible);
    if !wanted || extras.asked.is_some_and(|(asked, _)| asked == shown) {
        return;
    }
    extras.next_id += 1;
    let id = extras.next_id;
    data.source.send(Request::Measure {
        id,
        mode: shown.1,
        processing: shown.2,
        level: texture.level as u8,
        texels: texture.texels,
    });
    extras.asked = Some((shown, id));
}

/// A mask for the texture shown as it's shown, at the threshold asked for, its parts written into an R8 texture of its
/// own as they come.
fn generate_mask(
    mut requests: MessageReader<GenerateMask>,
    settings: Res<Settings>,
    state: Res<CubeState>,
    data: Res<Dataset>,
    mut extras: ResMut<Extras>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some(&GenerateMask(sigmas)) = requests.read().last() else {
        return;
    };
    let (texture, shown) = (state.shown(), shown(&settings, &state));
    extras.next_id += 1;
    let id = extras.next_id;
    data.source.send(Request::Mask {
        id,
        mode: shown.1,
        processing: shown.2,
        level: texture.level as u8,
        texels: texture.texels,
        threshold: sigmas,
    });
    extras.mask = Some(Mask {
        shown,
        id,
        sigmas,
        image: images.add(mask_image(texture.dims(), None)),
        voxels: None,
        ready: false,
    });
}

/// The level 0 channels the channel colours run from and to, swapped so that they run with velocity, once the
/// coarse level is in: those of the middle 98% of its flux above 5σ of level 0, or all of them when there's none.
fn channels(state: &CubeState, data: &Dataset) -> Option<(f32, f32)> {
    let cut = (data.info.sigma).map_or(f32::INFINITY, |sigma| 5.0 * sigma as f32);
    let (flux, per_plane) = state.line_flux(cut)?;
    let (from, to) = line_planes(&flux).map_or((0, data.info.dims[2]), |(from, to)| {
        (from * per_plane, to * per_plane)
    });
    let falls = (data.wcs.as_ref()).is_some_and(|wcs| wcs.velocity_falls(data.info.dims));
    let (from, to) = (from as f32, to as f32);
    Some(if falls { (to, from) } else { (from, to) })
}

/// The source's answers to the requests above.
pub fn receive(event: Event, extras: &mut ResMut<Extras>, uploads: &mut Uploads) {
    match event {
        Event::Measured { id, noise } => {
            if extras.asked.is_some_and(|(_, asked)| asked == id) && noise.is_some() {
                extras.noise = noise;
            }
        }
        Event::Mask { id, voxels } => {
            if let Some(mask) = extras.mask.as_mut().filter(|mask| mask.id == id) {
                mask.voxels = Some(voxels);
            }
        }
        // Not a change for the material, which takes the mask once its parts are written.
        Event::MaskPart {
            id,
            origin,
            dims,
            mask,
        } => {
            let extras = extras.bypass_change_detection();
            if let Some(image) = (extras.mask.as_ref())
                .filter(|m| m.id == id)
                .map(|m| m.image.id())
            {
                uploads.push_bytes(image, origin, dims, mask);
            }
        }
        Event::Cubelet { .. } | Event::PixelValue { .. } => {}
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RenderMode {
    #[default]
    Maximum,
    Average,
    DepthCued,
    Attenuated,
    PeakChannel,
    MeanChannel,
    Emission,
    Isosurface,
    Shells,
    ClippedMoment,
}

impl RenderMode {
    const ALL: [Self; 10] = [
        Self::Maximum,
        Self::Average,
        Self::DepthCued,
        Self::Attenuated,
        Self::PeakChannel,
        Self::MeanChannel,
        Self::Emission,
        Self::Isosurface,
        Self::Shells,
        Self::ClippedMoment,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Maximum => "Maximum",
            Self::Average => "Average",
            Self::DepthCued => "Maximum, depth cued",
            Self::Attenuated => "Attenuated maximum",
            Self::PeakChannel => "Peak channel colour",
            Self::MeanChannel => "Mean channel colour",
            Self::Emission => "Emission-absorption",
            Self::Isosurface => "Isosurface",
            Self::Shells => "Translucent shells",
            Self::ClippedMoment => "Clipped moment 0",
        }
    }

    fn about(self) -> &'static str {
        match self {
            Self::Maximum => "The brightest voxel along each ray",
            Self::Average => "The mean along each ray",
            Self::DepthCued => "Maximum, darker the further away its voxel is",
            Self::Attenuated => {
                "Maximum of each voxel dimmed by the emission in front of it, as napari's attenuated MIP. \
                 Voxels under the level count as empty"
            }
            Self::PeakChannel => {
                "Hue from the channel of the ray's peak, brightness from the peak (moment 9 over moment 8). \
                 Peaks under the level are left out. A rainbow colour map such as Turbo suits it"
            }
            Self::MeanChannel => {
                "Hue from the intensity-weighted mean channel, brightness from the clipped moment 0 \
                 (moment 1 over moment 0)"
            }
            Self::Emission => {
                "Emission-absorption, from the level up, through MIDA to the maximum. A colour map that \
                 isn't dark at its low end suits it"
            }
            Self::Isosurface => {
                "The first surface at the level, lit, with ambient occlusion and a shadow"
            }
            Self::Shells => "Translucent surfaces at 1, 2 and 4 times the level",
            Self::ClippedMoment => {
                "The mean along each ray, voxels under the level counting as zero"
            }
        }
    }

    /// The shader def compiling the mode, None for `VolumeParams::projection_mode`'s two.
    fn shader_def(self) -> Option<&'static str> {
        match self {
            Self::Maximum | Self::Average => None,
            Self::DepthCued => Some("MODE_DEPTH_CUED"),
            Self::Attenuated => Some("MODE_ATTENUATED"),
            Self::PeakChannel => Some("MODE_PEAK_CHANNEL"),
            Self::MeanChannel => Some("MODE_MEAN_CHANNEL"),
            Self::Emission => Some("MODE_EMISSION"),
            Self::Isosurface => Some("MODE_ISOSURFACE"),
            Self::Shells => Some("MODE_SHELLS"),
            Self::ClippedMoment => Some("MODE_CLIPPED_MOMENT"),
        }
    }

    /// `at`, the level, where the mode uses one: not Maximum, Average or depth cued, nor the clipped moments when a
    /// mask leaves what counts to it.
    fn level<T>(self, masked: bool, at: T) -> Option<T> {
        match self {
            Self::Maximum | Self::Average | Self::DepthCued => None,
            Self::ClippedMoment | Self::MeanChannel if masked => None,
            _ => Some(at),
        }
    }

    /// Maximum's 0 for those taking a ray's brightest voxel, which a max-downsampled texture suits, else Average's 1.
    pub fn projection_mode(self) -> u32 {
        match self {
            Self::Maximum | Self::DepthCued | Self::Attenuated | Self::PeakChannel => 0,
            _ => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModeSettings {
    pub mode: RenderMode,
    /// The level, in σ above the noise median, for the modes that use one.
    pub sigmas: f32,
    pub depth_cue: f32,
    /// The optical depth of the box's wider side at threshold Max.
    pub attenuation: f32,
    pub opacity: f32,
    /// -1 emission-absorption, 0 MIDA, 1 the maximum.
    pub mida: f32,
    pub snr_opacity: bool,
    pub lit: bool,
    /// Asinh is linear up to this, in σ.
    pub asinh_sigma: f32,
    pub processing: Processing,
    pub apply_mask: bool,
    /// How much of a voxel outside the mask shows, 0 to 1.
    pub outside_mask: f32,
}

impl Default for ModeSettings {
    fn default() -> Self {
        Self {
            mode: RenderMode::Maximum,
            sigmas: 5.0,
            depth_cue: 0.7,
            attenuation: 20.0,
            opacity: 100.0,
            mida: -1.0,
            snr_opacity: false,
            lit: false,
            asinh_sigma: 1.0,
            processing: Processing::default(),
            apply_mask: false,
            outside_mask: 0.25,
        }
    }
}

fn smoothing_name(smoothing: Smoothing) -> &'static str {
    match smoothing {
        Smoothing::Off => "None",
        Smoothing::Spectral => "Hanning",
        Smoothing::Spatial => "3D binomial",
    }
}

/// Field order must match `ModeParams` in volume.wesl.
#[derive(Clone, Copy, Debug, Default, PartialEq, ShaderType)]
pub struct ModeParams {
    pub noise_median: f32,
    pub noise_sigma: f32,
    pub sigmas: f32,
    /// The level in data units, f32::MIN for none.
    pub level: f32,
    pub depth_cue: f32,
    pub attenuation: f32,
    pub opacity: f32,
    pub mida: f32,
    pub asinh_sigma: f32,
    /// 1 when asinh's zero is the noise's likely maximum along the ray, not its median.
    pub ray_max: u32,
    /// A voxel outside the mask is pulled towards mask_zero, leaving mask_outside of its difference from it.
    pub mask_zero: f32,
    pub mask_outside: f32,
    /// Layers where the channel colours start and end, swapped so that they run with velocity.
    pub channel_from: f32,
    pub channel_to: f32,
}

/// The material's pipeline key: the mode and its variants, each pipeline compiling only its own code.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct VolumeKey {
    mode: RenderMode,
    asinh: bool,
    mask: bool,
    snr_opacity: bool,
    lit: bool,
}

impl From<&VolumeMaterial> for VolumeKey {
    fn from(material: &VolumeMaterial) -> Self {
        material.key
    }
}

impl VolumeKey {
    pub fn add_shader_defs(self, descriptor: &mut RenderPipelineDescriptor) {
        let Some(fragment) = descriptor.fragment.as_mut() else {
            return;
        };
        if let Some(def) = self.mode.shader_def() {
            fragment.shader_defs.extend(["CUSTOM".into(), def.into()]);
        }
        for (on, def) in [
            (self.asinh, "ASINH"),
            (self.mask, "MASK"),
            (self.snr_opacity, "SNR_OPACITY"),
            (self.lit, "LIT"),
        ] {
            if on {
                fragment.shader_defs.push(def.into());
            }
        }
    }
}

/// An R8Unorm 2D array texture of a mask, 1 inside, laid out as the cube texture so the shader samples both alike.
/// Without `mask` it's zeros until its parts are written in.
fn mask_image(dims: [usize; 3], mask: Option<Vec<u8>>) -> Image {
    let size = Extent3d {
        width: dims[0] as u32,
        height: dims[1] as u32,
        depth_or_array_layers: dims[2] as u32,
    };
    let (format, usage) = (TextureFormat::R8Unorm, RenderAssetUsages::RENDER_WORLD);
    let mut image = match mask {
        Some(mask) => Image::new(size, TextureDimension::D2, mask, format, usage),
        None => Image::new_uninit(size, TextureDimension::D2, format, usage),
    };
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    image
}

fn insert_no_mask(mut images: ResMut<Assets<Image>>) {
    let _ = images.insert(&NO_MASK, mask_image([1; 3], Some(vec![0])));
}

/// The mode's uniforms, mask and pipeline key, after main.rs's `update_material` has set the rest.
fn update_material(
    settings: Res<Settings>,
    state: Res<CubeState>,
    data: Res<Dataset>,
    extras: Res<Extras>,
    mut materials: ResMut<Assets<VolumeMaterial>>,
) {
    let Some(mut material) = materials.get_mut(&state.material) else {
        return;
    };
    let (p, m, info) = (&settings.params, &settings.modes, &data.info);
    let texture = state.shown();
    // Until the source has measured the texture, the noise taken at open.
    let opened = info.sigma.map(|sigma| Noise {
        median: 0.0,
        sigma: sigma as f32,
    });
    let noise = (extras.noise().or(opened)).unwrap_or(Noise {
        median: info.stats.mean as f32,
        sigma: (info.stats.std as f32).max(f32::MIN_POSITIVE),
    });
    let mask = extras.ready_mask().filter(|_| m.apply_mask);
    use RenderMode::*;
    let level = (m.mode.level(mask.is_some(), noise.at(m.sigmas))).unwrap_or(f32::MIN);
    let floor = view::floor_value(p);
    let layer = |channel: f32| channel / texture.factors[2] as f32 - texture.texels.min[2] as f32;
    let (channel_from, channel_to) = match extras.channels {
        Some((from, to)) => (layer(from), layer(to)),
        None => (0.0, texture.dims()[2] as f32),
    };
    material.modes = ModeParams {
        noise_median: noise.median,
        noise_sigma: noise.sigma,
        sigmas: m.sigmas,
        level,
        depth_cue: m.depth_cue,
        attenuation: m.attenuation,
        opacity: m.opacity,
        mida: m.mida,
        asinh_sigma: m.asinh_sigma,
        ray_max: u32::from(m.mode.projection_mode() == 0),
        mask_zero: match m.mode {
            Maximum | DepthCued | Attenuated => floor,
            _ => noise.median,
        },
        mask_outside: m.outside_mask,
        channel_from,
        channel_to,
    };
    material.mask = mask.cloned().unwrap_or(NO_MASK);
    let emission = m.mode == Emission;
    material.key = VolumeKey {
        mode: m.mode,
        asinh: p.scale_type == ASINH,
        mask: mask.is_some(),
        snr_opacity: emission && m.snr_opacity,
        lit: emission && m.lit,
    };
}

/// The planes holding the middle 98% of the flux, so the channel colours span the line rather than the band, and a few
/// noise peaks or artefacts away from it can't stretch them. None when there's none.
fn line_planes(line_flux: &[f32]) -> Option<(usize, usize)> {
    let total: f32 = line_flux.iter().sum();
    if !total.is_normal() {
        return None;
    }
    let (mut below, mut from, mut to) = (0.0, None, 0);
    for (z, &flux) in line_flux.iter().enumerate() {
        if from.is_none() && below + flux > 0.01 * total {
            from = Some(z);
        }
        if below < 0.99 * total {
            to = z + 1;
        }
        below += flux;
    }
    Some((from.unwrap_or(0), to))
}

/// The mode, its sliders, the σ level where it uses one, and the processing, which a server doesn't do. `masked` when a
/// mask is applied.
pub fn ui(
    ui: &mut egui::Ui,
    p: &mut VolumeParams,
    m: &mut ModeSettings,
    noise: Option<Noise>,
    masked: bool,
    remote: bool,
    unit: &str,
) {
    egui::ComboBox::from_label("Rendering mode")
        .selected_text(m.mode.name())
        .show_ui(ui, |ui| {
            for mode in RenderMode::ALL {
                ui.selectable_value(&mut m.mode, mode, mode.name())
                    .on_hover_text(mode.about());
            }
        })
        .response
        .on_hover_text(m.mode.about());
    p.projection_mode = m.mode.projection_mode();
    match m.mode {
        RenderMode::DepthCued => {
            ui.add(egui::Slider::new(&mut m.depth_cue, 0.0..=1.0).text("Depth cue"));
        }
        RenderMode::Attenuated => {
            ui.add(
                egui::Slider::new(&mut m.attenuation, 1.0..=200.0)
                    .logarithmic(true)
                    .text("Attenuation"),
            );
        }
        RenderMode::Emission => {
            ui.add(
                egui::Slider::new(&mut m.opacity, 1.0..=1000.0)
                    .logarithmic(true)
                    .text("Opacity"),
            );
            ui.add(egui::Slider::new(&mut m.mida, -1.0..=1.0).text("Emission .. Maximum"))
                .on_hover_text("-1 is emission-absorption, 0 MIDA, and 1 the maximum");
            ui.horizontal(|ui| {
                ui.checkbox(&mut m.snr_opacity, "Opacity from S/N")
                    .on_hover_text("Opacity on a log S/N ramp from the level to Max, not the colour map's position");
                ui.checkbox(&mut m.lit, "Lighting")
                    .on_hover_text("Lit from the gradient where it stands out from the noise");
            });
        }
        _ => {}
    }
    if let Some(level) = m.mode.level(masked, noise.map(|n| n.at(m.sigmas))) {
        let readout = level.map_or_else(|| MEASURING.into(), |level| format!("{level:.3e}{unit}"));
        sigma_row(ui, "Level", &mut m.sigmas, readout).on_hover_text(
            "Above the texture's noise median, in its noise σ: the isosurface, the shells, the clip, and the floor \
             for the channel colours, attenuation and emission",
        );
    }
    let processing = ui.add_enabled_ui(!remote, |ui| {
        ui.horizontal(|ui| {
            ui.checkbox(&mut m.processing.normalise, "Normalise noise")
                .on_hover_text("Each channel and spectrum scaled to the same noise, as SoFiA-2 does. Changes fluxes");
            egui::ComboBox::from_label("Smoothing")
                .selected_text(smoothing_name(m.processing.smoothing))
                .show_ui(ui, |ui| {
                    for smoothing in [Smoothing::Off, Smoothing::Spectral, Smoothing::Spatial] {
                        let name = smoothing_name(smoothing);
                        ui.selectable_value(&mut m.processing.smoothing, smoothing, name);
                    }
                });
        });
    });
    processing.response.on_disabled_hover_text(REMOTE);
}

/// Why a cube on a server has no processing or mask, and its level goes by the noise at open.
pub const REMOTE: &str = "Not available for a cube on a server yet, which sends its cubelets unprocessed. Levels in σ \
    use the cube's noise from when it was opened";

/// The mask's threshold in σ, making it and applying it. True when a mask is asked for.
pub fn mask_ui(
    ui: &mut egui::Ui,
    m: &mut ModeSettings,
    sigmas: &mut f32,
    noise: Option<Noise>,
    mask: Option<(f32, Option<usize>)>,
    unit: &str,
) -> bool {
    let readout = noise.map_or_else(|| MEASURING.into(), |n| format!("σ {:.3e}{unit}", n.sigma));
    sigma_row(ui, "Threshold", sigmas, readout).on_hover_text(
        "Each smoothed copy of the texture is clipped this far above its own noise median, in its own σ, as \
         SoFiA-2's finder does. The σ shown is the unsmoothed texture's. SoFiA-2's default is 5σ",
    );
    let mut generate = false;
    ui.horizontal(|ui| match mask {
        Some((sigmas, None)) => {
            ui.add_enabled(false, egui::Button::new(format!("Generating mask at {sigmas}σ")));
            ui.spinner();
        }
        mask => {
            if ui
                .button("Generate mask")
                .on_hover_text(
                    "Smooth and clip, as SoFiA-2's finder, without its reliability step. Made from the texture's \
                     level, so at its downsampling",
                )
                .clicked()
            {
                generate = true;
                m.apply_mask = true;
            }
            if let Some((sigmas, Some(voxels))) = mask {
                ui.label(format!("{voxels} voxels at {sigmas}σ"));
            }
        }
    });
    let ready = matches!(mask, Some((_, Some(_))));
    ui.horizontal(|ui| {
        ui.add_enabled(ready, egui::Checkbox::new(&mut m.apply_mask, "Apply mask"));
        ui.add_enabled(
            ready && m.apply_mask,
            egui::Slider::new(&mut m.outside_mask, 0.0..=1.0).text("outside it"),
        );
    });
    generate
}

/// A row picking one of SIGMA_LEVELS, then `readout`.
fn sigma_row(ui: &mut egui::Ui, label: &str, sigmas: &mut f32, readout: String) -> egui::Response {
    ui.horizontal(|ui| {
        ui.label(label);
        for level in SIGMA_LEVELS {
            if ui
                .selectable_label(*sigmas == level, format!("{level}σ"))
                .clicked()
            {
                *sigmas = level;
            }
        }
        ui.label(readout);
    })
    .response
}
