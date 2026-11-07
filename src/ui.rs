//! The egui panel: cube information, cursor readout, crop buttons and rendering controls.

use bevy::{
    diagnostic::{DiagnosticPath, DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
};
use bevy_egui::{EguiContexts, egui};

use crate::{
    CubeState, Dataset, Settings,
    config::{ASINH, COLOR_MAPS, GAMMA, JITTER, RAYMARCHING_STEPS, SCALING_TYPES, ZScale},
    fits::{Noise, PERCENTILES, Region},
    modes::{self, Extras, GenerateMask},
    selection::{CropRequest, Hover},
    shape::z_scale_range,
};

const HELP: &str = "Drag: orbit. Right or middle drag: pan. Scroll: zoom. Shift+drag: select region. \
    Drag a blue arrow or corner: resize. C or Enter: crop to selection. R: reset crop. Esc: clear selection.";

pub fn format_dims(d: [usize; 3]) -> String {
    format!("{} x {} x {}", d[0], d[1], d[2])
}

/// 1-based inclusive ranges, as iDaVIE displays them.
fn format_region(Region { min, max }: Region) -> String {
    format!(
        "x {}..{}, y {}..{}, z {}..{}",
        min[0] + 1,
        max[0] + 1,
        min[1] + 1,
        max[1] + 1,
        min[2] + 1,
        max[2] + 1
    )
}

pub fn ui(
    mut contexts: EguiContexts,
    data: Option<Res<Dataset>>,
    state: Option<Res<CubeState>>,
    settings: Option<ResMut<Settings>>,
    hover: Res<Hover>,
    diagnostics: Res<DiagnosticsStore>,
    windows: Query<&Window>,
    mut crop_requests: MessageWriter<CropRequest>,
    extras: Res<Extras>,
    mut generate: MessageWriter<GenerateMask>,
    // The mask's threshold, which only the mask it asks for takes.
    mut mask_sigmas: Local<Option<f32>>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    egui::Window::new("iDaVIE volume renderer")
        .default_width(420.0)
        .show(ctx, |ui| {
            let smoothed = |path: &DiagnosticPath| {
                diagnostics
                    .get(path)
                    .and_then(|d| d.smoothed())
                    .unwrap_or(0.0)
            };
            let fps = smoothed(&FrameTimeDiagnosticsPlugin::FPS);
            let frame_ms = smoothed(&FrameTimeDiagnosticsPlugin::FRAME_TIME);
            let resolution = windows.single().map_or(String::new(), |w| {
                format!(", {} x {} px", w.physical_width(), w.physical_height())
            });
            ui.label(format!("{fps:.0} fps, {frame_ms:.2} ms/frame{resolution}"));

            let (Some(data), Some(state), Some(mut settings)) = (data, state, settings) else {
                ui.label("Choose a FITS file to load");
                return;
            };
            let (noise, unit) = (extras.noise(), data.info.unit());
            let unit = unit.map_or(String::new(), |u| format!(" {u}"));
            egui::CollapsingHeader::new("Cube")
                .default_open(true)
                .show(ui, |ui| {
                    let crops = &mut crop_requests;
                    cube_section(ui, &data, &state, &hover, crops, noise, &unit)
                });
            if let Some(wcs) = &data.wcs {
                egui::CollapsingHeader::new("WCS").show(ui, |ui| {
                    for line in wcs.summary() {
                        ui.label(line);
                    }
                });
            }
            let mask = extras.mask();
            let masked = settings.modes.apply_mask && mask.is_some_and(|(_, n)| n.is_some());
            egui::CollapsingHeader::new("Rendering")
                .default_open(true)
                .show(ui, |ui| {
                    rendering_section(ui, &data, &mut settings, noise, masked, &unit)
                });
            egui::CollapsingHeader::new("Mask")
                .default_open(true)
                .show(ui, |ui| {
                    if !data.source.has_cube() {
                        ui.label(modes::REMOTE);
                        return;
                    }
                    let sigmas = mask_sigmas.get_or_insert(modes::MASK_SIGMAS);
                    let mut s = settings.clone();
                    if modes::mask_ui(ui, &mut s.modes, sigmas, noise, mask, &unit) {
                        generate.write(GenerateMask(*sigmas));
                    }
                    settings.set_if_neq(s);
                });
            ui.separator();
            ui.small(HELP);
        });
    Ok(())
}

fn cube_section(
    ui: &mut egui::Ui,
    data: &Dataset,
    state: &CubeState,
    hover: &Hover,
    crop_requests: &mut MessageWriter<CropRequest>,
    noise: Option<Noise>,
    unit: &str,
) {
    let info = &data.info;
    let stats = info.stats;
    let gb = (info.dims.iter().product::<usize>() * size_of::<f32>()) as f64 / 1e9;
    let cropped = state.cropped();
    egui::Grid::new("cube").num_columns(2).show(ui, |ui| {
        let mut row = |name: &str, value: String| {
            ui.label(name);
            ui.label(value);
            ui.end_row();
        };
        row("File", format!("{} (HDU {})", data.name, info.hdu));
        row(
            "Dimensions",
            format!("{} ({gb:.2} GB)", format_dims(info.dims)),
        );
        if let Some(planes) = info.other_planes() {
            row("Plane", planes);
        }
        row(
            "Range",
            format!("{:.4e} .. {:.4e}{unit}", stats.min, stats.max),
        );
        row(
            "Mean, std",
            format!("{:.4e}, {:.4e}", stats.mean, stats.std),
        );
        // The shown texture's once it's measured, which the σ settings go by, and until then the cube's at open.
        if let Some(sigma) = noise.map(|n| f64::from(n.sigma)).or(info.sigma) {
            row("Noise σ", format!("{sigma:.4e}{unit}"));
        }
        row("Non-finite voxels", stats.non_finite.to_string());
        row("Load time", format!("{:.2} s", data.load_seconds));
        let texture = state.shown();
        let [fx, fy, fz] = texture.factors;
        let resolution = if texture.factors == [1; 3] {
            "full resolution".to_string()
        } else {
            format!("downsampled {fx} x {fy} x {fz}")
        };
        let dims = texture.dims();
        let mb = (dims.iter().product::<usize>() * size_of::<f32>()) as f64 / 1e6;
        row(
            "Texture",
            format!("{} ({mb:.0} MB), {resolution}", format_dims(dims)),
        );
        row(
            "Showing",
            if cropped {
                format_region(state.visible)
            } else {
                "entire cube".to_string()
            },
        );
        if let Some(selection) = state.selection {
            row("Selection", format_region(selection));
        }
        // Until the source answers, the readout stays on the last voxel it answered for.
        if let Some((v, value)) = hover.voxel.and(hover.value) {
            let value = match data.approximate {
                Some(within) => format!("≈ {value:.4e} ± {within:.1e}{unit}"),
                None => format!("= {value:.4e}{unit}"),
            };
            row(
                "Cursor",
                format!("({}, {}, {}) {value}", v[0] + 1, v[1] + 1, v[2] + 1),
            );
            let grid = v.map(|i| i as f64 + 1.0);
            if let Some(world) = data.wcs.as_ref().and_then(|w| w.grid_to_world(grid)) {
                row("", world.join(", "));
            }
        }
    });
    ui.horizontal(|ui| {
        let button =
            |ui: &mut egui::Ui, enabled, text| ui.add_enabled(enabled, egui::Button::new(text));
        if button(ui, state.selection.is_some(), "Crop to selection").clicked() {
            crop_requests.write(CropRequest::ToSelection);
        }
        if button(ui, cropped, "Reset crop").clicked() {
            crop_requests.write(CropRequest::Reset);
        }
        // Only a server's or a baked cube's cubelets are cached. In the browser a baked cube has no Server panel, so
        // the button is here as well.
        #[cfg(target_arch = "wasm32")]
        if !data.source.has_cube()
            && (ui.button("Clear the cache"))
                .on_hover_text(crate::remote::CLEAR_CACHE)
                .clicked()
        {
            crate::web::clear_cache();
        }
    });
}

/// Data on its way, at the top right where nothing else is: a whole file downloading, or the shown cube's cubelets
/// that have come out of those asked for, with those that came from the cache and the bytes the rest took.
pub fn progress(mut contexts: EguiContexts, data: Option<Res<Dataset>>) -> Result {
    let mb = |bytes: f64| format!("{:.1}", bytes / 1e6);
    #[cfg(target_arch = "wasm32")]
    let download = crate::web::download();
    #[cfg(not(target_arch = "wasm32"))]
    let download = None::<(f64, f64)>;
    let (text, fraction) = if let Some((got, total)) = download {
        if total > 0.0 {
            (
                format!("downloading {} / {} MB", mb(got), mb(total)),
                got / total,
            )
        } else {
            (format!("downloading {} MB", mb(got)), 0.0)
        }
    } else if let Some((data, p)) = data.zip(crate::source::progress()) {
        let mut text = format!("cubelets {} / {}", p.done, p.asked);
        // Only a server's or a baked cube's are cached.
        let cached = p.done.saturating_sub(p.fetched);
        if !data.source.has_cube() && cached > 0 {
            text += &format!(" ({cached} cached)");
        }
        let got = mb(p.bytes as f64);
        match p.to_read.map(|to_read| p.bytes + to_read) {
            Some(total) if total > 0 => text += &format!("\n{got} / {} MB", mb(total as f64)),
            None if p.bytes > 0 => text += &format!("\n{got} MB"),
            _ => {}
        }
        (text, p.done as f64 / p.asked as f64)
    } else {
        return Ok(());
    };
    egui::Area::new(egui::Id::new("progress"))
        .anchor(egui::Align2::RIGHT_TOP, [-10.0, 10.0])
        .order(egui::Order::Foreground)
        .interactable(false)
        .show(contexts.ctx_mut()?, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_width(200.0);
                ui.label(text);
                ui.add(egui::ProgressBar::new(fraction as f32).desired_height(4.0));
            });
        });
    Ok(())
}

fn rendering_section(
    ui: &mut egui::Ui,
    data: &Dataset,
    settings: &mut ResMut<Settings>,
    noise: Option<Noise>,
    masked: bool,
    unit: &str,
) {
    let mut s = settings.clone();
    let p = &mut s.params;
    egui::ComboBox::from_label("Colour map")
        .selected_text(COLOR_MAPS[p.color_map_index as usize])
        .show_ui(ui, |ui| {
            for (i, name) in COLOR_MAPS.iter().enumerate() {
                ui.selectable_value(&mut p.color_map_index, i as u32, *name);
            }
        });
    egui::ComboBox::from_label("Scaling")
        .selected_text(SCALING_TYPES[p.scale_type as usize])
        .show_ui(ui, |ui| {
            for (i, name) in SCALING_TYPES.iter().enumerate() {
                ui.selectable_value(&mut p.scale_type, i as u32, *name);
            }
        });
    if p.scale_type == GAMMA {
        ui.add(egui::Slider::new(&mut p.scale_gamma, 0.0..=5.0).text("Gamma"));
    }
    if p.scale_type == ASINH {
        ui.add(
            egui::Slider::new(&mut s.modes.asinh_sigma, 0.1..=20.0)
                .logarithmic(true)
                .text("Softening (σ)"),
        )
        .on_hover_text(
            "Zero at the noise median, or for the maximum modes at the brightest the noise alone likely reaches \
             along the ray; linear up to the softening, logarithmic above",
        );
    }
    ui.horizontal_wrapped(|ui| {
        for percent in PERCENTILES {
            let clip = data.info.histogram.clip(percent);
            let button = ui
                .selectable_label(
                    (p.threshold_min, p.threshold_max) == clip,
                    format!("{percent}%"),
                )
                .on_hover_text(format!(
                    "Thresholds that keep the middle {percent}% of the whole cube's voxels"
                ));
            if button.clicked() {
                (p.threshold_min, p.threshold_max) = clip;
            }
        }
    });
    let value = |t: f32| p.scale_min + t * (p.scale_max - p.scale_min);
    let (min_value, max_value) = (value(p.threshold_min), value(p.threshold_max));
    // Each threshold is bounded by the other, as in iDaVIE, so Min can't pass Max.
    let (min, max) = (p.threshold_min, p.threshold_max);
    ui.add(
        egui::Slider::new(&mut p.threshold_min, 0.0..=max).text(format!("Min ({min_value:.3e})")),
    );
    ui.add(
        egui::Slider::new(&mut p.threshold_max, min..=1.0).text(format!("Max ({max_value:.3e})")),
    );
    modes::ui(
        ui,
        p,
        &mut s.modes,
        noise,
        masked,
        !data.source.has_cube(),
        unit,
    );
    ui.add(egui::Slider::new(&mut p.jitter, JITTER).text("Jitter"));
    ui.add(egui::Slider::new(&mut p.max_steps, RAYMARCHING_STEPS).text("Ray-marching steps"));
    z_scale_row(ui, data.info.dims, &mut s.z_scale);
    if data.wcs.is_some() {
        ui.checkbox(&mut s.wcs_axes, "WCS axes");
    }
    settings.set_if_neq(s);
}

/// The z axis's scale relative to cubic voxels: a preset, or any value in `z_scale_range`.
fn z_scale_row(ui: &mut egui::Ui, dims: [usize; 3], z_scale: &mut f32) {
    ui.horizontal(|ui| {
        ui.label("Z scale")
            .on_hover_text("How deep the box is drawn, relative to cubic voxels");
        let presets = [
            (
                ZScale::Cube,
                "Cube",
                "The whole cube as deep as its wider side",
            ),
            (
                ZScale::Voxels,
                "Cubic voxels",
                "Each voxel as deep as it's wide",
            ),
        ];
        // Only the first, when a cube's presets are the same.
        let current = presets
            .iter()
            .position(|(preset, ..)| preset.value(dims) == *z_scale);
        for (i, (preset, name, about)) in presets.into_iter().enumerate() {
            if ui
                .selectable_label(current == Some(i), name)
                .on_hover_text(about)
                .clicked()
            {
                *z_scale = preset.value(dims);
            }
        }
        // A percent of the value per pixel, so a drag goes as far at 0.1 as at 100.
        let speed = *z_scale * 0.01;
        ui.add(
            egui::DragValue::new(z_scale)
                .range(z_scale_range(dims))
                .speed(speed),
        );
    });
}
