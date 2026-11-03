//! The egui panel: cube information, cursor readout, crop buttons and rendering controls.

use bevy::{
    diagnostic::{DiagnosticPath, DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
};
use bevy_egui::{EguiContexts, egui};

use crate::{
    CubeState, Dataset, Settings,
    config::{COLOR_MAPS, GAMMA, JITTER, RAYMARCHING_STEPS, SCALING_TYPES},
    fits::{PERCENTILES, Region},
    selection::{CropRequest, Hover},
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
            egui::CollapsingHeader::new("Cube")
                .default_open(true)
                .show(ui, |ui| {
                    cube_section(ui, &data, &state, &hover, &mut crop_requests)
                });
            if let Some(wcs) = &data.wcs {
                egui::CollapsingHeader::new("WCS").show(ui, |ui| {
                    for line in wcs.summary() {
                        ui.label(line);
                    }
                });
            }
            egui::CollapsingHeader::new("Rendering")
                .default_open(true)
                .show(ui, |ui| rendering_section(ui, &data, &mut settings));
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
) {
    let info = &data.info;
    let stats = info.stats;
    let gb = (info.dims.iter().product::<usize>() * size_of::<f32>()) as f64 / 1e9;
    let unit = info.unit().map_or(String::new(), |u| format!(" {u}"));
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
        row("Non-finite voxels", stats.non_finite.to_string());
        row("Load time", format!("{:.2} s", data.load_seconds));
        let texture = state.texture();
        let [fx, fy, fz] = texture.factors;
        let resolution = if texture.factors == [1; 3] {
            "full resolution".to_string()
        } else {
            format!("downsampled {fx} x {fy} x {fz}")
        };
        let mb = (texture.dims.iter().product::<usize>() * size_of::<f32>()) as f64 / 1e6;
        row(
            "Texture",
            format!("{} ({mb:.0} MB), {resolution}", format_dims(texture.dims)),
        );
        row(
            "Showing",
            state
                .crop
                .as_ref()
                .map_or("entire cube".to_string(), |crop| format_region(crop.region)),
        );
        if let Some(selection) = state.selection {
            row("Selection", format_region(selection));
        }
        // Until the source answers, the readout stays on the last voxel it answered for.
        if let Some((v, value)) = hover.voxel.and(hover.value) {
            row(
                "Cursor",
                format!(
                    "({}, {}, {}) = {value:.4e}{unit}",
                    v[0] + 1,
                    v[1] + 1,
                    v[2] + 1
                ),
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
        if button(ui, state.crop.is_some(), "Reset crop").clicked() {
            crop_requests.write(CropRequest::Reset);
        }
    });
}

fn rendering_section(ui: &mut egui::Ui, data: &Dataset, settings: &mut ResMut<Settings>) {
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
    ui.horizontal(|ui| {
        ui.radio_value(&mut p.projection_mode, 0, "Maximum");
        ui.radio_value(&mut p.projection_mode, 1, "Average");
        ui.label("intensity projection");
    });
    ui.add(egui::Slider::new(&mut p.jitter, JITTER).text("Jitter"));
    ui.add(egui::Slider::new(&mut p.max_steps, RAYMARCHING_STEPS).text("Ray-marching steps"));
    if data.wcs.is_some() {
        ui.checkbox(&mut s.wcs_axes, "WCS axes");
    }
    settings.set_if_neq(s);
}
