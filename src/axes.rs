//! WCS ticks and labels on the edges of the visible box.

use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};

use crate::{
    CubeState, Dataset, Settings, UNITY_FROM_LOCAL, camera::Orbit, fits::Region,
    outline::edge_ends, region_bounds, wcs,
};

/// WCS ticks for the 12 edges of the visible box (see `edge_ends`), recomputed when it changes.
#[derive(Resource)]
pub struct AxisAnnotation {
    visible: Region,
    edges: Vec<Vec<wcs::Tick>>,
    titles: [Option<String>; 3],
}

pub fn update_axes(
    mut commands: Commands,
    data: Res<Dataset>,
    state: Res<CubeState>,
    axes: Option<Res<AxisAnnotation>>,
) {
    let visible = state.visible;
    if !data.is_changed() && axes.map_or(data.wcs.is_none(), |a| a.visible == visible) {
        return;
    }
    let Some(wcs) = &data.wcs else {
        commands.remove_resource::<AxisAnnotation>();
        return;
    };
    // GRID coordinates put voxel i's centre at i + 1, so the box spans min + 0.5 to max + 1.5.
    let extent = visible.extent();
    let grid = |u: [f32; 3]| -> [f64; 3] {
        std::array::from_fn(|i| visible.min[i] as f64 + 0.5 + f64::from(u[i]) * extent[i] as f64)
    };
    let edges = (0..12)
        .map(|e| {
            let (start, end) = edge_ends(e);
            wcs.edge_ticks(e / 4 + 1, grid(start), grid(end))
        })
        .collect();
    let titles = std::array::from_fn(|i| wcs.axis_title(i + 1));
    commands.insert_resource(AxisAnnotation {
        visible,
        edges,
        titles,
    });
}

/// Ticks on the silhouette edge of each axis (so the labels sit outside the volume), labelled with egui.
pub fn draw_axes(
    mut contexts: EguiContexts,
    mut gizmos: Gizmos,
    data: Res<Dataset>,
    settings: Res<Settings>,
    axes: Res<AxisAnnotation>,
    camera: Single<(&Camera, &Transform), With<Orbit>>,
) -> Result {
    if !settings.wcs_axes {
        return Ok(());
    }
    let (camera, transform) = *camera;
    // The local transform is already this frame's orbit; GlobalTransform may not be propagated yet.
    let view = GlobalTransform::from(*transform);
    let (lo, hi) = region_bounds(axes.visible, data.info.dims);
    let local = |u: [f32; 3]| (lo + (hi - lo) * Vec3::from_array(u)) * UNITY_FROM_LOCAL;
    let screen = |p: Vec3| camera.world_to_viewport(&view, p).ok();
    let centre = local([0.5; 3]);
    let Some(centre_on_screen) = screen(centre) else {
        return Ok(());
    };
    let tick_length = 0.03 * (hi - lo).max_element();
    let painter = contexts
        .ctx_mut()?
        .layer_painter(egui::LayerId::background());
    // Labels that would overlap one already drawn are skipped, which thins out foreshortened edges.
    let mut placed: Vec<egui::Rect> = Vec::new();
    let mut text = |at: Vec3, from: Vec3, label: &str, size: f32| {
        let (Some(at), Some(from)) = (screen(at), screen(from)) else {
            return;
        };
        let away = at - from;
        let align = match (away.x.abs() > away.y.abs(), away.x > 0.0, away.y > 0.0) {
            (true, true, _) => egui::Align2::LEFT_CENTER,
            (true, false, _) => egui::Align2::RIGHT_CENTER,
            (false, _, true) => egui::Align2::CENTER_TOP,
            (false, _, false) => egui::Align2::CENTER_BOTTOM,
        };
        let color = egui::Color32::from_gray(220);
        let galley =
            painter.layout_no_wrap(label.to_string(), egui::FontId::proportional(size), color);
        let rect = align.anchor_size(egui::pos2(at.x, at.y), galley.size());
        if placed.iter().any(|r| r.intersects(rect.expand(2.0))) {
            return;
        }
        placed.push(rect);
        painter.galley(rect.min, galley, color);
    };

    let mut titles = Vec::new();
    for axis in 0..3 {
        let edge_on_screen = |e: usize| {
            let (start, end) = edge_ends(e);
            let mid = std::array::from_fn(|i| (start[i] + end[i]) / 2.0);
            screen(local(mid)).map_or(f32::MIN, |p| p.distance(centre_on_screen) + 0.001 * p.y)
        };
        let e = (axis * 4..axis * 4 + 4)
            .max_by(|a, b| edge_on_screen(*a).total_cmp(&edge_on_screen(*b)))
            .unwrap();
        let (start, end) = edge_ends(e);
        let (start, end) = (local(start), local(end));
        let mid = (start + end) / 2.0;
        let along = (end - start).normalize();
        let outward = {
            let d = mid - centre;
            (d - along * d.dot(along)).normalize_or_zero()
        };
        for tick in &axes.edges[e] {
            let p = start.lerp(end, tick.at);
            gizmos.line(p, p + outward * tick_length, Color::WHITE);
            text(p + outward * tick_length * 1.5, p, &tick.label, 12.0);
        }
        if let Some(title) = &axes.titles[axis] {
            titles.push((mid + outward * tick_length * 5.0, mid, title));
        }
    }
    for (at, from, title) in titles {
        text(at, from, title, 14.0);
    }
    Ok(())
}
