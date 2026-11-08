// Port of iDaVIE's IDIA/BasicVolume shader (Assets/Shaders/Volumes/BasicVolume.cginc); mask modes, vignette and foveation omitted.
import bevy_pbr::render::mesh_functions::{get_world_from_local, get_local_from_world, mesh_position_local_to_clip};
import bevy_pbr::render::mesh_view_bindings::{view, globals};
import bevy_pbr::render::view_transformations::{frag_coord_to_ndc, position_ndc_to_world};
@if(DEPTH_PREPASS)
import bevy_pbr::prepass::utils::prepass_depth;

struct VolumeParams {
    slice_min: vec3<f32>,
    threshold_min: f32,
    slice_max: vec3<f32>,
    threshold_max: f32,
    highlight_min: vec3<f32>,
    scale_min: f32,
    highlight_max: vec3<f32>,
    scale_max: f32,
    jitter: f32,
    color_map_index: u32,
    num_color_maps: u32,
    scale_type: u32,
    scale_alpha: f32,
    scale_gamma: f32,
    scale_bias: f32,
    scale_contrast: f32,
    max_steps: u32,
    highlight_saturate_factor: f32,
    projection_mode: u32,
    // From the texture shown (volume.rs VolumeParams::set_texture).
    step_length: f32,
    step_count: f32,
    // A grid cell's size and the texture's, z in layers.
    cell_size: vec3<f32>,
    layers: f32,
    grid_cells: vec3<u32>,
    // 1 when the texture has a NaN.
    has_nan: u32,
    // The texture's box, which the clip box (slice_min, slice_max) lies in, and where the grid's first cell
    // starts before the texture, in texture coordinates with z in layers.
    texture_min: vec3<f32>,
    texture_max: vec3<f32>,
    grid_offset: vec3<f32>,
    // The whole cube's size in the world, which the z scale stretches.
    box_size: vec3<f32>,
};

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> params: VolumeParams;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var data_cube: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(2) var data_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(3) var color_map: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(4) var color_map_sampler: sampler;
// volume.rs grid_buffer: a cubelet to an entry, 1 if it holds a NaN and then its max (-inf for nothing but NaN).
@group(constants::MATERIAL_BIND_GROUP) @binding(5) var<storage, read> grid: array<vec2<u32>>;

const LOG: u32 = 1u;
const SQRT: u32 = 2u;
const SQUARE: u32 = 3u;
const POWER: u32 = 4u;
const GAMMA: u32 = 5u;

// The projection_mode of ui.rs's Average button; 0 is Maximum.
const AVERAGE: u32 = 1u;

const F32_MAX: f32 = 3.402823466e+38;

// Unity object space is left-handed: same x and y as Bevy's local space, z mirrored.
const UNITY_FROM_LOCAL = vec3<f32>(1.0, 1.0, -1.0);

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) ray_origin: vec3<f32>,
    @location(1) ray_direction: vec3<f32>,
    @location(2) @interpolate(flat) instance_index: u32,
};

@vertex
fn vertex(v: Vertex) -> VertexOutput {
    var out: VertexOutput;
    out.position = mesh_position_local_to_clip(get_world_from_local(v.instance_index), vec4(v.position, 1.0));
    let camera_local = (get_local_from_world(v.instance_index) * vec4(view.world_position, 1.0)).xyz;
    // Unity: direction = -ObjSpaceViewDir(vertex), origin = vertex - direction (the camera)
    out.ray_direction = (v.position - camera_local) * UNITY_FROM_LOCAL;
    out.ray_origin = camera_local * UNITY_FROM_LOCAL;
    out.instance_index = v.instance_index;
    return out;
}

fn intersect_box(origin: vec3<f32>, direction: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let inv_r = 1.0 / direction;
    let tbot = inv_r * (box_min - origin);
    let ttop = inv_r * (box_max - origin);
    let tmin = min(ttop, tbot);
    let tmax = max(ttop, tbot);
    let largest_tmin = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
    let smallest_tmax = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));
    return vec2(largest_tmin, smallest_tmax);
}

fn nrand(uv: vec2<f32>) -> f32 {
    return fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453);
}

// Bit test rather than iDaVIE's comparison trick, which WGSL compilers may fold away.
fn is_nan(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u;
}

fn in_box(position: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> bool {
    return all(position >= box_min) && all(position <= box_max);
}

// data_sampler does iDaVIE's point filtering and Repeat wrapping in x and y (volume.rs empty_cube_image), and
// fract wraps z the same way.
fn sample_cube(position: vec3<f32>) -> f32 {
    return fetch(vec3(position.xy, fract(position.z) * params.layers));
}

// z in layers. With MASK, the masked fetch below.
@if(!MASK)
fn fetch(a: vec3<f32>) -> f32 {
    return textureSampleLevel(data_cube, data_sampler, a.xy, i32(a.z), 0.0).r;
}

// The max, and the max in the selection.
fn max_sample(top: vec2<f32>, value: f32, position: vec3<f32>, h_min: vec3<f32>, h_max: vec3<f32>) -> vec2<f32> {
    if is_nan(value) {
        return top;
    }
    return vec2(max(top.x, value), select(top.y, max(top.y, value), in_box(position, h_min, h_max)));
}

// The sum, the count, and the count in the selection.
fn add_sample(acc: vec3<f32>, value: f32, position: vec3<f32>, h_min: vec3<f32>, h_max: vec3<f32>) -> vec3<f32> {
    if is_nan(value) {
        return acc;
    }
    return acc + vec3(value, 1.0, select(0.0, 1.0, in_box(position, h_min, h_max)));
}

// `nan` is a literal at each call, so each call compiles to its own loop.
fn lean_max(value: f32, nan: bool) -> f32 {
    return select(value, -F32_MAX, nan && is_nan(value));
}

fn lean_add(value: f32, nan: bool) -> vec2<f32> {
    return select(vec2(value, 1.0), vec2(0.0), nan && is_nan(value));
}

// The max of `count` samples from `a` on, `da` apart, with the tail in groups of 4, 2 and 1 rather than up to
// seven fetches in turn.
fn segment_max(a_in: vec3<f32>, da: vec3<f32>, count: i32, nan: bool) -> f32 {
    var a = a_in;
    var m = -F32_MAX;
    var k = 0;
    for (; k + 8 <= count; k += 8) {
        m = max(m, max(max(max(lean_max(fetch(a), nan), lean_max(fetch(a + da), nan)),
            max(lean_max(fetch(a + 2.0 * da), nan), lean_max(fetch(a + 3.0 * da), nan))),
            max(max(lean_max(fetch(a + 4.0 * da), nan), lean_max(fetch(a + 5.0 * da), nan)),
            max(lean_max(fetch(a + 6.0 * da), nan), lean_max(fetch(a + 7.0 * da), nan)))));
        a += 8.0 * da;
    }
    let rest = count - k;
    if (rest & 4) != 0 {
        m = max(m, max(max(lean_max(fetch(a), nan), lean_max(fetch(a + da), nan)),
            max(lean_max(fetch(a + 2.0 * da), nan), lean_max(fetch(a + 3.0 * da), nan))));
        a += 4.0 * da;
    }
    if (rest & 2) != 0 {
        m = max(m, max(lean_max(fetch(a), nan), lean_max(fetch(a + da), nan)));
        a += 2.0 * da;
    }
    if (rest & 1) != 0 {
        m = max(m, lean_max(fetch(a), nan));
    }
    return m;
}

// The sum and count of the valid samples among `count` from `a` on, as segment_max.
fn segment_sum(a_in: vec3<f32>, da: vec3<f32>, count: i32, nan: bool) -> vec2<f32> {
    var a = a_in;
    var acc = vec2(0.0);
    var k = 0;
    for (; k + 4 <= count; k += 4) {
        acc += (lean_add(fetch(a), nan) + lean_add(fetch(a + da), nan))
            + (lean_add(fetch(a + 2.0 * da), nan) + lean_add(fetch(a + 3.0 * da), nan));
        a += 4.0 * da;
    }
    if ((count - k) & 2) != 0 {
        acc += lean_add(fetch(a), nan) + lean_add(fetch(a + da), nan);
        a += 2.0 * da;
    }
    if ((count - k) & 1) != 0 {
        acc += lean_add(fetch(a), nan);
    }
    return acc;
}

fn grid_entry(cell: vec3<u32>) -> vec2<u32> {
    return grid[cell.x + params.grid_cells.x * (cell.y + params.grid_cells.y * cell.z)];
}

// The grid cell that sample k is in, of `steps` from `start` `da` apart, z in layers: its box, the step past it, and
// its max and NaN flag. The grid starts grid_offset before the texture.
struct Span {
    lo: vec3<f32>,
    hi: vec3<f32>,
    end: i32,
    max: f32,
    nan: bool,
};

fn span_at(start: vec3<f32>, da: vec3<f32>, inv_da: vec3<f32>, steps: i32, k: i32) -> Span {
    let a = start + f32(k) * da;
    let go = params.grid_offset;
    let cell = clamp(floor((a + go) / params.cell_size), vec3(0.0), vec3<f32>(params.grid_cells - 1u));
    let lo = cell * params.cell_size - go;
    let hi = min(lo + params.cell_size, texture_box());
    let t = select(vec3(F32_MAX), (select(lo, hi, da > vec3(0.0)) - start) * inv_da, da != vec3(0.0));
    let end = clamp(i32(ceil(min(min(min(t.x, t.y), t.z), f32(steps)))), k + 1, steps);
    let entry = grid_entry(vec3<u32>(cell));
    return Span(lo, hi, end, bitcast<f32>(entry.y), entry.x != 0u);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Distance to opaque geometry along the ray (iDaVIE reads _CameraDepthTexture), here the outline lines.
    var opaque_depth = F32_MAX;
    @if(DEPTH_PREPASS) {
        let depth = prepass_depth(in.position, 0u);
        // Reverse-Z: 0 is the far plane, so nothing opaque at this pixel.
        if depth > 0.0 {
            let world = position_ndc_to_world(vec3(frag_coord_to_ndc(in.position).xy, depth));
            let local = (get_local_from_world(in.instance_index) * vec4(world, 1.0)).xyz * UNITY_FROM_LOCAL;
            opaque_depth = length(local - in.ray_origin);
        }
    }

    let step_length = params.step_length;
    let max_steps = params.step_count;

    let direction = normalize(in.ray_direction);
    let t = intersect_box(in.ray_origin, direction, params.slice_min, params.slice_max);
    // Early exit of pixels missing the bounding box or occluded by opaque objects. Within a step counts as
    // occluded, since the last sample would land a step in front of the box and wrap round to the far side,
    // tinting the front edges of the outline. Precise has no steps, so within a voxel along the ray does.
    var margin = step_length;
    @if(VOXELS) {
        let rate = abs(direction / (params.texture_max - params.texture_min) * texels());
        margin = 1.0 / max(rate.x, max(rate.y, rate.z));
    }
    if t.x > t.y || t.y < 0.0 || opaque_depth < t.x + margin {
        return vec4(0.0);
    }
    let t_near = max(0.0, t.x);
    let t_far = min(t.y, opaque_depth);

    let p_near = in.ray_origin + direction * t_near + 0.5;
    let p_far = in.ray_origin + direction * t_far + 0.5;
    let ray_delta = p_far - p_near;
    let total_length = length(ray_delta);

    let step_vector = normalize(ray_delta);
    let required_steps = i32(clamp(total_length / step_length, 0.0, max_steps));

    // Unity's _Time.xy is (t / 20, t). Precise has no jitter.
    var jitter = params.jitter;
    @if(VOXELS) {
        jitter = 0.0;
    }
    let rand_vector = nrand(in.position.xy + vec2(globals.time / 20.0, globals.time)) * step_vector * step_length * jitter;

    let region_scale = 1.0 / (params.texture_max - params.texture_min);
    let region_offset = -(params.texture_min + 0.5);
    var position = (p_near + rand_vector + region_offset) * region_scale;
    let s = step_vector * step_length * region_scale;
    let h_min = (params.highlight_min + region_offset + 0.5) * region_scale;
    let h_max = (params.highlight_max + region_offset + 0.5) * region_scale;
    @if(ASINH) {
        set_noise_floor(ray_delta * region_scale);
    }
    // Precise's walk, from where the ray enters the box, in texels with z in layers.
    let walked = walk((p_near + region_offset) * region_scale * texels(), direction * region_scale * texels(), total_length);
    @if(VOXELS && !CUSTOM) {
        let layers = vec3(1.0, 1.0, params.layers);
        return every_voxel(walked, h_min * layers, h_max * layers);
    }

    // The guard keeps a constant cube from dividing zero by zero.
    let scale_range = select(params.scale_max - params.scale_min, 1.0, params.scale_max == params.scale_min);

    let average_mode = params.projection_mode == AVERAGE;
    // A selection is shown, so whether a sample is in it counts.
    let track = params.highlight_saturate_factor != 1.0;
    // Raw values: under floor_value a Maximum pixel's alpha is 0, and from stop on its colour is saturated.
    let floor_value = params.threshold_min * scale_range + params.scale_min;
    let stop = params.threshold_max * scale_range + params.scale_min;
    var top = vec2(-F32_MAX);
    var acc = vec3(0.0);
    // Only the first sample can fall outside the texture (just), where it wraps round.
    if required_steps > 0 {
        let value = sample_cube(position);
        top = max_sample(top, value, position, h_min, h_max);
        acc = add_sample(acc, value, position, h_min, h_max);
    }
    // z in layers from here on, so a sample's layer is its integer part.
    let zs = vec3(1.0, 1.0, params.layers);
    let start = position * zs;
    let da = s * zs;
    let l_min = h_min * zs;
    let l_max = h_max * zs;
    @if(CUSTOM) {
        let tail = position + f32(required_steps) * s + step_vector * (total_length - f32(required_steps + 1) * step_length - length(rand_vector)) * region_scale;
        let tail_steps = max(total_length - f32(required_steps) * step_length, 0.0) / step_length;
        let jittered = t_near + length(rand_vector);
        return march(display_ray(start, da, required_steps, tail * zs, tail_steps, jittered, step_length, direction, in.ray_origin, region_scale * zs, in.instance_index, l_min, l_max, walked));
    }
    if average_mode {
        let rest = max(required_steps - 1, 0);
        if track {
            // Unrolled so the fetches overlap.
            var k = 1;
            for (; k + 4 <= required_steps; k += 4) {
                let a0 = start + f32(k) * da;
                let a1 = a0 + da;
                let a2 = a1 + da;
                let a3 = a2 + da;
                let v0 = fetch(a0);
                let v1 = fetch(a1);
                let v2 = fetch(a2);
                let v3 = fetch(a3);
                acc = add_sample(acc, v0, a0, l_min, l_max);
                acc = add_sample(acc, v1, a1, l_min, l_max);
                acc = add_sample(acc, v2, a2, l_min, l_max);
                acc = add_sample(acc, v3, a3, l_min, l_max);
            }
            for (; k < required_steps; k++) {
                let a = start + f32(k) * da;
                acc = add_sample(acc, fetch(a), a, l_min, l_max);
            }
        } else if params.has_nan != 0u {
            acc += vec3(segment_sum(start + da, da, rest, true), 0.0);
        } else {
            acc += vec3(segment_sum(start + da, da, rest, false), 0.0);
        }
    } else {
        // Front to back a grid cell at a time, stepping over a cell that can't raise the max.
        let inv_da = 1.0 / da;
        var k = 1;
        while k < required_steps {
            let span = span_at(start, da, inv_da, required_steps, k);
            let a = start + f32(k) * da;
            if span.max >= floor_value && (span.max > top.x || (span.max == top.x && track)) {
                if track && all(span.hi >= l_min) && all(span.lo <= l_max) {
                    for (var j = k; j < span.end; j++) {
                        let b = start + f32(j) * da;
                        top = max_sample(top, fetch(b), b, l_min, l_max);
                    }
                } else if span.nan {
                    top.x = max(top.x, segment_max(a, da, span.end - k, true));
                } else {
                    top.x = max(top.x, segment_max(a, da, span.end - k, false));
                }
            }
            k = span.end;
            // Once the maximum is past threshold_max the colour can't change, unless a selection is shown.
            if !track && top.x >= stop {
                return shade((top.x - params.scale_min) / scale_range, false);
            }
        }
    }
    position += f32(required_steps) * s;
    let remaining_step_length = total_length - f32(required_steps + 1) * step_length - length(rand_vector);
    position += step_vector * remaining_step_length * region_scale;
    // iDaVIE weights this sample by remaining_step_length, which is negative and can zero the average.
    let leftover_length = max(total_length - f32(required_steps) * step_length, 0.0);
    let value = sample_cube(position);
    if average_mode {
        var sum = acc.x * step_length;
        var weight = acc.y * step_length;
        var in_highlight = acc.z > 0.0;
        if !is_nan(value) {
            sum += value * leftover_length;
            weight += leftover_length;
            in_highlight = in_highlight || in_box(position, h_min, h_max);
        } else if acc.y == 0.0 {
            return vec4(0.0);
        }
        return shade((select(sum, sum / weight, weight > 0.0) - params.scale_min) / scale_range, in_highlight);
    }
    top = max_sample(top, value, position, h_min, h_max);
    if top.x == -F32_MAX {
        return vec4(0.0);
    }
    return shade((top.x - params.scale_min) / scale_range, top.y == top.x);
}

// Precise: the ray o + t d in texels, z in layers, t from 0 to its object-space length `end`, through every voxel
// it passes once each by Amanatides and Woo's traversal, mostly eight voxels at a time so that their fetches overlap.
// After each batch the walk steps over the rest of the grid cell it's in, and any cells after it, if they can't change
// the result. Visiting some of such a cell's voxels doesn't change the result either, so the check needn't fall on a
// cell boundary, and neighbouring rays don't wait on each other at each cell they cross.
struct Walk {
    o: vec3<f32>,
    d: vec3<f32>,
    end: f32,
    // Per axis, the way the ray goes, -1, 0 or 1, and how far it goes to cross a texel.
    step: vec3<f32>,
    delta: vec3<f32>,
    // The voxels the clip box takes in, lo to hi, and how many texels the grid starts before the texture.
    lo: vec3<f32>,
    hi: vec3<f32>,
    grid: vec3<f32>,
    // Per axis, the voxel just past the clip box the way the ray goes, which the walk reaches only on leaving it.
    exit: vec3<i32>,
};

fn walk(o: vec3<f32>, d: vec3<f32>, end: f32) -> Walk {
    let size = texels();
    let scale = size / (params.texture_max - params.texture_min);
    // A crop's faces are on voxel boundaries, give or take rounding.
    let lo = max(floor((params.slice_min - params.texture_min) * scale + 1e-3), vec3(0.0));
    let hi = min(ceil((params.slice_max - params.texture_min) * scale - 1e-3), size);
    let grid = round(params.grid_offset * vec3(size.xy, 1.0));
    let exit = vec3<i32>(select(lo - 1.0, hi, d > vec3(0.0)));
    return Walk(o, d, end, sign(d), select(vec3(F32_MAX), abs(1.0 / d), d != vec3(0.0)), lo, hi, grid, exit);
}

// Where the ray reaches the far side of the box from `at` to `at + size` along each axis, F32_MAX along one it doesn't
// move on.
fn boundaries(w: Walk, at: vec3<f32>, size: f32) -> vec3<f32> {
    return select(vec3(F32_MAX), (at + select(vec3(0.0), vec3(size), w.d > vec3(0.0)) - w.o) / w.d, w.d != vec3(0.0));
}

// The cell of the first voxel, found from the voxel rather than the ray's start, which can round onto the far side of a
// crop face that's also a cell's.
fn first_cell(w: Walk) -> vec3<f32> {
    let at = clamp(floor(w.o), w.lo, w.hi - 1.0);
    return clamp(cell_of(w, at), vec3(0.0), vec3<f32>(params.grid_cells - 1u));
}

// The cell a voxel is in.
fn cell_of(w: Walk, at: vec3<f32>) -> vec3<f32> {
    return floor((at + w.grid) / params.cell_size.z);
}

// A cell's box in texels, inside the clip box.
fn cell_lo(w: Walk, cell: vec3<f32>) -> vec3<f32> {
    return max(cell * params.cell_size.z - w.grid, w.lo);
}

fn cell_hi(w: Walk, cell: vec3<f32>) -> vec3<f32> {
    return min((cell + 1.0) * params.cell_size.z - w.grid, w.hi);
}

// Where the ray leaves a cell it steps over, and the cell after it.
fn past_cell(w: Walk, cell: vec3<f32>) -> vec4<f32> {
    let e = boundaries(w, cell * params.cell_size.z - w.grid, params.cell_size.z);
    let t = min(min(min(e.x, e.y), e.z), w.end);
    return vec4(cell + select(vec3(0.0), w.step, e == vec3(t)), t);
}

// The walk so far: the voxel it's at, where the ray enters it and where it crosses the next boundary along each axis,
// the cell it's in with that cell's max and NaN flag, and the length of the cells stepped over since it was last
// cleared.
struct Walking {
    at: vec3<i32>,
    t: f32,
    next: vec3<f32>,
    cell: vec3<i32>,
    max: f32,
    nan: bool,
    skipped: f32,
};

fn walking(w: Walk) -> Walking {
    return Walking(vec3(0), 0.0, vec3(F32_MAX), vec3<i32>(first_cell(w)), -F32_MAX, false, 0.0);
}

// Whether a cell with this max and NaN flag can change the result: its max is above `below`, or equal to it with
// `ties`, or it has a NaN that doesn't count as nothing.
fn worth(m: f32, nan: bool, below: f32, ties: bool, nan_skips: bool) -> bool {
    return m > below || (ties && m == below) || (nan && !nan_skips);
}

// From cell `first`, which the ray enters at `entered`, into the first cell worth visiting, adding the length of the
// cells stepped over to `skipped`. False at the end of the ray.
fn enter_from(
    it: ptr<function, Walking>, w: Walk, first: vec3<f32>, entered: f32, below: f32, ties: bool, nan_skips: bool,
) -> bool {
    var cell = first;
    var t = entered;
    for (var i = 0; i < i32(dot(vec3<f32>(params.grid_cells), vec3(1.0))) + 2 && t < w.end; i++) {
        let lo = cell_lo(w, cell);
        let hi = cell_hi(w, cell);
        if any(lo >= hi) {
            return false;
        }
        let entry = grid_entry(vec3<u32>(cell));
        let m = bitcast<f32>(entry.y);
        if worth(m, entry.x != 0u, below, ties, nan_skips) {
            let at = clamp(floor(w.o + w.d * t), lo, hi - 1.0);
            *it = Walking(vec3<i32>(at), t, boundaries(w, at, 1.0), vec3<i32>(cell), m, entry.x != 0u, (*it).skipped);
            return true;
        }
        let past = past_cell(w, cell);
        (*it).skipped += past.w - t;
        cell = past.xyz;
        t = past.w;
    }
    return false;
}

// Into the first cell worth visiting.
fn enter_first(it: ptr<function, Walking>, w: Walk, below: f32, ties: bool, nan_skips: bool) -> bool {
    return enter_from(it, w, vec3<f32>((*it).cell), 0.0, below, ties, nan_skips);
}

// Whether the walk's voxel is in the clip box, before the end of the ray.
fn in_run(it: Walking, w: Walk) -> bool {
    return all(it.at != w.exit) && it.t < w.end;
}

// The cell of the walk's voxel. In a new cell, the boundary crossings start afresh from the voxel, as they do on
// entering a cell after a skip, so that rounding doesn't build up along the ray.
fn recross(it: ptr<function, Walking>, w: Walk) -> vec3<i32> {
    let cell = vec3<i32>(cell_of(w, vec3<f32>((*it).at)));
    if any(cell != (*it).cell) {
        (*it).next = boundaries(w, vec3<f32>((*it).at), 1.0);
    }
    return cell;
}

// After a batch: on to the next voxel, stepping over the rest of the walk's cell and the cells after it if they aren't
// worth visiting. Every ray looks its cell up after every batch, so that rays don't diverge, and acts on it after the
// next, by when the fetch has arrived. So a cell not worth visiting is still walked for up to two batches after the
// walk reaches it. False at the end of the ray.
fn keep_on(it: ptr<function, Walking>, w: Walk, below: f32, ties: bool, nan_skips: bool) -> bool {
    if !in_run(*it, w) {
        return false;
    }
    let cell = recross(it, w);
    let leave = all(cell == (*it).cell) && !worth((*it).max, (*it).nan, below, ties, nan_skips);
    let entry = grid_entry(vec3<u32>(cell));
    (*it).cell = cell;
    (*it).max = bitcast<f32>(entry.y);
    (*it).nan = entry.x != 0u;
    if !leave {
        return true;
    }
    let past = past_cell(w, vec3<f32>(cell));
    (*it).skipped += past.w - (*it).t;
    return enter_from(it, w, past.xyz, past.w, below, ties, nan_skips);
}

// A voxel of the walk: its texel, where the ray enters and leaves it, and whether it counts, which it doesn't once the
// ray has left the clip box or ended. One that doesn't is still fetched.
struct Voxel {
    at: vec3<i32>,
    t: f32,
    out: f32,
    on: bool,
};

// The walk's voxel, with the walk moved on to the next one if it counts.
fn take(it: ptr<function, Walking>, w: Walk) -> Voxel {
    let on = in_run(*it, w);
    let out = min(min((*it).next.x, (*it).next.y), min((*it).next.z, w.end));
    var at = (*it).at;
    // One that doesn't count can lie past the texture. A browser checks every load's bounds, but native shaders aren't
    // checked, so there it's fetched from inside the clip box.
    @if(GUARD_LOADS) {
        at = select(vec3<i32>(w.lo), at, on);
    }
    let v = Voxel(at, (*it).t, out, on);
    // A voxel that doesn't count leaves the walk where it is.
    let crossed = (*it).next == vec3(select(-F32_MAX, out, on));
    (*it).at += select(vec3(0), vec3<i32>(w.step), crossed);
    (*it).next += select(vec3(0.0), w.delta, crossed);
    (*it).t = select((*it).t, out, on);
    return v;
}

fn length_in(v: Voxel) -> f32 {
    return max(v.out - v.t, 0.0);
}

// The voxel's middle in texture coordinates, z in layers.
fn middle(v: Voxel) -> vec3<f32> {
    return (vec3<f32>(v.at) + 0.5) * texel_size();
}

// The most voxels a ray can pass through.
fn most_voxels() -> i32 {
    return i32(dot(texels(), vec3(1.0))) + 3;
}

fn is_nan4(value: vec4<f32>) -> vec4<bool> {
    return (bitcast<vec4<u32>>(value) & vec4(0x7fffffffu)) > vec4(0x7f800000u);
}

// Four voxels of the walk in turn, and their values once fetched.
struct Four {
    a: Voxel,
    b: Voxel,
    c: Voxel,
    d: Voxel,
    x: vec4<f32>,
};

fn four(it: ptr<function, Walking>, w: Walk) -> Four {
    let a = take(it, w);
    let b = take(it, w);
    let c = take(it, w);
    let d = take(it, w);
    return Four(a, b, c, d, vec4(0.0));
}

fn fetched(f: Four) -> Four {
    return Four(f.a, f.b, f.c, f.d, vec4(load(f.a.at), load(f.b.at), load(f.c.at), load(f.d.at)));
}

// Eight voxels of the walk, as two Fours, fetched once all eight are found so that the fetches overlap.
struct Eight {
    f: Four,
    g: Four,
};

fn eight(it: ptr<function, Walking>, w: Walk) -> Eight {
    let f = four(it, w);
    let g = four(it, w);
    return Eight(fetched(f), fetched(g));
}

fn on4(f: Four) -> vec4<bool> {
    return vec4(f.a.on, f.b.on, f.c.on, f.d.on);
}

fn lengths4(f: Four) -> vec4<f32> {
    return vec4(length_in(f.a), length_in(f.b), length_in(f.c), length_in(f.d));
}

fn in_box4(f: Four, l_min: vec3<f32>, l_max: vec3<f32>) -> vec4<bool> {
    return vec4(in_box(middle(f.a), l_min, l_max), in_box(middle(f.b), l_min, l_max), in_box(middle(f.c), l_min, l_max),
        in_box(middle(f.d), l_min, l_max));
}

// The values of the voxels that count, the others and NaN as -F32_MAX.
fn counted4(f: Four) -> vec4<f32> {
    return select(vec4(-F32_MAX), f.x, on4(f) & !is_nan4(f.x));
}

// The sum and length of the voxels that count, each value weighted by the ray's length in it, and how many are in the
// selection when it's shown.
fn weighted4(f: Four, track: bool, l_min: vec3<f32>, l_max: vec3<f32>) -> vec3<f32> {
    let counts = on4(f) & !is_nan4(f.x);
    let l = select(vec4(0.0), lengths4(f), counts);
    var selected = 0.0;
    if track {
        selected = dot(select(vec4(0.0), vec4(1.0), counts & in_box4(f, l_min, l_max)), vec4(1.0));
    }
    return vec3(dot(select(vec4(0.0), f.x, counts), l), dot(l, vec4(1.0)), selected);
}

fn max_voxel(top: vec2<f32>, value: f32, v: Voxel, l_min: vec3<f32>, l_max: vec3<f32>) -> vec2<f32> {
    return select(top, max_sample(top, value, middle(v), l_min, l_max), v.on);
}

fn max4(top: vec2<f32>, f: Four, l_min: vec3<f32>, l_max: vec3<f32>) -> vec2<f32> {
    let ab = max_voxel(max_voxel(top, f.x.x, f.a, l_min, l_max), f.x.y, f.b, l_min, l_max);
    return max_voxel(max_voxel(ab, f.x.z, f.c, l_min, l_max), f.x.w, f.d, l_min, l_max);
}

// Maximum over the voxels, stepping over cells that can't raise it, as the fixed step does, which includes the rest of
// a cell once its max is reached.
@if(VOXELS && !CUSTOM && !VOXELS_AVERAGE)
fn every_voxel(w: Walk, l_min: vec3<f32>, l_max: vec3<f32>) -> vec4<f32> {
    let scale_range = select(params.scale_max - params.scale_min, 1.0, params.scale_max == params.scale_min);
    let track = params.highlight_saturate_factor != 1.0;
    let floor_value = params.threshold_min * scale_range + params.scale_min;
    let stop = params.threshold_max * scale_range + params.scale_min;
    // The max, and the max in the selection, which is compared with it, so a tie counts when a selection is shown.
    var top = vec2(-F32_MAX);
    var it = walking(w);
    // Under the floor a cell whose max is the floor counts, as it does for the fixed step.
    var more = enter_first(&it, w, floor_value, true, true);
    if track {
        for (var i = 0; more && i < most_voxels(); i += 4) {
            top = max4(top, fetched(four(&it, w)), l_min, l_max);
            more = keep_on(&it, w, max(floor_value, top.x), true, true);
        }
    } else {
        for (var i = 0; more && i < most_voxels(); i += 8) {
            let e = eight(&it, w);
            let m = max(counted4(e.f), counted4(e.g));
            top.x = max(top.x, max(max(m.x, m.y), max(m.z, m.w)));
            if top.x >= stop {
                return shade((top.x - params.scale_min) / scale_range, false);
            }
            more = keep_on(&it, w, max(floor_value, top.x), top.x < floor_value, true);
        }
    }
    if top.x == -F32_MAX {
        return vec4(0.0);
    }
    return shade((top.x - params.scale_min) / scale_range, top.y == top.x);
}

// Average over the voxels, weighting each by the ray's length in it, stepping over cells with nothing but NaN.
@if(VOXELS && !CUSTOM && VOXELS_AVERAGE)
fn every_voxel(w: Walk, l_min: vec3<f32>, l_max: vec3<f32>) -> vec4<f32> {
    let scale_range = select(params.scale_max - params.scale_min, 1.0, params.scale_max == params.scale_min);
    let track = params.highlight_saturate_factor != 1.0;
    // The sum, the length, and whether any of it is in the selection.
    var acc = vec3(0.0);
    var it = walking(w);
    var more = enter_first(&it, w, -F32_MAX, true, true);
    for (var i = 0; more && i < most_voxels(); i += 8) {
        let e = eight(&it, w);
        acc += weighted4(e.f, track, l_min, l_max) + weighted4(e.g, track, l_min, l_max);
        // Without a NaN there's no cell to step over, but the crossings still start afresh in each new cell.
        if params.has_nan != 0u {
            more = keep_on(&it, w, -F32_MAX, true, true);
        } else {
            more = in_run(it, w);
            it.cell = recross(&it, w);
        }
    }
    if acc.y == 0.0 {
        return vec4(0.0);
    }
    return shade((acc.x / acc.y - params.scale_min) / scale_range, acc.z > 0.0);
}

// Thresholds, scaling and the colour map, for a ray's value normalised to the scale range.
fn shade(normalised: f32, in_highlight: bool) -> vec4<f32> {
    // Unity's v axis points up the atlas, wgpu's points down, so no "1.0 -" here.
    let color_map_offset = (0.5 + f32(params.color_map_index)) / f32(params.num_color_maps);
    let x = scaled(normalised);
    let color_fraction = select(params.highlight_saturate_factor, 1.0, in_highlight);
    let color_map_color = vec4(textureSampleLevel(color_map, color_map_sampler, vec2(x, color_map_offset), 0.0).rgb, x);
    return mix(vec4(x), color_map_color, color_fraction);
}

// A value normalised to the scale range, to its place on the colour map.
fn scaled(normalised: f32) -> f32 {
    let ray_value = clamp(normalised, params.threshold_min, params.threshold_max);
    // The guard keeps equal thresholds from dividing zero by zero.
    var x = (ray_value - params.threshold_min) / max(params.threshold_max - params.threshold_min, 1e-6);

    if params.scale_type == SQUARE {
        x = x * x;
    } else if params.scale_type == SQRT {
        x = sqrt(x);
    } else if params.scale_type == LOG {
        x = clamp(log(params.scale_alpha * x + 1.0) / log(params.scale_alpha), 0.0, 1.0);
    } else if params.scale_type == POWER {
        x = (pow(params.scale_alpha, x) - 1.0) / params.scale_alpha;
    } else if params.scale_type == GAMMA {
        x = pow(x, params.scale_gamma);
    }
    // Its own pipeline (scale_type 6), so the others don't pay for the noise floor.
    @if(ASINH) {
        // Lupton's asinh from the noise floor, linear up to asinh_sigma and logarithmic above.
        let range = select(params.scale_max - params.scale_min, 1.0, params.scale_max == params.scale_min);
        let low = max(params.threshold_min, (noise_floor - params.scale_min) / range);
        let beta = max(modes.asinh_sigma * modes.noise_sigma / range, 1e-6);
        x = asinh(max(ray_value - low, 0.0) / beta) / asinh(max(params.threshold_max - low, 1e-6) / beta);
    }

    x = clamp(x - params.scale_bias, 0.0, 1.0);
    return clamp((x - 0.5) * params.scale_contrast + 0.5, 0.0, 1.0);
}

// Everything from here is the rendering modes beyond Maximum and Average (modes.rs). Each compiles alone, under
// CUSTOM and its MODE_ def, with MASK, SNR_OPACITY and LIT for its variants.

struct ModeParams {
    noise_median: f32,
    noise_sigma: f32,
    sigmas: f32,
    // In data units; f32::MIN for none.
    level: f32,
    depth_cue: f32,
    attenuation: f32,
    opacity: f32,
    mida: f32,
    asinh_sigma: f32,
    ray_max: u32,
    mask_zero: f32,
    mask_outside: f32,
    // Layers where the channel colours start and end.
    channel_from: f32,
    channel_to: f32,
};

@group(constants::MATERIAL_BIND_GROUP) @binding(6) var<uniform> modes: ModeParams;
// modes.rs mask_image: above 0 inside the mask, laid out as data_cube.
@group(constants::MATERIAL_BIND_GROUP) @binding(7) var mask: texture_2d_array<f32>;

// Where asinh starts on this pixel's ray: the noise median, or for the maximum modes the brightest the noise alone
// likely reaches over the ray's n texels, sqrt(2 ln n) σ above it.
@if(ASINH)
var<private> noise_floor: f32;

@if(ASINH)
fn set_noise_floor(delta: vec3<f32>) {
    let n = length(delta * texels());
    noise_floor = modes.noise_median + select(0.0, sqrt(2.0 * log(max(n, 2.0))) * modes.noise_sigma, modes.ray_max != 0u);
}

// Outside the mask a value is pulled towards mask_zero.
fn masked(value: f32, inside: bool) -> f32 {
    return select(modes.mask_zero + modes.mask_outside * (value - modes.mask_zero), value, inside);
}

@if(MASK)
fn fetch(a: vec3<f32>) -> f32 {
    let value = textureSampleLevel(data_cube, data_sampler, a.xy, i32(a.z), 0.0).r;
    return masked(value, textureSampleLevel(mask, data_sampler, a.xy, i32(a.z), 0.0).r > 0.0);
}

// A ray's samples, z in layers: `steps` from `start`, `da` apart, then the tail, weighing `tail_steps` of a step.
// Distances and directions are in display space (see display_ray).
struct Ray {
    start: vec3<f32>,
    da: vec3<f32>,
    inv_da: vec3<f32>,
    steps: i32,
    tail: vec3<f32>,
    tail_steps: f32,
    // From the camera to the first sample.
    t0: f32,
    step_length: f32,
    direction: vec3<f32>,
    origin: vec3<f32>,
    // Texture coordinates, z in layers, per display-space unit.
    scale: vec3<f32>,
    light: vec3<f32>,
    // The selection's box, z in layers.
    selection_min: vec3<f32>,
    selection_max: vec3<f32>,
    // The clip box.
    clip_min: vec3<f32>,
    clip_max: vec3<f32>,
    // Precise's walk, in display-space lengths from where the ray enters the box, which is t0 as Precise
    // has no jitter.
    walk: Walk,
};

// A Ray from the fragment's object-space distances and directions, converted to display space: object space scaled
// to params.box_size, so that depth, opacity per unit length and lighting follow the z scale.
fn display_ray(start: vec3<f32>, da: vec3<f32>, steps: i32, tail: vec3<f32>, tail_steps: f32, t0: f32, step_length: f32, direction: vec3<f32>, origin: vec3<f32>, scale: vec3<f32>, instance: u32, selection_min: vec3<f32>, selection_max: vec3<f32>, w: Walk) -> Ray {
    let size = params.box_size;
    let stretch = length(direction * size);
    return Ray(start, da, 1.0 / da, steps, tail, tail_steps, t0 * stretch, step_length * stretch, normalize(direction * size), origin * size, scale / size, key_light(instance), selection_min, selection_max, params.slice_min * size, params.slice_max * size, walk(w.o, w.d / stretch, w.end * stretch));
}

fn in_selection(r: Ray, a: vec3<f32>) -> bool {
    return in_box(a, r.selection_min, r.selection_max);
}

fn crosses_selection(r: Ray) -> bool {
    let t = intersect_box(r.start, r.da, r.selection_min, r.selection_max);
    // Precise takes the whole ray, so the part step at its end counts too.
    var end = f32(r.steps);
    @if(VOXELS) {
        end += r.tail_steps;
    }
    return t.x <= t.y && t.y >= 0.0 && t.x <= end;
}

// As shade desaturates what's outside a selection shown.
fn saturated(color: vec4<f32>, selected: bool) -> vec4<f32> {
    let fraction = select(params.highlight_saturate_factor, 1.0, selected);
    return vec4(mix(vec3(dot(color.rgb, vec3(0.299, 0.587, 0.114))), color.rgb, fraction), color.a);
}

fn sample_at(r: Ray, k: i32) -> vec3<f32> {
    return r.start + f32(k) * r.da;
}

// Sample k, or for k = steps the tail.
fn ray_sample(r: Ray, k: i32) -> vec3<f32> {
    return select(sample_at(r, k), r.tail, k >= r.steps);
}

fn cell_span(r: Ray, k: i32) -> Span {
    return span_at(r.start, r.da, r.inv_da, r.steps, k);
}

fn texture_box() -> vec3<f32> {
    return vec3(1.0, 1.0, params.layers);
}

fn texel_size() -> vec3<f32> {
    return vec3(1.0 / vec2<f32>(textureDimensions(data_cube)), 1.0);
}

// The texture's size in texels, z in layers.
fn texels() -> vec3<f32> {
    return vec3(vec2<f32>(textureDimensions(data_cube)), params.layers);
}

fn normalised(value: f32) -> f32 {
    let range = select(params.scale_max - params.scale_min, 1.0, params.scale_max == params.scale_min);
    return (value - params.scale_min) / range;
}

// A threshold in data units.
fn raw(threshold: f32) -> f32 {
    return params.scale_min + threshold * (params.scale_max - params.scale_min);
}

fn color_at(x: f32) -> vec3<f32> {
    let row = (0.5 + f32(params.color_map_index)) / f32(params.num_color_maps);
    return textureSampleLevel(color_map, color_map_sampler, vec2(x, row), 0.0).rgb;
}

// The colour of a place along the texture's channels, z in layers.
fn channel_color(z: f32) -> vec3<f32> {
    return color_at(clamp((z - modes.channel_from) / (modes.channel_to - modes.channel_from), 0.0, 1.0));
}

fn filled(value: f32) -> f32 {
    return select(value, modes.noise_median, is_nan(value));
}

fn better(best: vec2<f32>, value: f32, k: i32) -> vec2<f32> {
    return select(best, vec2(value, f32(k)), value > best.x);
}

// The largest sample and its step (`steps` for the tail), stepping over cells that can't beat it or reach `floor`,
// and stopping once it reaches `stop`. NaN never wins a comparison, so needs no test.
fn argmax(r: Ray, floor: f32, stop: f32) -> vec2<f32> {
    var best = vec2(-F32_MAX, -1.0);
    var k = 0;
    while k < r.steps {
        let span = cell_span(r, k);
        if span.max >= floor && span.max > best.x {
            var a = sample_at(r, k);
            var j = k;
            // Eight at a time, as segment_max, looking for which one only when they beat it; max of a NaN is
            // undefined, so not in a cell with one.
            for (; j + 8 <= span.end; j += 8) {
                let lo = vec4(fetch(a), fetch(a + r.da), fetch(a + 2.0 * r.da), fetch(a + 3.0 * r.da));
                let hi = vec4(fetch(a + 4.0 * r.da), fetch(a + 5.0 * r.da), fetch(a + 6.0 * r.da), fetch(a + 7.0 * r.da));
                let m = max(lo, hi);
                if span.nan || max(max(m.x, m.y), max(m.z, m.w)) > best.x {
                    for (var i = 0; i < 4; i++) {
                        best = better(best, lo[i], j + i);
                    }
                    for (var i = 0; i < 4; i++) {
                        best = better(best, hi[i], j + 4 + i);
                    }
                }
                a += 8.0 * r.da;
            }
            for (; j < span.end; j++) {
                best = better(best, fetch(a), j);
                a += r.da;
            }
        }
        k = span.end;
        if best.x >= stop {
            return best;
        }
    }
    return better(best, fetch(r.tail), r.steps);
}

// The ray's largest value, how far it is from the camera, and where it is.
struct Peak {
    value: f32,
    t: f32,
    at: vec3<f32>,
};

@if(!VOXELS)
fn peak(r: Ray, floor: f32, stop: f32) -> Peak {
    let best = argmax(r, floor, stop);
    return Peak(best.x, r.t0 + best.y * r.step_length, select(sample_at(r, i32(best.y)), r.tail, best.y >= f32(r.steps)));
}

@if(VOXELS)
fn peak(r: Ray, floor: f32, stop: f32) -> Peak {
    var best = Peak(-F32_MAX, 0.0, vec3(0.0));
    var it = walking(r.walk);
    var more = enter_first(&it, r.walk, floor, true, true);
    for (var i = 0; more && best.value < stop && i < most_voxels(); i += 8) {
        let e = eight(&it, r.walk);
        best = better4(better4(best, e.f, r, stop), e.g, r, stop);
        // Under the floor a cell whose max is the floor still counts; past it a tie can't beat the best.
        more = keep_on(&it, r.walk, max(floor, best.value), best.value < floor, true);
    }
    return best;
}

// The best so far, or the voxel if it's higher and the best hasn't reached `stop`.
fn better_voxel(best: Peak, value: f32, v: Voxel, r: Ray, stop: f32) -> Peak {
    if v.on && best.value < stop && value > best.value {
        return Peak(value, r.t0 + 0.5 * (v.t + v.out), middle(v));
    }
    return best;
}

@if(VOXELS)
fn better4(best: Peak, f: Four, r: Ray, stop: f32) -> Peak {
    let ab = better_voxel(better_voxel(best, f.x.x, f.a, r, stop), f.x.y, f.b, r, stop);
    return better_voxel(better_voxel(ab, f.x.z, f.c, r, stop), f.x.w, f.d, r, stop);
}

@if(MODE_DEPTH_CUED)
fn march(r: Ray) -> vec4<f32> {
    let best = peak(r, raw(params.threshold_min), raw(params.threshold_max));
    if best.value == -F32_MAX {
        return vec4(0.0);
    }
    let color = shade(normalised(best.value), in_selection(r, best.at));
    // Darker by up to depth_cue from the box's nearest point (about) to its far side.
    let diag = length(r.clip_max - r.clip_min);
    let near = length(r.origin - 0.5 * (r.clip_min + r.clip_max)) - 0.5 * diag;
    return vec4(color.rgb * (1.0 - modes.depth_cue * clamp((best.t - near) / diag, 0.0, 1.0)), color.a);
}

@if(MODE_PEAK_CHANNEL)
fn march(r: Ray) -> vec4<f32> {
    let best = peak(r, modes.level, F32_MAX);
    if best.value < modes.level {
        return vec4(0.0);
    }
    let color = vec4(channel_color(floor(best.at.z) + 0.5), scaled(normalised(best.value)));
    return saturated(color, in_selection(r, best.at));
}

// napari's attenuated MIP: each sample, 0 at `lo` and 1 at threshold Max, dimmed by the optical depth in front of
// it. Returns the peak and the optical depth.
fn attenuate(s: vec2<f32>, value: f32, lo: f32, inv: f32, depth: f32) -> vec2<f32> {
    let u = select(clamp((value - lo) * inv, 0.0, 1.0), 0.0, is_nan(value));
    return vec2(max(s.x, u * exp(-s.y)), s.y + u * depth);
}

// The peak and optical depth along the ray, from `lo` up.
@if(MODE_ATTENUATED && !VOXELS)
fn attenuated(r: Ray, lo: f32, inv: f32) -> vec2<f32> {
    let depth = modes.attenuation * r.step_length;
    var s = vec2(0.0);
    var k = 0;
    // Once the attenuation alone is under the peak, nothing behind can beat it.
    while k < r.steps && exp(-s.y) > s.x {
        let span = cell_span(r, k);
        // A cell under lo adds nothing to either.
        if span.max > lo {
            var a = sample_at(r, k);
            var j = k;
            for (; j + 4 <= span.end && exp(-s.y) > s.x; j += 4) {
                let v0 = fetch(a);
                let v1 = fetch(a + r.da);
                let v2 = fetch(a + 2.0 * r.da);
                let v3 = fetch(a + 3.0 * r.da);
                s = attenuate(attenuate(attenuate(attenuate(s, v0, lo, inv, depth), v1, lo, inv, depth), v2, lo, inv, depth), v3, lo, inv, depth);
                a += 4.0 * r.da;
            }
            for (; j < span.end && exp(-s.y) > s.x; j++) {
                s = attenuate(s, fetch(a), lo, inv, depth);
                a += r.da;
            }
        }
        k = span.end;
    }
    return attenuate(s, fetch(r.tail), lo, inv, depth * r.tail_steps);
}

@if(MODE_ATTENUATED && VOXELS)
fn attenuated(r: Ray, lo: f32, inv: f32) -> vec2<f32> {
    var s = vec2(0.0);
    var it = walking(r.walk);
    var more = enter_first(&it, r.walk, lo, false, true);
    for (var i = 0; more && exp(-s.y) > s.x && i < most_voxels(); i += 8) {
        let e = eight(&it, r.walk);
        s = attenuate4(attenuate4(s, e.f, lo, inv), e.g, lo, inv);
        more = keep_on(&it, r.walk, lo, false, true);
    }
    return s;
}

// As attenuate, for a voxel that counts, until nothing behind can beat the peak.
@if(MODE_ATTENUATED && VOXELS)
fn attenuate_voxel(s: vec2<f32>, value: f32, v: Voxel, lo: f32, inv: f32) -> vec2<f32> {
    if v.on && exp(-s.y) > s.x {
        return attenuate(s, value, lo, inv, modes.attenuation * length_in(v));
    }
    return s;
}

@if(MODE_ATTENUATED && VOXELS)
fn attenuate4(s: vec2<f32>, f: Four, lo: f32, inv: f32) -> vec2<f32> {
    let ab = attenuate_voxel(attenuate_voxel(s, f.x.x, f.a, lo, inv), f.x.y, f.b, lo, inv);
    return attenuate_voxel(attenuate_voxel(ab, f.x.z, f.c, lo, inv), f.x.w, f.d, lo, inv);
}

@if(MODE_ATTENUATED)
fn march(r: Ray) -> vec4<f32> {
    // The noise under the level would dim everything behind it.
    let lo = max(raw(params.threshold_min), modes.level);
    let hi = raw(params.threshold_max);
    let s = attenuated(r, lo, 1.0 / max(hi - lo, 1e-30));
    if s.x <= 0.0 {
        return vec4(0.0);
    }
    return shade(normalised(lo + s.x * (hi - lo)), crosses_selection(r));
}

// A sample's share of the clipped moments: its value, or the noise median at or under the level; a count of one;
// and its value above the median weighing its layer, for the mean channel. NaN adds nothing.
fn clipped(value: f32, z: f32) -> vec4<f32> {
    let c = select(modes.noise_median, value, value > modes.level);
    let w = max(c - modes.noise_median, 0.0);
    return select(vec4(c, 1.0, w * (floor(z) + 0.5), w), vec4(0.0), is_nan(value));
}

// The clipped moments over the ray. A cell under the level counts without being sampled, unless it has a NaN.
@if(!VOXELS)
fn moments(r: Ray) -> vec4<f32> {
    var acc = vec4(0.0);
    var k = 0;
    while k < r.steps {
        let span = cell_span(r, k);
        if span.max <= modes.level && !span.nan {
            acc += vec4(modes.noise_median, 1.0, 0.0, 0.0) * f32(span.end - k);
        } else {
            var a = sample_at(r, k);
            var j = k;
            for (; j + 4 <= span.end; j += 4) {
                let a1 = a + r.da;
                let a2 = a1 + r.da;
                let a3 = a2 + r.da;
                let v0 = fetch(a);
                let v1 = fetch(a1);
                let v2 = fetch(a2);
                let v3 = fetch(a3);
                acc += (clipped(v0, a.z) + clipped(v1, a1.z)) + (clipped(v2, a2.z) + clipped(v3, a3.z));
                a = a3 + r.da;
            }
            for (; j < span.end; j++) {
                acc += clipped(fetch(a), a.z);
                a += r.da;
            }
        }
        k = span.end;
    }
    return acc + clipped(fetch(r.tail), r.tail.z) * r.tail_steps;
}

@if(VOXELS)
fn moments(r: Ray) -> vec4<f32> {
    var acc = vec4(0.0);
    var it = walking(r.walk);
    var more = enter_first(&it, r.walk, modes.level, false, false);
    for (var i = 0; i < most_voxels(); i += 8) {
        // Cells stepped over count as the noise median.
        acc += vec4(modes.noise_median, 1.0, 0.0, 0.0) * it.skipped;
        it.skipped = 0.0;
        if !more {
            break;
        }
        let e = eight(&it, r.walk);
        acc = clipped4(clipped4(acc, e.f), e.g);
        more = keep_on(&it, r.walk, modes.level, false, false);
    }
    return acc;
}

fn clipped_voxel(acc: vec4<f32>, value: f32, v: Voxel) -> vec4<f32> {
    return select(acc, acc + clipped(value, f32(v.at.z)) * length_in(v), v.on);
}

fn clipped4(acc: vec4<f32>, f: Four) -> vec4<f32> {
    let ab = clipped_voxel(clipped_voxel(acc, f.x.x, f.a), f.x.y, f.b);
    return clipped_voxel(clipped_voxel(ab, f.x.z, f.c), f.x.w, f.d);
}

@if(MODE_CLIPPED_MOMENT)
fn march(r: Ray) -> vec4<f32> {
    let acc = moments(r);
    if acc.y <= 0.0 {
        return vec4(0.0);
    }
    return shade(normalised(acc.x / acc.y), crosses_selection(r));
}

@if(MODE_MEAN_CHANNEL)
fn march(r: Ray) -> vec4<f32> {
    let acc = moments(r);
    if acc.w <= 0.0 {
        return vec4(0.0);
    }
    let color = vec4(channel_color(acc.z / acc.w), scaled(normalised(acc.x / acc.y)));
    return saturated(color, crosses_selection(r));
}

fn snr(value: f32) -> f32 {
    return (value - modes.noise_median) / modes.noise_sigma;
}

// Log S/N from the level, 0, to threshold Max, 1.
fn snr_ramp(value: f32) -> f32 {
    let top = max(snr(raw(params.threshold_max)) / modes.sigmas, 1.0001);
    return clamp(log(max(snr(value), 1e-6) / modes.sigmas) / log(top), 0.0, 1.0);
}

struct Glow {
    color: vec3<f32>,
    alpha: f32,
    peak: f32,
};

// One sample of emission-absorption, front to back, with opacity per unit length so the step count doesn't change
// it. MIDA (Bruckner and Gröller 2009): a new maximum along the ray fades what's in front of it, less so as mida
// goes from 0 to -1.
fn glow(g: Glow, value: f32, a: vec3<f32>, steps: f32, r: Ray) -> Glow {
    if !(value >= modes.level) {
        return g;
    }
    let x = scaled(normalised(value));
    var rgb = color_at(x);
    var o = x;
    @if(SNR_OPACITY) {
        o = snr_ramp(value);
    }
    let alpha = 1.0 - exp(-modes.opacity * o * r.step_length * steps);
    @if(LIT) {
        if alpha > 1e-3 {
            rgb = lit(rgb, a, r);
        }
    }
    let beta = 1.0 - max(x - g.peak, 0.0) * (1.0 + min(modes.mida, 0.0));
    let color = beta * g.color + (1.0 - beta * g.alpha) * alpha * rgb;
    return Glow(color, beta * g.alpha + (1.0 - beta * g.alpha) * alpha, max(g.peak, x));
}

// Until it's opaque and, for MIDA, nothing further can be a new maximum.
fn glowing(g: Glow) -> bool {
    return !(g.alpha > 0.99 && (modes.mida <= -1.0 || g.peak >= 1.0));
}

@if(MODE_EMISSION && !VOXELS)
fn emitted(r: Ray) -> Glow {
    var g = Glow(vec3(0.0), 0.0, 0.0);
    var k = 0;
    while k < r.steps && glowing(g) {
        let span = cell_span(r, k);
        // Nothing in a cell under the level shows.
        if span.max >= modes.level {
            var a = sample_at(r, k);
            var j = k;
            for (; j + 4 <= span.end && glowing(g); j += 4) {
                let a1 = a + r.da;
                let a2 = a1 + r.da;
                let a3 = a2 + r.da;
                let v0 = fetch(a);
                let v1 = fetch(a1);
                let v2 = fetch(a2);
                let v3 = fetch(a3);
                g = glow(glow(glow(glow(g, v0, a, 1.0, r), v1, a1, 1.0, r), v2, a2, 1.0, r), v3, a3, 1.0, r);
                a = a3 + r.da;
            }
            for (; j < span.end && glowing(g); j++) {
                g = glow(g, fetch(a), a, 1.0, r);
                a += r.da;
            }
        }
        k = span.end;
    }
    if glowing(g) {
        g = glow(g, fetch(r.tail), r.tail, r.tail_steps, r);
    }
    return g;
}

// Each voxel's opacity from the ray's length in it.
@if(MODE_EMISSION && VOXELS)
fn emitted(r: Ray) -> Glow {
    var g = Glow(vec3(0.0), 0.0, 0.0);
    var it = walking(r.walk);
    var more = enter_first(&it, r.walk, modes.level, true, true);
    for (var i = 0; more && glowing(g) && i < most_voxels(); i += 8) {
        let e = eight(&it, r.walk);
        g = glow4(glow4(g, e.f, r), e.g, r);
        more = keep_on(&it, r.walk, modes.level, true, true);
    }
    return g;
}

@if(MODE_EMISSION && VOXELS)
fn glow_voxel(g: Glow, value: f32, v: Voxel, r: Ray) -> Glow {
    if v.on && glowing(g) {
        return glow(g, value, middle(v), length_in(v) / r.step_length, r);
    }
    return g;
}

@if(MODE_EMISSION && VOXELS)
fn glow4(g: Glow, f: Four, r: Ray) -> Glow {
    let ab = glow_voxel(glow_voxel(g, f.x.x, f.a, r), f.x.y, f.b, r);
    return glow_voxel(glow_voxel(ab, f.x.z, f.c, r), f.x.w, f.d, r);
}

@if(MODE_EMISSION)
fn march(r: Ray) -> vec4<f32> {
    let g = emitted(r);
    var color = select(vec4(0.0), vec4(g.color / g.alpha, g.alpha), g.alpha > 0.0);
    if modes.mida > 0.0 {
        // Towards the maximum, drawn as Maximum draws it.
        let blended = mix(vec4(g.color, g.alpha), vec4(color_at(g.peak) * g.peak, g.peak), modes.mida);
        color = vec4(blended.rgb / max(blended.a, 1e-30), blended.a);
    }
    return saturated(color, crosses_selection(r));
}

fn to_display(local_from_world: mat4x4<f32>, world: vec3<f32>) -> vec3<f32> {
    return normalize((local_from_world * vec4(world, 0.0)).xyz * UNITY_FROM_LOCAL * params.box_size);
}

// A key light about 50 degrees above left of the view, so shading shows relief and shadows fall down to the right.
fn key_light(instance: u32) -> vec3<f32> {
    let local_from_world = get_local_from_world(instance);
    let right = to_display(local_from_world, view.world_from_view[0].xyz);
    let up = to_display(local_from_world, view.world_from_view[1].xyz);
    let back = to_display(local_from_world, view.world_from_view[2].xyz);
    return normalize(back + up - 0.7 * right);
}

fn blinn_phong(albedo: vec3<f32>, n: vec3<f32>, r: Ray, light: f32) -> vec3<f32> {
    let h = normalize(r.light - r.direction);
    let diffuse = max(dot(n, r.light), 0.0);
    let specular = pow(max(dot(n, h), 0.0), 40.0);
    return albedo * (0.2 + 0.8 * diffuse * light) + 0.25 * specular * light;
}

// A unit normal facing the viewer, or the way back along the ray where there's no gradient.
fn facing(n: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    let len = length(n);
    let u = select(-direction, n / len, len > 0.0);
    return select(u, -u, dot(u, direction) > 0.0);
}

// A texel, masked.
fn load(i: vec3<i32>) -> f32 {
    var value = textureLoad(data_cube, i.xy, i.z, 0).r;
    @if(MASK) {
        value = masked(value, textureLoad(mask, i.xy, i.z, 0).r > 0.0);
    }
    return value;
}

// A texel, masked, with NaN as the noise median so that it can't poison an interpolation.
fn texel(i: vec3<i32>) -> f32 {
    let last = vec3(vec2<i32>(textureDimensions(data_cube)), i32(params.layers)) - 1;
    return filled(load(clamp(i, vec3(0), last)));
}

// By hand, since R32Float is unfilterable: 8 loads.
fn trilinear(a: vec3<f32>) -> f32 {
    let u = a / texel_size() - 0.5;
    let b = floor(u);
    let f = u - b;
    let i = vec3<i32>(b);
    let x00 = mix(texel(i), texel(i + vec3(1, 0, 0)), f.x);
    let x10 = mix(texel(i + vec3(0, 1, 0)), texel(i + vec3(1, 1, 0)), f.x);
    let x01 = mix(texel(i + vec3(0, 0, 1)), texel(i + vec3(1, 0, 1)), f.x);
    let x11 = mix(texel(i + vec3(0, 1, 1)), texel(i + vec3(1, 1, 1)), f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

// Central differences of the trilinear field two texels either side, per texture coordinate (z in layers): smooth
// enough in noise for a surface.
fn gradient(a: vec3<f32>) -> vec3<f32> {
    let h = 2.0 * texel_size();
    let g = vec3(
        trilinear(a + vec3(h.x, 0.0, 0.0)) - trilinear(a - vec3(h.x, 0.0, 0.0)),
        trilinear(a + vec3(0.0, h.y, 0.0)) - trilinear(a - vec3(0.0, h.y, 0.0)),
        trilinear(a + vec3(0.0, 0.0, h.z)) - trilinear(a - vec3(0.0, 0.0, h.z)),
    );
    return g / (2.0 * h);
}

// The cheap one, for every lit sample: the nearest texels either side.
fn nearest_gradient(a: vec3<f32>) -> vec3<f32> {
    let h = texel_size();
    let g = vec3(
        filled(fetch(a + vec3(h.x, 0.0, 0.0))) - filled(fetch(a - vec3(h.x, 0.0, 0.0))),
        filled(fetch(a + vec3(0.0, h.y, 0.0))) - filled(fetch(a - vec3(0.0, h.y, 0.0))),
        filled(fetch(a + vec3(0.0, 0.0, h.z))) - filled(fetch(a - vec3(0.0, 0.0, h.z))),
    );
    return g / (2.0 * h);
}

// Lit where the gradient stands out from the noise, fully from 2σ a texel, so the noise isn't.
fn lit(rgb: vec3<f32>, a: vec3<f32>, r: Ray) -> vec3<f32> {
    let g = nearest_gradient(a);
    let edge = min(length(g * texel_size()) / (2.0 * modes.noise_sigma), 1.0);
    return mix(rgb, blinn_phong(rgb, facing(-g * r.scale, r.direction), r, 1.0), edge);
}

fn inside(iso: f32, value: f32) -> f32 {
    return smoothstep(iso - modes.noise_sigma, iso + modes.noise_sigma, filled(value));
}

fn tangent(n: vec3<f32>) -> vec3<f32> {
    return normalize(cross(n, select(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), abs(n.x) > 0.5)));
}

// The share of 16 hemisphere samples, at two radii, outside the surface.
fn occlusion(p: vec3<f32>, n: vec3<f32>, iso: f32, r: Ray) -> f32 {
    let t = tangent(n);
    let b = cross(n, t);
    var occluded = 0.0;
    for (var i = 0; i < 16; i++) {
        let k = f32(i % 8);
        let phi = k * 2.39996 + f32(i / 8);
        let cos_theta = 1.0 - (k + 0.5) / 8.0;
        let dir = (t * cos(phi) + b * sin(phi)) * sqrt(1.0 - cos_theta * cos_theta) + n * cos_theta;
        let radius = select(0.02, 0.06, i >= 8);
        occluded += inside(iso, fetch(clamp(p + dir * radius * r.scale, vec3(0.0), texture_box())));
    }
    return 1.0 - occluded / 16.0;
}

// The light reaching p along 32 steps, of a hundredth of the box each, towards the light.
fn shadow(p: vec3<f32>, n: vec3<f32>, iso: f32, r: Ray) -> f32 {
    let step = r.light * 0.01 * r.scale;
    var q = p + n * 0.01 * r.scale;
    var light = 1.0;
    for (var i = 0; i < 32 && light > 0.0; i++) {
        q += step;
        if any(q < vec3(0.0)) || any(q > texture_box()) {
            break;
        }
        light = min(light, 1.0 - inside(iso, fetch(q)));
    }
    return light;
}

// Bisection on the trilinear field between a sample under the level and one at or over it.
fn refine(below: vec3<f32>, above: vec3<f32>, iso: f32) -> vec3<f32> {
    var lo = below;
    var hi = above;
    for (var i = 0; i < 6; i++) {
        let mid = 0.5 * (lo + hi);
        if trilinear(mid) >= iso {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return 0.5 * (lo + hi);
}

// A level's colour, kept off the colour map's dark end so that it shows against black.
fn level_color(level: f32) -> vec3<f32> {
    return color_at(mix(0.4, 1.0, scaled(normalised(level))));
}

// Where the trilinear field crosses the level about a nearest-sample hit at sample k (steps for the tail), from the
// sample before it to two after, the tail at most, refined. One at the first sample is where the box cuts the
// emission, and is left as it is. w is 0 where the field stays under the level, as at a lone noisy texel.
fn crossing(r: Ray, k: i32) -> vec4<f32> {
    let iso = modes.level;
    var below = sample_at(r, k - 2);
    for (var j = k - 1; j <= min(k + 2, r.steps); j++) {
        let a = ray_sample(r, max(j, 0));
        if trilinear(a) >= iso {
            if j <= 0 {
                return vec4(a, 1.0);
            }
            return vec4(refine(below, a, iso), 1.0);
        }
        below = a;
    }
    return vec4(0.0);
}

// The surface at p, lit with Blinn-Phong, ambient occlusion and a shadow ray.
fn surface(r: Ray, p: vec3<f32>) -> vec4<f32> {
    let iso = modes.level;
    let n = facing(-gradient(p) * r.scale, r.direction);
    let light = blinn_phong(level_color(iso), n, r, shadow(p, n, iso, r));
    return saturated(vec4(light * occlusion(p, n, iso, r), 1.0), in_selection(r, p));
}

@if(MODE_ISOSURFACE)
fn march(r: Ray) -> vec4<f32> {
    let iso = modes.level;
    var k = 0;
    while k < r.steps {
        let span = cell_span(r, k);
        // Nearest samples four at a time, each hit checked on the trilinear field.
        var j = select(span.end, k, span.max >= iso);
        while j < span.end {
            let a = sample_at(r, j);
            let values = vec4(fetch(a), fetch(a + r.da), fetch(a + 2.0 * r.da), fetch(a + 3.0 * r.da));
            let hit = (values >= vec4(iso)) & (vec4(0, 1, 2, 3) < vec4(span.end - j));
            if any(hit) {
                let i = select(select(select(3, 2, hit.z), 1, hit.y), 0, hit.x);
                let c = crossing(r, j + i);
                if c.w > 0.0 {
                    return surface(r, c.xyz);
                }
                j += i + 1;
            } else {
                j += 4;
            }
        }
        k = span.end;
    }
    if fetch(r.tail) >= iso {
        let c = crossing(r, r.steps);
        if c.w > 0.0 {
            return surface(r, c.xyz);
        }
    }
    return vec4(0.0);
}

// The crossings between two samples of the shells at 1, 2 and 4 times the level, lit and composited front to back.
// Lower shells are fainter, and a shell seen face on fainter still, so the ones inside show through.
fn cross_shells(acc: vec4<f32>, v0: f32, v1: f32, a0: vec3<f32>, a1: vec3<f32>, r: Ray) -> vec4<f32> {
    if max(v0, v1) < modes.level {
        return acc;
    }
    var out = acc;
    let rising = v1 > v0;
    for (var j = 0; j < 3; j++) {
        // Lower shells come first on the way up, higher ones on the way down.
        let i = select(2 - j, j, rising);
        let level = modes.noise_median + (modes.level - modes.noise_median) * exp2(f32(i));
        if (v0 >= level) == (v1 >= level) {
            continue;
        }
        let p = mix(a0, a1, (level - v0) / (v1 - v0));
        // The cheap gradient: noise can cross a shell often.
        let n = facing(-nearest_gradient(p) * r.scale, r.direction);
        let alpha = 0.5 * f32(i + 1) / 3.0 * (0.3 + 0.7 * pow(1.0 - abs(dot(n, r.direction)), 1.5));
        out += (1.0 - out.a) * vec4(blinn_phong(level_color(level), n, r, 1.0) * alpha, alpha);
    }
    return out;
}

@if(MODE_SHELLS)
fn march(r: Ray) -> vec4<f32> {
    var acc = vec4(0.0);
    var a0 = r.start;
    var v0 = filled(fetch(a0));
    var k = 0;
    while k < r.steps && acc.a <= 0.97 {
        let span = cell_span(r, k);
        if span.max < modes.level && v0 < modes.level {
            // No crossing in the cell or into it: carry its last sample on.
            a0 = sample_at(r, span.end - 1);
            v0 = filled(fetch(a0));
        } else {
            var a = sample_at(r, k);
            var j = k;
            for (; j + 4 <= span.end && acc.a <= 0.97; j += 4) {
                let a1 = a + r.da;
                let a2 = a1 + r.da;
                let a3 = a2 + r.da;
                let v = vec4(filled(fetch(a)), filled(fetch(a1)), filled(fetch(a2)), filled(fetch(a3)));
                acc = cross_shells(acc, v0, v.x, a0, a, r);
                acc = cross_shells(acc, v.x, v.y, a, a1, r);
                acc = cross_shells(acc, v.y, v.z, a1, a2, r);
                acc = cross_shells(acc, v.z, v.w, a2, a3, r);
                v0 = v.w;
                a0 = a3;
                a = a3 + r.da;
            }
            for (; j < span.end && acc.a <= 0.97; j++) {
                let v = filled(fetch(a));
                acc = cross_shells(acc, v0, v, a0, a, r);
                v0 = v;
                a0 = a;
                a += r.da;
            }
        }
        k = span.end;
    }
    if acc.a <= 0.97 {
        acc = cross_shells(acc, v0, filled(fetch(r.tail)), a0, r.tail, r);
    }
    return saturated(select(vec4(0.0), vec4(acc.rgb / acc.a, acc.a), acc.a > 0.0), crosses_selection(r));
}
