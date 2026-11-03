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

// z in layers.
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
    // tinting the front edges of the outline.
    if t.x > t.y || t.y < 0.0 || opaque_depth < t.x + step_length {
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

    // Unity's _Time.xy is (t / 20, t)
    let rand_vector = nrand(in.position.xy + vec2(globals.time / 20.0, globals.time)) * step_vector * step_length * params.jitter;

    let region_scale = 1.0 / (params.texture_max - params.texture_min);
    let region_offset = -(params.texture_min + 0.5);
    var position = (p_near + rand_vector + region_offset) * region_scale;
    let s = step_vector * step_length * region_scale;
    let h_min = (params.highlight_min + region_offset + 0.5) * region_scale;
    let h_max = (params.highlight_max + region_offset + 0.5) * region_scale;

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
        let go = params.grid_offset;
        var k = 1;
        while k < required_steps {
            let a = start + f32(k) * da;
            let cell = clamp(floor((a + go) / params.cell_size), vec3(0.0), vec3<f32>(params.grid_cells - 1u));
            let lo = cell * params.cell_size - go;
            let hi = min(lo + params.cell_size, zs);
            let t = select(vec3(F32_MAX), (select(lo, hi, da > vec3(0.0)) - start) * inv_da, da != vec3(0.0));
            let k_exit = clamp(i32(ceil(min(min(min(t.x, t.y), t.z), f32(required_steps)))), k + 1, required_steps);
            let entry = grid_entry(vec3<u32>(cell));
            let cell_max = bitcast<f32>(entry.y);
            if cell_max >= floor_value && (cell_max > top.x || (cell_max == top.x && track)) {
                if track && all(hi >= l_min) && all(lo <= l_max) {
                    for (var j = k; j < k_exit; j++) {
                        let b = start + f32(j) * da;
                        top = max_sample(top, fetch(b), b, l_min, l_max);
                    }
                } else if entry.x != 0u {
                    top.x = max(top.x, segment_max(a, da, k_exit - k, true));
                } else {
                    top.x = max(top.x, segment_max(a, da, k_exit - k, false));
                }
            }
            k = k_exit;
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

// Thresholds, scaling and the colour map, for a ray's value normalised to the scale range.
fn shade(normalised: f32, in_highlight: bool) -> vec4<f32> {
    let ray_value = clamp(normalised, params.threshold_min, params.threshold_max);
    // Unity's v axis points up the atlas, wgpu's points down, so no "1.0 -" here.
    let color_map_offset = (0.5 + f32(params.color_map_index)) / f32(params.num_color_maps);
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

    x = clamp(x - params.scale_bias, 0.0, 1.0);
    x = clamp((x - 0.5) * params.scale_contrast + 0.5, 0.0, 1.0);

    let color_fraction = select(params.highlight_saturate_factor, 1.0, in_highlight);
    let color_map_color = vec4(textureSampleLevel(color_map, color_map_sampler, vec2(x, color_map_offset), 0.0).rgb, x);
    return mix(vec4(x), color_map_color, color_fraction);
}
