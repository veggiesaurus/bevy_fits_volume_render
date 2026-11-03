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
};

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> params: VolumeParams;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var data_cube: texture_3d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(2) var data_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(3) var color_map: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(4) var color_map_sampler: sampler;

const LOG: u32 = 1u;
const SQRT: u32 = 2u;
const SQUARE: u32 = 3u;
const POWER: u32 = 4u;
const GAMMA: u32 = 5u;

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

// data_sampler does iDaVIE's point filtering and Repeat wrapping (volume.rs cube_image).
fn sample_cube(position: vec3<f32>) -> f32 {
    return textureSampleLevel(data_cube, data_sampler, position, 0.0).r;
}

struct Ray {
    value: f32,
    cumulative_length: f32,
    in_highlight: bool,
    has_value: bool,
};

fn maximum(ray: Ray, value: f32, position: vec3<f32>, highlight_min: vec3<f32>, highlight_max: vec3<f32>) -> Ray {
    var r = ray;
    if !is_nan(value) && value >= r.value {
        r.in_highlight = in_box(position, highlight_min, highlight_max) || (r.in_highlight && value == r.value);
        r.value = value;
        r.has_value = true;
    }
    return r;
}

fn average(ray: Ray, value: f32, position: vec3<f32>, step_length: f32, highlight_min: vec3<f32>, highlight_max: vec3<f32>) -> Ray {
    var r = ray;
    if !is_nan(value) {
        r.value += value * step_length;
        r.cumulative_length += step_length;
        r.in_highlight = r.in_highlight || in_box(position, highlight_min, highlight_max);
        r.has_value = true;
    }
    return r;
}

fn accumulate(ray: Ray, position: vec3<f32>, step_length: f32, highlight_min: vec3<f32>, highlight_max: vec3<f32>) -> Ray {
    let value = sample_cube(position);
    if params.projection_mode == 1u {
        return average(ray, value, position, step_length, highlight_min, highlight_max);
    }
    return maximum(ray, value, position, highlight_min, highlight_max);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Distance to opaque geometry along the ray (iDaVIE reads _CameraDepthTexture), here the outline lines.
    var opaque_depth = 3.402823466e+38;
    @if(DEPTH_PREPASS) {
        let depth = prepass_depth(in.position, 0u);
        // Reverse-Z: 0 is the far plane, so nothing opaque at this pixel.
        if depth > 0.0 {
            let world = position_ndc_to_world(vec3(frag_coord_to_ndc(in.position).xy, depth));
            let local = (get_local_from_world(in.instance_index) * vec4(world, 1.0)).xyz * UNITY_FROM_LOCAL;
            opaque_depth = length(local - in.ray_origin);
        }
    }

    // iDaVIE's step, sqrt(max_length) / max_steps, with the step count lifted so the longest rays (sqrt(3) on
    // the full cube) reach the far side rather than stopping at max_steps. Capped at about two steps per texel
    // along the texture's diagonal, so a small texture (a crop, or a heavily downsampled cube) costs less.
    let max_length = length(params.slice_max - params.slice_min);
    let max_steps = min(sqrt(max_length) * f32(params.max_steps), ceil(2.0 * length(vec3<f32>(textureDimensions(data_cube)))));
    let step_length = max_length / max_steps;

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

    let region_scale = 1.0 / (params.slice_max - params.slice_min);
    let region_offset = -(params.slice_min + 0.5);
    var position = (p_near + rand_vector + region_offset) * region_scale;
    let adjusted_step_vector = step_vector * step_length * region_scale;
    let highlight_min = (params.highlight_min + region_offset + 0.5) * region_scale;
    let highlight_max = (params.highlight_max + region_offset + 0.5) * region_scale;

    // The guard keeps a constant cube from dividing zero by zero.
    let scale_range = select(params.scale_max - params.scale_min, 1.0, params.scale_max == params.scale_min);

    var ray = Ray(select(-3.402823466e+38, 0.0, params.projection_mode == 1u), 0.0, false, false);
    // Unrolled so the texture fetches overlap; the step positions are the same as one step at a time.
    let s = adjusted_step_vector;
    let h_min = highlight_min;
    let h_max = highlight_max;
    if params.projection_mode == 1u {
        var i = 0;
        for (; i + 4 <= required_steps; i += 4) {
            let p1 = position + s;
            let p2 = p1 + s;
            let p3 = p2 + s;
            let v0 = sample_cube(position);
            let v1 = sample_cube(p1);
            let v2 = sample_cube(p2);
            let v3 = sample_cube(p3);
            ray = average(ray, v0, position, step_length, h_min, h_max);
            ray = average(ray, v1, p1, step_length, h_min, h_max);
            ray = average(ray, v2, p2, step_length, h_min, h_max);
            ray = average(ray, v3, p3, step_length, h_min, h_max);
            position = p3 + s;
        }
        for (; i < required_steps; i++) {
            ray = average(ray, sample_cube(position), position, step_length, h_min, h_max);
            position += s;
        }
    } else {
        // Once the maximum is past threshold_max the colour can't change, unless a selection is shown.
        let can_stop = params.highlight_saturate_factor == 1.0;
        var i = 0;
        for (; i + 8 <= required_steps; i += 8) {
            let p1 = position + s;
            let p2 = p1 + s;
            let p3 = p2 + s;
            let p4 = p3 + s;
            let p5 = p4 + s;
            let p6 = p5 + s;
            let p7 = p6 + s;
            let v0 = sample_cube(position);
            let v1 = sample_cube(p1);
            let v2 = sample_cube(p2);
            let v3 = sample_cube(p3);
            let v4 = sample_cube(p4);
            let v5 = sample_cube(p5);
            let v6 = sample_cube(p6);
            let v7 = sample_cube(p7);
            ray = maximum(ray, v0, position, h_min, h_max);
            ray = maximum(ray, v1, p1, h_min, h_max);
            ray = maximum(ray, v2, p2, h_min, h_max);
            ray = maximum(ray, v3, p3, h_min, h_max);
            ray = maximum(ray, v4, p4, h_min, h_max);
            ray = maximum(ray, v5, p5, h_min, h_max);
            ray = maximum(ray, v6, p6, h_min, h_max);
            ray = maximum(ray, v7, p7, h_min, h_max);
            position = p7 + s;
            if can_stop && (ray.value - params.scale_min) / scale_range >= params.threshold_max {
                i = required_steps;
                break;
            }
        }
        for (; i < required_steps; i++) {
            ray = maximum(ray, sample_cube(position), position, h_min, h_max);
            position += s;
        }
    }
    let remaining_step_length = total_length - f32(required_steps + 1) * step_length - length(rand_vector);
    position += step_vector * remaining_step_length * region_scale;
    // iDaVIE weights this sample by remaining_step_length, which is negative and can zero the average.
    let leftover_length = max(total_length - f32(required_steps) * step_length, 0.0);
    ray = accumulate(ray, position, leftover_length, highlight_min, highlight_max);
    if params.projection_mode == 1u && ray.cumulative_length > 0.0 {
        ray.value /= ray.cumulative_length;
    }

    if !ray.has_value {
        return vec4(0.0);
    }

    var ray_value = (ray.value - params.scale_min) / scale_range;
    ray_value = clamp(ray_value, params.threshold_min, params.threshold_max);
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

    let color_fraction = select(params.highlight_saturate_factor, 1.0, ray.in_highlight);
    let color_map_color = vec4(textureSampleLevel(color_map, color_map_sampler, vec2(x, color_map_offset), 0.0).rgb, x);
    return mix(vec4(x), color_map_color, color_fraction);
}
