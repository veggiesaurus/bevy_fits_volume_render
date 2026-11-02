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

// Point filtering with Repeat wrapping, which is what iDaVIE's `new Texture3D` gets by default.
fn sample_cube(position: vec3<f32>) -> f32 {
    let size = vec3<i32>(textureDimensions(data_cube));
    let texel = vec3<i32>(floor(position * vec3<f32>(size)));
    return textureLoad(data_cube, ((texel % size) + size) % size, 0).r;
}

struct Ray {
    value: f32,
    cumulative_length: f32,
    in_highlight: bool,
    has_value: bool,
};

fn accumulate(ray: Ray, position: vec3<f32>, step_length: f32, highlight_min: vec3<f32>, highlight_max: vec3<f32>) -> Ray {
    var r = ray;
    let step_value = sample_cube(position);
    if is_nan(step_value) {
        return r;
    }
    if params.projection_mode == 1u {
        r.value += step_value * step_length;
        r.cumulative_length += step_length;
        r.in_highlight = r.in_highlight || in_box(position, highlight_min, highlight_max);
        r.has_value = true;
    } else if step_value >= r.value {
        r.in_highlight = in_box(position, highlight_min, highlight_max) || (r.in_highlight && step_value == r.value);
        r.value = step_value;
        r.has_value = true;
    }
    return r;
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

    let direction = normalize(in.ray_direction);
    let t = intersect_box(in.ray_origin, direction, params.slice_min, params.slice_max);
    // Early exit of pixels missing the bounding box or occluded by opaque objects
    if t.x > t.y || t.y < 0.0 || opaque_depth < t.x {
        return vec4(0.0);
    }
    let t_near = max(0.0, t.x);
    let t_far = min(t.y, opaque_depth);

    let p_near = in.ray_origin + direction * t_near + 0.5;
    let p_far = in.ray_origin + direction * t_far + 0.5;
    let ray_delta = p_far - p_near;
    let total_length = length(ray_delta);

    // iDaVIE takes the square root of max_length here, which leaves max_steps short of the longest rays.
    let max_length = length(params.slice_max - params.slice_min);
    let step_length = max_length / f32(params.max_steps);
    let step_vector = normalize(ray_delta);
    let required_steps = i32(clamp(total_length / step_length, 0.0, f32(params.max_steps)));

    // Unity's _Time.xy is (t / 20, t)
    let rand_vector = nrand(in.position.xy + vec2(globals.time / 20.0, globals.time)) * step_vector * step_length * params.jitter;

    let region_scale = 1.0 / (params.slice_max - params.slice_min);
    let region_offset = -(params.slice_min + 0.5);
    var position = (p_near + rand_vector + region_offset) * region_scale;
    let adjusted_step_vector = step_vector * step_length * region_scale;
    let highlight_min = (params.highlight_min + region_offset + 0.5) * region_scale;
    let highlight_max = (params.highlight_max + region_offset + 0.5) * region_scale;

    var ray = Ray(select(-3.402823466e+38, 0.0, params.projection_mode == 1u), 0.0, false, false);
    for (var i = 0; i < required_steps; i++) {
        ray = accumulate(ray, position, step_length, highlight_min, highlight_max);
        position += adjusted_step_vector;
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

    // The guards keep a constant cube, or equal thresholds, from dividing zero by zero.
    var ray_value = (ray.value - params.scale_min) / max(params.scale_max - params.scale_min, 1e-30);
    ray_value = clamp(ray_value, params.threshold_min, params.threshold_max);
    // Unity's v axis points up the atlas, wgpu's points down, so no "1.0 -" here.
    let color_map_offset = (0.5 + f32(params.color_map_index)) / f32(params.num_color_maps);
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
