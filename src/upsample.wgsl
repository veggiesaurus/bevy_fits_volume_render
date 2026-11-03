// A finer texture's placeholder: each texel takes the nearest texel of the coarser one (upload.rs Upsample).
struct Params {
    scale: vec3<f32>,
    offset: vec3<f32>,
    src_max: vec3<u32>,
    dims: vec3<u32>,
};

@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<r32float, write>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 4)
fn upsample(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id >= params.dims) {
        return;
    }
    let s = vec3<u32>(clamp(floor(vec3<f32>(id) * params.scale + params.offset), vec3(0.0), vec3<f32>(params.src_max)));
    textureStore(dst, id.xy, id.z, textureLoad(src, s.xy, s.z, 0));
}
