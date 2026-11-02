# bevy_fits_volume_render

> [!WARNING]
> This is highly experimental. It's a proof of concept, so expect bugs, missing features and breaking changes, and don't rely on it for anything.

A highly experimental proof of concept port of iDaVIE's volume renderer to Rust and Bevy 0.20. It loads a FITS cube, downsamples it to fit a GPU memory budget and renders it with a port of iDaVIE's ray-marching shader (`BasicVolume.cginc`), on the desktop or in a browser over WebGPU. There's no VR.

## Running

```
cargo run --release -- ~/cubes/m25n512_hicube_gal15.fits
cargo run --release -- --help
```

`--config PATH` picks a config file, `--hdu N` picks an HDU (0 is the primary; by default it takes the first image with three or more axes) and any config key can be overridden as `--key value`, e.g. `--gpu-memory-limit-mb 128 --default-color-map Viridis`.

The config lives at `$XDG_CONFIG_HOME/idavie-bevy/config.toml` (`~/.config/idavie-bevy/config.toml` when that's unset) and is written with the defaults on first run. The keys are the rendering subset of iDaVIE's `config.json`, with the same defaults:

```toml
gpu_memory_limit_mb = 384       # texture budget; the cube is downsampled to fit, as in iDaVIE
max_mode_downsampling = true    # max rather than mean when downsampling
max_raymarching_steps = 384
default_color_map = "Inferno"   # any name from iDaVIE's ColorMapEnum
default_scaling_type = "Linear" # Linear, Log, Sqrt, Square, Power or Gamma
jitter = 1.0
```

Drag to orbit, right- or middle-drag to pan, scroll to zoom, shift-drag to select a region, C or Enter to crop to it, R to reset the crop and Esc to clear the selection. A selection covers the two axes facing the camera and keeps the depth range of the previous one, so rotate and shift-drag again to narrow the third axis, or drag one of the blue arrows on its faces or squares on its corners to resize it. Cropping reloads that region from the full resolution data, the same as iDaVIE, and the panel shows the downsampling factors before and after.

`SCREENSHOT=out.png` saves a frame once rendering has settled and exits, which is handy for side by side comparisons with iDaVIE.

## Browser

```
./build-web.sh
python3 -m http.server -d web 8000
```

Then open http://localhost:8000 and pick a file, or pass `?file=<url>` for one served from the same origin. Any other query parameter overrides a config key (`?file=cube.fits&default_color_map=Viridis`), and a `config.toml` next to `index.html` is read if there is one. It needs a WebGPU browser; it's been tested in Chrome on macOS only.

The build needs rustup's toolchain with the `wasm32-unknown-unknown` target and `wasm-bindgen-cli` 0.2.130, matching the `wasm-bindgen` in `Cargo.lock`. The script puts `~/.cargo/bin` first on `PATH` because Homebrew's Rust has no wasm32 standard library. A cold release build takes over ten minutes and the wasm is about 70 MB; `wasm-opt` runs if it's installed.

starlink-ast is Rust apart from the PAL, ERFA and WCSLIB routines it still vendors as C. `wasm32-unknown-unknown` has no libc headers to compile those against, so the script uses Homebrew's `llvm` clang with the musl headers from Homebrew's `emscripten` (headers only, nothing is built with emcc). The C only needs libm and `strcmp`, which Rust's `compiler_builtins` already provides on wasm, so WCS works the same in the browser as natively.

## Differences from iDaVIE

The shader is a line for line port. The Unity specifics are translated rather than copied: the mirrored z axis of Unity's object space, the colour map atlas row order, linear colour space with an sRGB colour map, point sampling with Repeat wrapping and `_Time` for the jitter. Things that are deliberately different or missing:

- no masks, moment maps, catalogues or vignette
- the outlines are opaque line meshes, so the shader's depth clamp lets the volume cover their back edges. iDaVIE draws every edge on top, which reads fine in stereo but not on a flat screen
- no foveated rendering for now, so every pixel gets `max_raymarching_steps`, as in iDaVIE with `foveatedRendering` off
- only point filtering (iDaVIE's `bilinearFiltering` is off by default)
- uncompressed FITS only, using the first three axes, and the cube is always shown with X=Y=Z proportions
- integer BLANK values become NaN; iDaVIE turns off cfitsio's null check and shows them as `BZERO + BSCALE * BLANK`
- downsampling a crop clamps edge blocks to the crop rather than to the full cube, which looks like a latent bug in iDaVIE's `DataCropAndDownsample`

WCS comes from [starlink-ast](https://github.com/Starlink/starlink-ast-rs), the Rust port of AST, with spectral axes shown in MHz or km/s. The port doesn't include AST's `Plot3D`, so the WCS ticks on the bounding box are our own. Each axis is labelled on the box edge that's on the silhouette, at round values found with AST's transforms and formatted with AST's `Format`. The tick spacing is chosen by hand because `astGap` isn't ported either.

## Licence

LGPL-3.0, the same as iDaVIE, since the shader is a port of iDaVIE's and `assets/allmaps.png` is copied from it. iDaVIE's ray marching shader is partly adapted from NVIDIA's OpenGL SDK sample "Render to 3D Texture", so this software contains source code provided by NVIDIA Corporation.
