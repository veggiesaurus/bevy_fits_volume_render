# bevy_fits_volume_render

> [!WARNING]
> This is highly experimental. It's a proof of concept, so expect bugs, missing features and breaking changes, and don't rely on it for anything.

A highly experimental proof of concept port of iDaVIE's volume renderer to Rust and Bevy 0.20. It loads a FITS cube, downsamples it to fit a GPU memory budget and renders it with a port of iDaVIE's ray-marching shader (`BasicVolume.cginc`), on the desktop or in a browser over WebGPU. There's no VR.

I'm one of the original authors of [iDaVIE](https://github.com/idia-astro/iDaVIE), and this port has the iDaVIE team's blessing.

## Running

```
cargo run --release -- ~/cubes/m25n512_hicube_gal15.fits
cargo run --release -- --help
```

`--config PATH` picks a config file, `--hdu N` picks an HDU (0 is the primary; by default it takes the first image with two or more axes) and any config key can be overridden as `--key value`, e.g. `--gpu-memory-limit-mb 128 --default-color-map Viridis`.

The config lives at `$XDG_CONFIG_HOME/idavie-bevy/config.toml` (`~/.config/idavie-bevy/config.toml` when that's unset) and is written with the defaults on first run. The keys are the rendering subset of iDaVIE's `config.json`, with the same defaults, plus `jitter`, which iDaVIE sets on the material, and the starting thresholds, which iDaVIE always starts at 0 and 1:

```toml
gpu_memory_limit_mb = 384       # texture budget; the cube is downsampled to fit, as in iDaVIE
max_raymarching_steps = 384
default_color_map = "Inferno"   # any name from iDaVIE's ColorMapEnum
default_scaling_type = "Linear" # Linear, Log, Sqrt, Square, Power or Gamma
jitter = 1.0
threshold_min = 0.0             # where the colour map starts and ends, as fractions of the data range
threshold_max = 1.0
```

The view only redraws on input (and while data is still arriving), so an idle window leaves the GPU alone. Drag to orbit, right- or middle-drag to pan, scroll to zoom, shift-drag to select a region, C or Enter to crop to it, R to reset the crop and Esc to clear the selection. A selection covers the two axes facing the camera and keeps the depth range of the previous one, so rotate and shift-drag again to narrow the third axis, or drag one of the blue arrows on its faces or squares on its corners to resize it. Cropping keeps the selection, the same as iDaVIE. A crop the texture on screen already holds finely enough is immediate; otherwise it starts from the current view and sharpens as the finer data arrives, and the panel shows the downsampling factors before and after. Every cube comes up the same way, coarse first.

`SCREENSHOT=out.png` saves a frame once rendering has settled and exits, which is handy for side by side comparisons with iDaVIE.

## Browser

```
./build-web.sh
python3 -m http.server -d web 8000
```

A build of `main` is published at https://fits.veggiesaurus.net. Locally, open http://localhost:8000 and pick a file, or pass `?file=<url>` for one served from the same origin or with CORS headers that allow it (`https://` can be left off, as in `?file=static.veggiesaurus.net/cubes/bin5_off2.fits`). The example cubes linked on the page are on R2 behind `static.veggiesaurus.net`. `?hdu=N` picks an HDU, any other query parameter overrides a config key (`?file=cube.fits&default_color_map=Viridis`), and a `config.toml` next to `index.html` is read if there is one. It needs a WebGPU browser; it's been tested in Chrome on macOS only.

The build needs rustup's toolchain with the `wasm32-unknown-unknown` target and `wasm-bindgen-cli` 0.2.130, matching the `wasm-bindgen` in `Cargo.lock`. The script puts `~/.cargo/bin` first on `PATH` because Homebrew's Rust has no wasm32 standard library. The script builds `--release` with wasm `simd128`, which every WebGPU browser supports; the thin LTO `dist` profile (`cargo build --profile dist`, for a native build to ship) made the wasm larger and no faster. A cold build is slow, and the wasm is about 42 MB once `wasm-opt -O1` (from Homebrew's `binaryen`) has run, which compresses smaller than `-O`; the script skips it if it isn't installed.

starlink-ast is Rust apart from the PAL, ERFA and WCSLIB routines it still vendors as C. `wasm32-unknown-unknown` has no libc headers to compile those against, so the script uses Homebrew's `llvm` clang with the musl headers from Homebrew's `emscripten` (headers only, nothing is built with emcc). `WASM_LLVM` and `WASM_MUSL` point it at others instead, which is how the Pages workflow builds without Homebrew installs. The C only needs libm and `strcmp`, which Rust's `compiler_builtins` already provides on wasm, so WCS works the same in the browser as natively.

## Differences from iDaVIE

The shader takes the same samples as iDaVIE's, apart from the step changes below, and adds guards so a constant cube or equal thresholds don't divide zero by zero. It's rearranged for speed (an unrolled loop, and Maximum rays stop once their colour can't change) without changing the image. The Unity specifics are translated rather than copied: the mirrored z axis of Unity's object space, the colour map atlas row order, linear colour space with an sRGB colour map, point sampling with Repeat wrapping and `_Time` for the jitter. Things that are deliberately different or missing:

- no masks, moment maps, catalogues, vignette or cursor voxel box
- the outlines are opaque line meshes, so the shader's depth clamp lets the volume cover their back edges, and edges behind the box are drawn dark grey. iDaVIE draws every edge on top, which reads fine in stereo but not on a flat screen
- after a crop the outline shrinks to the crop and the camera re-frames it, and again on reset. iDaVIE keeps the full cube's outline with the crop inside it and leaves the camera where it is
- no foveated rendering for now, so every pixel gets the full step count, as in iDaVIE with `foveatedRendering` off. iDaVIE's default config turns it on, so turn it off in its `config.json` for side by side comparisons
- only point filtering (iDaVIE's `bilinearFiltering` is off by default)
- uncompressed FITS only (funpack a tile-compressed `.fits.fz` first), showing the first three axes longer than 1 (a 2D image is one plane) and the first plane of any others (Stokes I of an IQU cube), and the cube is always shown with X=Y=Z proportions
- integer BLANK values become NaN; iDaVIE turns off cfitsio's null check and shows them as `BZERO + BSCALE * BLANK`
- the cube is downsampled to a fixed set of power-of-two levels, plus iDaVIE's factors for the whole cube when they aren't one of them, with blocks aligned to the whole cube. So a crop can get up to 2.6x fewer texels than in iDaVIE, whose factors step through 1, 2, 3, 4, 5
- Maximum reads a max-downsampled copy and Average a mean-downsampled one, where iDaVIE's config picks one for both
- the downsampling factors are checked against the rounded up texture size, for the memory budget as well as the 2048 texel limit. iDaVIE checks the rounded down size, which lets an axis come out at 2049 texels (over the 3D texture limit) or the texture go slightly over budget
- the step length is iDaVIE's, the square root of the slice diagonal over `max_raymarching_steps`, but a ray can take up to that square root times `max_raymarching_steps` steps. iDaVIE stops at `max_raymarching_steps`, so on the full cube its rays longer than about 1.32 (corner-on views) jump to a single sample one step short of the far face
- in Average mode the last sample is weighted by the length left over. iDaVIE's weight is negative, and on rays one or two steps long it can take the summed length to zero or below, so the average blows up or changes sign at the edges
- infinite voxels are left out of the range and statistics along with NaN ones, since one would scale every other voxel to zero. The mean and standard deviation count finite voxels only, where iDaVIE's divide by every voxel, NaNs included
- the selection outline is only white if cropping to it would actually load at full resolution, so the 2048 texel limit counts as well as the memory budget. It stays white or yellow after the drag, where iDaVIE swaps it for a pulsing white box
- everything outside the selection stays desaturated for as long as there is one, including after a crop or reset. iDaVIE only desaturates while the selection is being drawn
- a selection is clamped to the visible crop, so cropping can only narrow the view. In iDaVIE a selection can reach outside the crop, and cropping to it widens the view without a reset
- WCS ticks are drawn on the box by default (the WCS axes checkbox turns them off); iDaVIE has none
- percentile buttons (90 to 100%) set the thresholds to keep the middle share of the voxels, from an exact histogram built while the cube loads (about 20 ms per GB natively, 0.2 s per GB in the browser); iDaVIE has none
- the near plane follows the camera in, so a small crop doesn't end up behind it
- the step count is capped at two per texel along the texture's diagonal, so a small crop or a heavily downsampled cube takes fewer steps than in iDaVIE. Voxel edges can shift by a fraction of a voxel compared with iDaVIE
- the view only redraws on input, so the jitter noise holds still while nothing changes; iDaVIE renders every frame

WCS comes from [starlink-ast](https://github.com/Starlink/starlink-ast-rs), the Rust port of AST, with spectral axes shown in MHz or km/s. The port doesn't include AST's `Plot3D`, so the WCS ticks on the bounding box are our own. Each axis is labelled on the box edge that's on the silhouette, at round values found with AST's transforms and formatted with AST's `Format`. The tick spacing is chosen by hand because `astGap` isn't ported either.

## Example data

The example cubes linked from the web page are on R2 behind `static.veggiesaurus.net`. Credit the surveys if you use them:

- NGC 4631 and NGC 5055, HI: [HALOGAS](https://zenodo.org/records/3715549) data release 1 (Heald et al. 2011, A&A 526, A118), Westerbork Synthesis Radio Telescope, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- M83, HI: [THINGS](https://things.www3.mpia.de/) (Walter et al. 2008, AJ 136, 2563), the natural weighted cube
- M83, CO(2-1): [PHANGS-ALMA](https://www.canfar.net/storage/vault/file/phangs/RELEASES/PHANGS-ALMA/by_product/cubes/nativeres/ngc5236_12m%2B7m%2Btp_co21.fits) (Leroy et al. 2021, ApJS 257, 43 and ApJS 255, 19), 12m+7m+TP at native resolution; the ALMA project codes are in the release README
- M33 and IC 10, HI: [LGLBS](https://www.canfar.net/storage/vault/list/LGLBS/RELEASES/LGLBS-HI-v1.0) v1.0 (Koch et al. 2025, ApJS 279, 35), VLA C+D configurations plus GBT at 30". M33 is the 2.1 km/s cube cropped to -55.5 to -317.3 km/s, which keeps M33 and drops the Milky Way's HI; IC 10 is the 0.8 km/s cube as released
- the simulated galaxy: a [MARTINI](https://github.com/kyleaoman/martini) mock observation by K. Oman, converted from float64 to float32
- HD 163296, CO(2-1): ALMA Science Verification data
- the CO 1-0 test cube: from CARTA's test images

## Licence

LGPL-3.0, the same as iDaVIE, since the shader is a port of iDaVIE's and `assets/allmaps.png` is copied from it. iDaVIE's ray marching shader is partly adapted from NVIDIA's OpenGL SDK sample "Render to 3D Texture", so this software contains source code provided by NVIDIA Corporation.
