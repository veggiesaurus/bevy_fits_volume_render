# bevy_fits_volume_render

> [!WARNING]
> This is highly experimental. It's a proof of concept, so expect bugs, missing features and breaking changes, and don't rely on it for anything.

A highly experimental proof of concept port of iDaVIE's volume renderer to Rust and Bevy 0.20. It loads a FITS cube, downsamples it to fit a GPU memory budget and renders it with a port of iDaVIE's ray-marching shader (`BasicVolume.cginc`), on the desktop or in a browser over WebGPU. There's no VR.

I'm one of the original authors of [iDaVIE](https://github.com/idia-astro/iDaVIE), and this port has the iDaVIE team's blessing.

## Running

CARTA's protocol definitions are a submodule, so clone with `--recursive`, or run `git submodule update --init` in a clone.

```
cargo run --release -- ~/cubes/m25n512_hicube_gal15.fits
cargo run --release -- --help
```

`--config PATH` picks a config file, `--hdu N` picks an HDU (0 is the primary; by default it takes the first image with two or more axes), `--stokes N` a plane of a Stokes axis (0, the first, by default) and any config key can be overridden as `--key value`, e.g. `--gpu-memory-limit-mb 128 --default-color-map Viridis`.

The config lives at `$XDG_CONFIG_HOME/idavie-bevy/config.toml` (`~/.config/idavie-bevy/config.toml` when that's unset) and is written with the defaults on first run. The keys are the rendering subset of iDaVIE's `config.json`, with the same defaults, plus `jitter`, which iDaVIE sets on the material, the starting thresholds, which iDaVIE always starts at 0 and 1, and the z scale a cube opens at:

```toml
gpu_memory_limit_mb = 384       # texture budget; the cube is downsampled to fit, as in iDaVIE
max_raymarching_steps = 384
default_color_map = "Inferno"   # any name from iDaVIE's ColorMapEnum
default_scaling_type = "Linear" # Linear, Log, Sqrt, Square, Power, Gamma or Asinh
jitter = 1.0
threshold_min = 0.0             # where the colour map starts and ends, as fractions of the data range
threshold_max = 1.0
z_scale = "cube"                # "cube" (as deep as it's wide), "voxels" (cubic voxels) or a number relative to cubic voxels
zfp_tolerance_sigmas = 1.0      # how far a server's cubelets may be off, in the cube's noise σ: 0.5, 1 or 2
cache_memory_mb = 1024          # a server's cubelets kept in memory, compressed (256 by default in the browser)
```

The view only redraws on input (and while data is still arriving), so an idle window leaves the GPU alone. Drag to orbit, right- or middle-drag to pan, scroll to zoom, shift-drag to select a region, C or Enter to crop to it, R to reset the crop and Esc to clear the selection. A selection covers the two axes facing the camera and keeps the depth range of the previous one, so rotate and shift-drag again to narrow the third axis, or drag one of the blue arrows on its faces or squares on its corners to resize it. Cropping keeps the selection, the same as iDaVIE. A crop the texture on screen already holds finely enough is immediate; otherwise it starts from the current view and sharpens as the finer data arrives, and the panel shows the downsampling factors before and after. Every cube comes up the same way, coarse first.

`SCREENSHOT=out.png` saves a frame once rendering has settled and exits, which is handy for side by side comparisons with iDaVIE.

## Server

```
cargo run --release -p fits-server -- --root ~/cubes
cargo run --release -- --server 'ws://127.0.0.1:3002/?token=...' [path/under/the/root.fits]
```

`fits-server` serves the FITS files under `--root` as compressed cubelets, over a subset of CARTA's protocol with messages for volumes added to it (`crates/fits-core/carta-protobuf`, a submodule on carta-protobuf's `angus/volumes` branch). It listens on 127.0.0.1:3002 by default (`--host`, `--port`) and takes its token from `CARTA_AUTH_TOKEN`, or makes one and prints the URL with it. The same port takes `wss://` as well as `ws://`, with a certificate for localhost that fits-server makes on first run and keeps in `~/.config/fits-server`, or one given with `--tls-cert` and `--tls-key`. The native viewer connects over `ws://`. fits-server holds a cube in memory once, however many clients open it, and keeps a closed one until it needs the room; `--memory-limit-gb` caps them all (half the machine's memory by default), and an open that would go over it fails. Replies such as cursor values go ahead of the cubelets queued for a client, and a client that's slow to read holds up the cutting rather than filling the server's memory. The viewer's Server panel lists the folders and cubes under the root and opens or closes one, or a path after the URL opens it straight away. Cubelets come within 0.5, 1 or 2σ of the cube's noise (1σ by default), and the mean pyramid's coarser levels within the same share of their own, lower, noise. The statistics, histogram and cursor values are exact. Masks, noise normalisation and smoothing aren't served yet. The browser connects too, as below.

The viewer keeps the cubelets a server sends, compressed, so that a cube opened again doesn't fetch them again: up to `cache_memory_mb` of them in memory (1024 by default, 256 in the browser), and on disk, a file per cube in `~/.cache/idavie-bevy/cubelets` (or `$XDG_CACHE_HOME`'s) up to 4 GB in all, the least recently used going first. A cubelet kept at a tighter tolerance serves a looser one. Clear the cache on the Server panel, or in the browser in the Cube section, empties both. Local files aren't cached, since the viewer holds the whole cube and cuts its cubelets as fast as it could decode them.

## Browser

```
./build-web.sh
python3 -m http.server -d web 8000
```

A build of `main` is published at https://fits.veggiesaurus.net. Locally, open http://localhost:8000 and pick a file, or pass `?file=<url>` for one served from the same origin or with CORS headers that allow it (`https://` can be left off, as in `?file=static.veggiesaurus.net/cubes/bin5_off2.fits`). The example cubes linked on the page are on R2 behind `static.veggiesaurus.net`. `?hdu=N` picks an HDU, any other query parameter overrides a config key (`?file=cube.fits&default_color_map=Viridis`), and a `config.toml` next to `index.html` is read if there is one. It needs a WebGPU browser; it's been tested in Chrome on macOS only.

`?server=ws://127.0.0.1:3002#token=...` opens the Server panel on a fits-server, and `&path=` opens a cube under its root straight away. The token goes after the `#`, which isn't sent with requests, and the page takes it out of the address bar at once; without it, the panel asks for the token. The viewer sends it to the server in its first message rather than in the URL, since a browser prints a socket's URL in the console when it can't connect. The worker holds the socket and decodes the cubelets, as it does a file's, and keeps the server's cubelets in the site's origin private file system (OPFS) where the native viewer keeps them on disk.

The example cubes on the page are baked, so that they open without downloading the whole cube. `fits-bake` writes each cube's max and mean pyramids as cubelets within 1σ of its noise, after an index of them, to `CUBE.fits.cubelets` beside it:

```
cargo run --release -p fits-server --example fits-bake -- [--sigmas S] [--exact-level-0] [--out DIR] cube.fits...
```

`?file=` with a `.cubelets` URL reads the index, then only the cubelets the view needs, by range request, and keeps them in OPFS as it does a server's, which Clear the cache in the Cube section empties. Its levels are the viewer's at the default budget, which a viewer with another budget or fewer texture layers takes as they are, showing the finest that fits. The cursor's values come from level 0's cubelets, so they're shown as within the tolerance (`≈ 1.2e-2 ± 4.0e-3`), unless `--exact-level-0` kept level 0 exact, which makes a cube's file three to seven times larger. Masks, noise normalisation and smoothing aren't available for a baked cube, as for a server's.

The published site is https. In Chrome, `?server=ws://127.0.0.1:3002` works once you allow its Local Network Access prompt for the site. Safari blocks `ws://` from an https page as mixed content, so it needs `wss://` and a certificate it trusts. Once, after fits-server has run and made its certificate:

```
security add-trusted-cert -p ssl -k ~/Library/Keychains/login.keychain-db ~/.config/fits-server/localhost.pem
```

macOS asks for your password, and trusts that one certificate for TLS servers (it's for localhost and 127.0.0.1 only, and can't sign others). Safari needs this step: accepting its warning for https://127.0.0.1:3002/ in a tab isn't enough, and the socket still fails. That page is a check instead, since it opens with no warning once the certificate is trusted. Then `https://fits.veggiesaurus.net/?server=wss://127.0.0.1:3002#token=...` connects, and keeps doing so across restarts until fits-server renews the certificate, after about two years. Chrome on macOS trusts it too. A certificate from mkcert (`mkcert -install`, then `mkcert localhost 127.0.0.1`, given as `--tls-cert` and `--tls-key`) is trusted by both without the step. Serving the page locally, as above, needs none of this. (Checked in Safari, and in headless Chrome 156 and Playwright's WebKit 26.6.)

The file never reaches the page's thread. It goes to a Web Worker (`web/worker.js` and the `fits-worker` crate, a second, small wasm module), which downloads or reads it, holds the cube, builds its levels, cuts cubelets and answers the cursor readout, so the page only renders. The two talk in `postMessage`s of transferred `ArrayBuffer`s, and the worker drops one file before reading the next, so a cube is only ever in memory once. Natively the same code runs in-process on Bevy's task pools.

The build needs rustup's toolchain with the `wasm32-unknown-unknown` target and `wasm-bindgen-cli` 0.2.130, matching the `wasm-bindgen` in `Cargo.lock`. The script puts `~/.cargo/bin` first on `PATH` because Homebrew's Rust has no wasm32 standard library. The script builds `--release` with wasm `simd128`, which every WebGPU browser supports; the thin LTO `dist` profile (`cargo build --profile dist`, for a native build to ship) made the wasm larger and no faster. A cold build is slow, and the viewer's wasm is about 42 MB (the worker's about 170 kB) once `wasm-opt -O1` (from Homebrew's `binaryen`) has run, which compresses smaller than `-O`; the script skips it if it isn't installed.

starlink-ast is Rust apart from the PAL, ERFA and WCSLIB routines it still vendors as C. `wasm32-unknown-unknown` has no libc headers to compile those against, so the script uses Homebrew's `llvm` clang with the musl headers from Homebrew's `emscripten` (headers only, nothing is built with emcc). `WASM_LLVM` and `WASM_MUSL` point it at others instead, which is how the Pages workflow builds without Homebrew installs. The C only needs libm and `strcmp`, which Rust's `compiler_builtins` already provides on wasm, so WCS works the same in the browser as natively.

## Differences from iDaVIE

The shader takes the same samples as iDaVIE's, apart from the step changes below, and adds guards so a constant cube or equal thresholds don't divide zero by zero. It's rearranged for speed (an unrolled loop, and Maximum rays stop once their colour can't change) without changing the image. The Unity specifics are translated rather than copied: the mirrored z axis of Unity's object space, the colour map atlas row order, linear colour space with an sRGB colour map, point sampling with Repeat wrapping and `_Time` for the jitter. Things that are deliberately different or missing:

- no mask files, catalogues, vignette or cursor voxel box
- the outlines are opaque line meshes, so the shader's depth clamp lets the volume cover their back edges, and edges behind the box are drawn dark grey. iDaVIE draws every edge on top, which reads fine in stereo but not on a flat screen
- after a crop the outline shrinks to the crop and the camera re-frames it, and again on reset. iDaVIE keeps the full cube's outline with the crop inside it and leaves the camera where it is
- no foveated rendering for now, so every pixel gets the full step count, as in iDaVIE with `foveatedRendering` off. iDaVIE's default config turns it on, so turn it off in its `config.json` for side by side comparisons
- only point filtering (iDaVIE's `bilinearFiltering` is off by default)
- tile-compressed `.fits.fz` is read as fpack writes it with RICE_1, GZIP_1, GZIP_2 or no compression, but not PLIO_1 or HCOMPRESS_1 (funpack those first)
- the first three axes longer than 1 are shown (a 2D image is one plane), and the first plane of any others, except that a Stokes axis is left out when three others are longer than 1. A local file's Stokes plane can then be picked with `--stokes`, but not yet a server's or one in the browser
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
- x and y keep the voxels' proportions, where iDaVIE always draws them the same length. So the z scale's Cube preset is iDaVIE's X=Y=Z and Cubic voxels its X=Y only for a square cube, and any other scale can be typed in
- WCS ticks are drawn on the box by default (the WCS axes checkbox turns them off); iDaVIE has none
- percentile buttons (90 to 100%) set the thresholds to keep the middle share of the voxels, from an exact histogram built while the cube loads (about 20 ms per GB natively, 0.2 s per GB in the browser); iDaVIE has none
- the near plane follows the camera in, so a small crop doesn't end up behind it, and the far plane and zoom limits follow the region shown, so a deep box isn't cut off
- the step count is capped at two per texel along the texture's diagonal, so a small crop or a heavily downsampled cube takes fewer steps than in iDaVIE. Voxel edges can shift by a fraction of a voxel compared with iDaVIE
- the view only redraws on input, so the jitter noise holds still while nothing changes; iDaVIE renders every frame
- rendering modes past Maximum and Average: depth cued and attenuated maximum, peak and mean channel colours, emission-absorption with MIDA, S/N opacity and lighting, an isosurface, translucent shells and a clipped moment 0. Asinh scaling starts from the noise, and the noise can be normalised, the cube smoothed and a SoFiA-style smooth-and-clip mask made, all by the data source (the worker, in the browser). The noise and the mask come from the shown texture's level, so from a downsampled cube when that's what's shown

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
