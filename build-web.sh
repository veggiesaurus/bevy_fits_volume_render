#!/bin/sh
set -e
cd "$(dirname "$0")"
# Homebrew's rustc may shadow rustup's on PATH and has no wasm32 target.
export PATH="$HOME/.cargo/bin:$PATH"
# starlink-ast compiles some vendored C (PAL, ERFA, part of WCSLIB). wasm32-unknown-unknown has no libc, so
# borrow Emscripten's musl headers; Rust's compiler_builtins provides the maths functions at link time.
# brew --prefix succeeds for a formula that isn't installed, so check the files themselves.
musl="$(brew --prefix emscripten)/libexec/system/lib/libc/musl"
llvm="$(brew --prefix llvm)/bin"
[ -d "$musl" ] || { echo "Needs Homebrew's emscripten for its musl headers: brew install emscripten" >&2; exit 1; }
[ -x "$llvm/clang" ] || { echo "Needs Homebrew's llvm for a wasm32 clang: brew install llvm" >&2; exit 1; }
export CC_wasm32_unknown_unknown="$llvm/clang"
export AR_wasm32_unknown_unknown="$llvm/llvm-ar"
export CFLAGS_wasm32_unknown_unknown="-isystem $musl/arch/emscripten -isystem $musl/arch/generic -isystem $musl/include"

# simd128 lets LLVM vectorise the FITS decoding and statistics; every WebGPU browser supports it.
export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS="-C target-feature=+simd128"
cargo build --profile dist --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --remove-name-section --out-dir web/pkg \
  target/wasm32-unknown-unknown/dist/bevy_fits_volume_render.wasm
if command -v wasm-opt >/dev/null; then
  wasm-opt -O web/pkg/bevy_fits_volume_render_bg.wasm -o web/pkg/bevy_fits_volume_render_bg.wasm
fi
