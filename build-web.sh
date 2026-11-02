#!/bin/sh
set -e
cd "$(dirname "$0")"
# Homebrew's rustc may shadow rustup's on PATH and has no wasm32 target.
export PATH="$HOME/.cargo/bin:$PATH"
# starlink-ast compiles some vendored C (PAL, ERFA, part of WCSLIB). wasm32-unknown-unknown has no libc, so
# borrow Emscripten's musl headers; Rust's compiler_builtins provides the maths functions at link time.
musl="$(brew --prefix emscripten)/libexec/system/lib/libc/musl"
export CC_wasm32_unknown_unknown="$(brew --prefix llvm)/bin/clang"
export AR_wasm32_unknown_unknown="$(brew --prefix llvm)/bin/llvm-ar"
export CFLAGS_wasm32_unknown_unknown="-isystem $musl/arch/emscripten -isystem $musl/arch/generic -isystem $musl/include"

cargo build --release --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --remove-name-section --out-dir web/pkg \
  target/wasm32-unknown-unknown/release/bevy_fits_volume_render.wasm
if command -v wasm-opt >/dev/null; then
  wasm-opt -O web/pkg/bevy_fits_volume_render_bg.wasm -o web/pkg/bevy_fits_volume_render_bg.wasm
fi
