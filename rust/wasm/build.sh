#!/usr/bin/env bash
# Build the browser demo: wasm module + JS glue into rust/wasm/www/pkg, then serve www/.
#   ./build.sh [export_dir]    e.g. ./build.sh ../../artifacts/random_d8
# Requires: rustup target add wasm32-unknown-unknown; cargo install wasm-bindgen-cli --version <same as Cargo.lock>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"   # rust/wasm
EXPORT="${1:+$(cd "$1" && pwd)}"
cd "$HERE/.."                               # rust/ (workspace root)
cargo build --release -p vlm-wasm --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir "$HERE/www/pkg" target/wasm32-unknown-unknown/release/vlm_wasm.wasm
if [ -n "$EXPORT" ]; then
  mkdir -p "$HERE/www/model"
  cp "$EXPORT/config.json" "$EXPORT/model.safetensors" "$EXPORT/tokenizer.json" "$HERE/www/model/"
fi
echo "serve with: python -m http.server -d $HERE/www 8000   (then open http://localhost:8000)"
