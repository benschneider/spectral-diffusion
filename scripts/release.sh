#!/usr/bin/env bash
# Helper script to run basic checks and publish RIFFT to crates.io and PyPI.
set -euo pipefail

if [[ -z "${CARGO_REGISTRY_TOKEN:-}" ]]; then
  echo "warning: CARGO_REGISTRY_TOKEN not set; cargo publish will prompt unless cached login exists" >&2
fi
if [[ -z "${PIP_INDEX_URL:-}" ]]; then
  echo "info: using default PyPI index" >&2
fi

cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test

python -m pip install --upgrade pip maturin

read -p "Ready to publish crates.io (y/N)? " ans
if [[ "${ans,,}" == "y" ]]; then
  cargo publish
fi

read -p "Ready to publish wheels to PyPI via maturin (y/N)? " ans2
if [[ "${ans2,,}" == "y" ]]; then
  RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-nightly} maturin publish --release --features python,simd_avx2 "$@"
fi
