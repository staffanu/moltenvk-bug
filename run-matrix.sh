#!/usr/bin/env bash
# Run the 4 (or 5) variants and print a compact summary.
set -euo pipefail
cd "$(dirname "$0")"

# Locate validation layer manifest from a Homebrew-installed loader + LunarG SDK setup.
if [[ -z "${VK_LAYER_PATH:-}" ]]; then
    sdk_glob=(/Users/staffanu/VulkanSDK/*/macOS/share/vulkan/explicit_layer.d)
    if [[ -d "${sdk_glob[0]}" ]]; then
        export VK_LAYER_PATH="${sdk_glob[0]}"
    fi
fi

ITER="${ITER:-10000}"
BIN=./build/repro

if [[ ! -x "$BIN" ]]; then
    cmake --build build
fi

run_one() {
    local barrier="$1" layout="$2"
    echo "============================================================"
    echo "VARIANT barrier=$barrier layout=$layout iterations=$ITER"
    echo "============================================================"
    "$BIN" --barrier="$barrier" --layout="$layout" --iterations="$ITER" || true
    echo
}

run_one BUFFER SHARED
run_one BUFFER SEPARATE
run_one MEMORY SHARED
run_one MEMORY SEPARATE

if [[ "${INCLUDE_NONE:-0}" == "1" ]]; then
    run_one NONE SEPARATE
fi
