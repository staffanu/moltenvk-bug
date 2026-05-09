#!/usr/bin/env bash
# Run the variant matrix and print a compact summary.
set -euo pipefail
cd "$(dirname "$0")"

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
    local label="$1"; shift
    echo "============================================================"
    echo "VARIANT $label  iterations=$ITER"
    echo "  $BIN $* --iterations=$ITER"
    echo "============================================================"
    "$BIN" "$@" --iterations="$ITER" || true
    echo
}

# Original 4-variant matrix (single frame, single thread, no extras).
run_one "1: BUFFER + SHARED"   --barrier=BUFFER --layout=SHARED
run_one "2: BUFFER + SEPARATE" --barrier=BUFFER --layout=SEPARATE
run_one "3: MEMORY + SHARED"   --barrier=MEMORY --layout=SHARED
run_one "4: MEMORY + SEPARATE" --barrier=MEMORY --layout=SEPARATE

if [[ "${INCLUDE_NONE:-0}" == "1" ]]; then
    run_one "5: NONE + SEPARATE" --barrier=NONE --layout=SEPARATE
fi

# Extended matrix: pipelined + threaded reader, with and without neighbours.
if [[ "${EXTENDED:-1}" == "1" ]]; then
    run_one "6: pipelined frames=3"                         --barrier=BUFFER --layout=SHARED   --frames-in-flight=3
    run_one "7: pipelined frames=3 + threaded-reader"       --barrier=BUFFER --layout=SHARED   --frames-in-flight=3 --threaded-reader
    run_one "8: pipelined frames=3 + threaded + MEMORY"     --barrier=MEMORY --layout=SHARED   --frames-in-flight=3 --threaded-reader
    run_one "9: pipelined + threaded + neighbours=4"        --barrier=BUFFER --layout=SHARED   --frames-in-flight=3 --threaded-reader --neighbours=4
    run_one "10: cmd-split + pipelined + threaded"          --barrier=BUFFER --layout=SHARED   --frames-in-flight=3 --threaded-reader --cmd-split
    run_one "11: SEPARATE + pipelined + threaded"           --barrier=BUFFER --layout=SEPARATE --frames-in-flight=3 --threaded-reader
    run_one "12: kitchen sink (BUFFER)"                     --barrier=BUFFER --layout=SHARED   --frames-in-flight=3 --threaded-reader --neighbours=4 --cmd-split
    run_one "13: kitchen sink (MEMORY)"                     --barrier=MEMORY --layout=SHARED   --frames-in-flight=3 --threaded-reader --neighbours=4 --cmd-split
fi
