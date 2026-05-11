# vulkan-test

Standalone repro for a suspected MoltenVK / Metal flush+invalidate bug seen in
museld on AMD on macOS. Plan lives in
`../museld/vulkan-flush-invalidate-repro-plan.md`.

## Build

```
cmake -S . -B build
cmake --build build
```

Requires the LunarG Vulkan SDK and Homebrew's `vulkan-loader` / `glslang`.

## Run

The Homebrew vulkan-loader does not see the SDK's validation layer manifest by
default. Use the wrapper, which sets `VK_LAYER_PATH` to the SDK location:

```
./run-matrix.sh                       # all 4 variants, 10000 iters each
ITER=200 ./run-matrix.sh              # quick check
INCLUDE_NONE=1 ./run-matrix.sh        # also run the no-barrier variant
```

Or one variant at a time:

```
VK_LAYER_PATH=~/VulkanSDK/1.4.341.1/macOS/share/vulkan/explicit_layer.d \
  ./build/repro --barrier=BUFFER --layout=SHARED --iterations=10000
```

## The three programs

- `repro` (built from `main.cpp`, ~770 LOC) — full matrix harness with all the
  flags. Use this for sweeping variants.
- `compute_repro` (built from `compute_repro.cpp`, ~310 LOC) — hardcoded failing
  shape: 2 frames in flight, BUFFER barrier, SHARED `VkDeviceMemory`, threaded
  reader, compute shader writes B = A xor 0xA5A5A5A5. Validation opt-in via
  `REPRO_VALIDATION=1`.
- `minimal_repro` (built from `minimal_repro.cpp`, ~268 LOC) — smallest reproducer.
  No compute pipeline, no shader, no descriptors. Just `vkCmdCopyBuffer(A->B)`
  with two pipelined frames sharing one `VkDeviceMemory` and a reader thread
  doing `vkInvalidateMappedMemoryRanges`. Reproduces at ~29-35% per iteration.

`minimal_repro` is the file to attach to a MoltenVK bug report.

## Variant matrix

| Variant | --barrier | --layout |
|---|---|---|
| 1 | BUFFER | SHARED |
| 2 | BUFFER | SEPARATE |
| 3 | MEMORY | SHARED |
| 4 | MEMORY | SEPARATE |
| 5 | NONE   | (any) |

Variant 2 = museld's known-good shape. Variants 1 / 3 / 4 are suspected to
reproduce corruption on this hardware. See the plan for the decision tree.

## Notes

- Hard-fails at startup if no memory type with
  `DEVICE_LOCAL | HOST_VISIBLE | HOST_CACHED` (and not `HOST_COHERENT`) exists —
  that path is the whole point of the test.
- Returns exit code 1 if any iteration mismatched, 0 if clean.
