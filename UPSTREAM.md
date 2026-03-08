# Upstream: InterpretML

## Source

- **Repository:** https://github.com/interpretml/interpret
- **License:** MIT
- **Component used:** `shared/libebm/` (C++ EBM core library)

## Tracked version

- **Tag:** v0.7.5
- **Submodule path:** `upstream/interpret/`

## Modifications for WASM

1. **Zone registration bypass:** Only CPU scalar zone (cpu_64) is compiled. AVX2, AVX512, and CUDA zones are excluded by not defining `BRIDGE_AVX2_32` or `BRIDGE_AVX512F_32`, which makes the CPUID detection and SIMD registration code compile out naturally.

2. **No threading:** Single-threaded WASM build. No `-pthread` flag.

3. **C++ exceptions enabled:** `-fexceptions` flag required because libebm uses C++ exceptions internally (CreateBooster throws on invalid inputs). Without this flag, Emscripten silently drops exceptions, causing cryptic failures.

4. **int64 boundary handling:** libebm uses `int64_t` (IntEbm) extensively. The C glue layer (`csrc/wl_ebm_api.c`) provides int32 wrappers for all functions called from JS, since Emscripten legalizes i64 to (lo32, hi32) pairs at the JS boundary.

5. **Prediction and serialization:** libebm has no predict() or serialization API. Training uses the full libebm C API (CreateBooster, GenerateTermUpdate, GetBestTermScores). The C glue layer implements lookup-table scoring (`wl_ebm_predict_scores`, `wl_ebm_predict_classes`, `wl_ebm_explain_local`) and a flat setter-based API for loading model data from JS (no JSON parsing in C).

6. **Model format:** JSON blob inside WLRN bundle. Contains bin edges, term scores, intercept, and feature metadata. Designed for cross-language parity (Python can produce and consume identical structures).

## Build output

- `wasm/ebm.js` -- ~728KB (WASM embedded via SINGLE_FILE=1)
- Flags: `-O2 -std=c++14 -fexceptions -DNDEBUG`
- Zones: cpu_64 (scalar double) only

## Update policy

- Track stable tags (not main branch)
- Apply patches via `patches/` directory if needed
- Verify golden test fixtures after each update
- Run both JS (27 tests) and Python (cross-language compat) tests
