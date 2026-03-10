#!/bin/bash
set -euo pipefail

# Build InterpretML libebm as WASM via Emscripten
# Prerequisites: emsdk activated (emcc in PATH)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
UPSTREAM_DIR="${PROJECT_DIR}/upstream/interpret/shared/libebm"
OUTPUT_DIR="${PROJECT_DIR}/wasm"

# Verify prerequisites
if ! command -v emcc &> /dev/null; then
  echo "ERROR: emcc not found. Activate emsdk first:"
  echo "  source ~/tools/emsdk/emsdk_env.sh"
  exit 1
fi

if [ ! -f "$UPSTREAM_DIR/inc/libebm.h" ]; then
  echo "ERROR: libebm upstream not found at ${UPSTREAM_DIR}"
  echo "  git submodule update --init"
  exit 1
fi

echo "=== Applying patches ==="
if [ -d "${PROJECT_DIR}/patches" ] && ls "${PROJECT_DIR}/patches"/*.patch &> /dev/null 2>&1; then
  for patch in "${PROJECT_DIR}/patches"/*.patch; do
    echo "Applying: $(basename "$patch")"
    (cd "${PROJECT_DIR}/upstream/interpret" && git apply --check "$patch" 2>/dev/null && git apply "$patch") || \
      echo "  (already applied or not applicable)"
  done
else
  echo "  No patches found"
fi

echo "=== Collecting source files ==="
mkdir -p "$OUTPUT_DIR"

# Collect main libebm .cpp files (root level, not special/ or tests/)
MAIN_SOURCES=""
for f in "$UPSTREAM_DIR"/*.cpp; do
  [ -f "$f" ] && MAIN_SOURCES="$MAIN_SOURCES $f"
done

# Unzoned sources
UNZONED_SOURCES="$UPSTREAM_DIR/unzoned/logging.cpp $UPSTREAM_DIR/unzoned/unzoned.cpp"

# CPU zone (scalar, double precision -- the only zone for WASM)
CPU_ZONE_SOURCE="$UPSTREAM_DIR/compute/cpu_ebm/cpu_64.cpp"

# Compute accessors (zone dispatch -- no CPUID when BRIDGE_AVX* not defined)
COMPUTE_ACCESSORS="$UPSTREAM_DIR/compute_accessors.cpp"

# Our C glue layer
GLUE_SOURCE="${PROJECT_DIR}/csrc/wl_ebm_api.c"

echo "  Main sources: $(echo $MAIN_SOURCES | wc -w | tr -d ' ') files"
echo "  Unzoned: 2 files"
echo "  CPU zone: 1 file"
echo "  Compute accessors: 1 file"
echo "  Glue: 1 file"

echo "=== Compiling C glue (as C) ==="
BUILD_TMP="${PROJECT_DIR}/.build_tmp"
mkdir -p "$BUILD_TMP"

emcc \
  "$GLUE_SOURCE" \
  -c \
  -O2 \
  -DNDEBUG \
  -DLIBEBM_EXPORTS \
  -I "$UPSTREAM_DIR/inc" \
  -o "${BUILD_TMP}/wl_ebm_api.o"

echo "=== Compiling WASM ==="

# Exported functions from our C glue
EXPORTED_FUNCTIONS='[
  "_wl_ebm_get_last_error",
  "_wl_ebm_predict_scores",
  "_wl_ebm_predict_classes",
  "_wl_ebm_explain_local",
  "_wl_ebm_alloc_model",
  "_wl_ebm_set_intercept",
  "_wl_ebm_set_feature",
  "_wl_ebm_set_feature_edge",
  "_wl_ebm_set_term",
  "_wl_ebm_set_term_feature",
  "_wl_ebm_set_term_bin_count",
  "_wl_ebm_set_term_score",
  "_wl_ebm_free_model",
  "_wl_ebm_measure_rng",
  "_wl_ebm_init_rng",
  "_wl_ebm_generate_seed",
  "_wl_ebm_get_histogram_cut_count",
  "_wl_ebm_cut_quantile",
  "_wl_ebm_discretize",
  "_wl_ebm_measure_dataset_header",
  "_wl_ebm_measure_feature",
  "_wl_ebm_measure_classification_target",
  "_wl_ebm_measure_regression_target",
  "_wl_ebm_measure_weight",
  "_wl_ebm_fill_dataset_header",
  "_wl_ebm_fill_feature",
  "_wl_ebm_fill_classification_target",
  "_wl_ebm_fill_regression_target",
  "_wl_ebm_fill_weight",
  "_wl_ebm_sample_without_replacement",
  "_wl_ebm_create_booster",
  "_wl_ebm_free_booster",
  "_wl_ebm_generate_term_update",
  "_wl_ebm_apply_term_update",
  "_wl_ebm_get_best_term_scores",
  "_wl_ebm_get_current_term_scores",
  "_wl_ebm_create_interaction_detector",
  "_wl_ebm_free_interaction_detector",
  "_wl_ebm_calc_interaction_strength",
  "_malloc",
  "_free"
]'

# Remove newlines for emcc
EXPORTED_FUNCTIONS=$(echo "$EXPORTED_FUNCTIONS" | tr -d '\n' | tr -s ' ')

EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue","HEAPF64","HEAPU8","HEAP32","HEAP8","UTF8ToString"]'

# Include paths
INCLUDES="-I $UPSTREAM_DIR/inc -I $UPSTREAM_DIR -I $UPSTREAM_DIR/bridge -I $UPSTREAM_DIR/compute -I $UPSTREAM_DIR/compute/objectives -I $UPSTREAM_DIR/compute/metrics -I $UPSTREAM_DIR/unzoned"

# Compile all sources together
# Key: do NOT define BRIDGE_AVX2_32 or BRIDGE_AVX512F_32 (skips CPUID and SIMD zones)
# The CPU zone file needs -DZONE_cpu but that's defined inside cpu_64.cpp itself
# Main files use -DZONE_main which is defined inside each file via #define ZONE_main
em++ \
  "${BUILD_TMP}/wl_ebm_api.o" \
  $MAIN_SOURCES \
  $UNZONED_SOURCES \
  $CPU_ZONE_SOURCE \
  -std=c++14 \
  -O2 \
  -DNDEBUG \
  -DLIBEBM_EXPORTS \
  -fexceptions \
  -fno-math-errno -fno-trapping-math -ffp-contract=off \
  -Wno-c++11-narrowing \
  $INCLUDES \
  -o "${OUTPUT_DIR}/ebm.js" \
  -s MODULARIZE=1 \
  -s SINGLE_FILE=1 \
  -s SINGLE_FILE_BINARY_ENCODE=0 \
  -s EXPORT_NAME=createEBM \
  -s EXPORTED_FUNCTIONS="${EXPORTED_FUNCTIONS}" \
  -s EXPORTED_RUNTIME_METHODS="${EXPORTED_RUNTIME_METHODS}" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=33554432 \
  -s ENVIRONMENT='web,node' \
  -s FORCE_FILESYSTEM=0

echo "=== Verifying exports ==="
bash "${SCRIPT_DIR}/verify-exports.sh"

echo "=== Writing BUILD_INFO ==="
cat > "${OUTPUT_DIR}/BUILD_INFO" <<EOF
upstream: InterpretML libebm v0.7.5
upstream_commit: $(cd "${PROJECT_DIR}/upstream/interpret" && git rev-parse HEAD 2>/dev/null || echo "unknown")
build_date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
emscripten: $(emcc --version | head -1)
build_flags: -O2 -std=c++14 SINGLE_FILE=1 cpu_64 only
wasm_embedded: true
zones: cpu_64 (scalar double)
EOF

rm -rf "$BUILD_TMP"

echo "=== Build complete ==="
ls -lh "${OUTPUT_DIR}/ebm.js"
cat "${OUTPUT_DIR}/BUILD_INFO"
