#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GLUE="${1:-${PROJECT_DIR}/wasm/ebm.cjs}"

if [ ! -f "$GLUE" ]; then
  echo "ERROR: glue file not found: $GLUE"
  exit 1
fi

REQUIRED_SYMBOLS=(
  wl_ebm_get_last_error
  wl_ebm_predict_scores
  wl_ebm_predict_classes
  wl_ebm_explain_local
  wl_ebm_alloc_model
  wl_ebm_set_intercept
  wl_ebm_set_feature
  wl_ebm_set_feature_edge
  wl_ebm_set_term
  wl_ebm_set_term_feature
  wl_ebm_set_term_bin_count
  wl_ebm_set_term_score
  wl_ebm_free_model
  wl_ebm_measure_rng
  wl_ebm_init_rng
  wl_ebm_generate_seed
  wl_ebm_get_histogram_cut_count
  wl_ebm_cut_quantile
  wl_ebm_discretize
  wl_ebm_measure_dataset_header
  wl_ebm_measure_feature
  wl_ebm_measure_classification_target
  wl_ebm_measure_regression_target
  wl_ebm_fill_dataset_header
  wl_ebm_fill_feature
  wl_ebm_fill_classification_target
  wl_ebm_fill_regression_target
  wl_ebm_sample_without_replacement
  wl_ebm_create_booster
  wl_ebm_free_booster
  wl_ebm_generate_term_update
  wl_ebm_apply_term_update
  wl_ebm_get_best_term_scores
  wl_ebm_get_current_term_scores
  wl_ebm_create_interaction_detector
  wl_ebm_free_interaction_detector
  wl_ebm_calc_interaction_strength
)

missing=0
for fn in "${REQUIRED_SYMBOLS[@]}"; do
  if ! grep -q "_${fn}" "$GLUE"; then
    echo "MISSING: ${fn}"
    missing=$((missing + 1))
  fi
done

if [ $missing -gt 0 ]; then
  echo "ERROR: ${missing} symbol(s) missing"
  exit 1
fi

echo "All ${#REQUIRED_SYMBOLS[@]} exports verified"
