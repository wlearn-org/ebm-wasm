/*
 * wl_ebm_api.c -- C glue layer for libebm WASM port
 *
 * Provides:
 *   - WlEbmModel struct for prediction/explanation without libebm handles
 *   - Lookup-table scoring (predict_scores, predict_classes, explain_local)
 *   - JSON-based model deserialization via vendored cJSON
 *   - int32 wrappers for libebm functions that use int64 (IntEbm)
 *
 * libebm has no predict() or serialization API. Training uses the full
 * libebm C API from JS. Prediction is pure lookup-table evaluation in C.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "libebm.h"

/* ------------------------------------------------------------------ */
/* Error handling                                                      */
/* ------------------------------------------------------------------ */

static char g_last_error[512] = {0};

static void set_error(const char *msg) {
  strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
  g_last_error[sizeof(g_last_error) - 1] = '\0';
}

const char *wl_ebm_get_last_error(void) {
  return g_last_error;
}

/* ------------------------------------------------------------------ */
/* WlEbmModel -- in-memory model for prediction                        */
/* ------------------------------------------------------------------ */

typedef struct {
  int task;           /* 0=classification, 1=regression */
  int n_features;
  int n_terms;
  int n_scores;       /* 1 for binary/regression, n_classes for multiclass */
  double *intercept;  /* [n_scores] */

  /* per-term */
  int *term_n_dims;       /* [n_terms] */
  int **term_features;    /* [n_terms][n_dims] */
  int **term_bin_counts;  /* [n_terms][n_dims] */
  double **term_scores;   /* [n_terms][flat_size * n_scores] */
  int *term_flat_sizes;   /* [n_terms] -- product of bin_counts per term */

  /* per-feature */
  double **bin_edges;     /* [n_features][n_cuts] */
  int *n_cuts;            /* [n_features] */
  int *feature_types;     /* [n_features] 0=continuous, 1=nominal */
} WlEbmModel;

/* ------------------------------------------------------------------ */
/* Binning: find bin index for a continuous value                       */
/* ------------------------------------------------------------------ */

static int find_bin(const double *edges, int n_cuts, double val) {
  /* NaN -> last bin (missing bin) */
  if (val != val) return n_cuts; /* isnan without math.h dependency */

  /* Binary search: edges are sorted ascending */
  int lo = 0, hi = n_cuts;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (val < edges[mid]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

/* ------------------------------------------------------------------ */
/* Predict scores (raw, before link function)                          */
/* ------------------------------------------------------------------ */

int wl_ebm_predict_scores(const WlEbmModel *model, const double *X,
                           int nrow, int ncol, double *out) {
  if (!model || !X || !out) {
    set_error("null argument to predict_scores");
    return -1;
  }
  if (ncol != model->n_features) {
    set_error("feature count mismatch in predict_scores");
    return -1;
  }

  int ns = model->n_scores;

  for (int r = 0; r < nrow; r++) {
    const double *row = X + r * ncol;

    /* Start with intercept */
    for (int s = 0; s < ns; s++) {
      out[r * ns + s] = model->intercept[s];
    }

    /* Add contribution from each term */
    for (int t = 0; t < model->n_terms; t++) {
      int n_dims = model->term_n_dims[t];
      int flat_idx = 0;
      int stride = 1;

      /* Compute flat index from per-dimension bin indices (row-major) */
      for (int d = n_dims - 1; d >= 0; d--) {
        int fi = model->term_features[t][d];
        int bin;
        if (model->feature_types[fi] == 1) {
          /* Nominal: value IS the bin index */
          int v = (int)row[fi];
          bin = (v >= 0 && v < model->term_bin_counts[t][d]) ? v : model->term_bin_counts[t][d] - 1;
        } else {
          bin = find_bin(model->bin_edges[fi], model->n_cuts[fi], row[fi]);
        }
        /* Clamp to valid range (includes +1 for missing/unseen) */
        if (bin >= model->term_bin_counts[t][d]) {
          bin = model->term_bin_counts[t][d] - 1;
        }
        flat_idx += bin * stride;
        stride *= model->term_bin_counts[t][d];
      }

      for (int s = 0; s < ns; s++) {
        out[r * ns + s] += model->term_scores[t][flat_idx * ns + s];
      }
    }
  }
  return 0;
}

/* ------------------------------------------------------------------ */
/* Predict classes (apply link + argmax)                                */
/* ------------------------------------------------------------------ */

int wl_ebm_predict_classes(const WlEbmModel *model, const double *X,
                            int nrow, int ncol, double *out) {
  if (!model || !X || !out) {
    set_error("null argument to predict_classes");
    return -1;
  }

  int ns = model->n_scores;

  /* Allocate temp for raw scores */
  double *scores = (double *)malloc((size_t)nrow * ns * sizeof(double));
  if (!scores) {
    set_error("OOM in predict_classes");
    return -1;
  }

  int ret = wl_ebm_predict_scores(model, X, nrow, ncol, scores);
  if (ret != 0) {
    free(scores);
    return ret;
  }

  if (model->task == 1) {
    /* Regression: just copy scores */
    for (int r = 0; r < nrow; r++) {
      out[r] = scores[r];
    }
  } else if (ns == 1) {
    /* Binary classification: sigmoid + threshold */
    for (int r = 0; r < nrow; r++) {
      double p = 1.0 / (1.0 + exp(-scores[r]));
      out[r] = (p > 0.5) ? 1.0 : 0.0;
    }
  } else {
    /* Multiclass: argmax */
    for (int r = 0; r < nrow; r++) {
      int best = 0;
      for (int c = 1; c < ns; c++) {
        if (scores[r * ns + c] > scores[r * ns + best]) {
          best = c;
        }
      }
      out[r] = (double)best;
    }
  }

  free(scores);
  return 0;
}

/* ------------------------------------------------------------------ */
/* Explain local (per-term contributions)                              */
/* ------------------------------------------------------------------ */

int wl_ebm_explain_local(const WlEbmModel *model, const double *X,
                          int nrow, int ncol, double *out) {
  if (!model || !X || !out) {
    set_error("null argument to explain_local");
    return -1;
  }
  if (ncol != model->n_features) {
    set_error("feature count mismatch in explain_local");
    return -1;
  }

  int ns = model->n_scores;
  int nt = model->n_terms;

  for (int r = 0; r < nrow; r++) {
    const double *row = X + r * ncol;

    for (int t = 0; t < nt; t++) {
      int n_dims = model->term_n_dims[t];
      int flat_idx = 0;
      int stride = 1;

      for (int d = n_dims - 1; d >= 0; d--) {
        int fi = model->term_features[t][d];
        int bin;
        if (model->feature_types[fi] == 1) {
          int v = (int)row[fi];
          bin = (v >= 0 && v < model->term_bin_counts[t][d]) ? v : model->term_bin_counts[t][d] - 1;
        } else {
          bin = find_bin(model->bin_edges[fi], model->n_cuts[fi], row[fi]);
        }
        if (bin >= model->term_bin_counts[t][d]) {
          bin = model->term_bin_counts[t][d] - 1;
        }
        flat_idx += bin * stride;
        stride *= model->term_bin_counts[t][d];
      }

      for (int s = 0; s < ns; s++) {
        out[(r * nt + t) * ns + s] = model->term_scores[t][flat_idx * ns + s];
      }
    }
  }
  return 0;
}

/* ------------------------------------------------------------------ */
/* Model lifecycle: set_model / free_model                             */
/*                                                                     */
/* Model data is passed as flat arrays from JS. No JSON parsing in C.  */
/* JS builds the model struct by calling set_model_* functions.        */
/* ------------------------------------------------------------------ */

WlEbmModel *wl_ebm_alloc_model(int task, int n_features, int n_terms, int n_scores) {
  WlEbmModel *m = (WlEbmModel *)calloc(1, sizeof(WlEbmModel));
  if (!m) return NULL;

  m->task = task;
  m->n_features = n_features;
  m->n_terms = n_terms;
  m->n_scores = n_scores;

  m->intercept = (double *)calloc(n_scores, sizeof(double));
  m->term_n_dims = (int *)calloc(n_terms, sizeof(int));
  m->term_features = (int **)calloc(n_terms, sizeof(int *));
  m->term_bin_counts = (int **)calloc(n_terms, sizeof(int *));
  m->term_scores = (double **)calloc(n_terms, sizeof(double *));
  m->term_flat_sizes = (int *)calloc(n_terms, sizeof(int));
  m->bin_edges = (double **)calloc(n_features, sizeof(double *));
  m->n_cuts = (int *)calloc(n_features, sizeof(int));
  m->feature_types = (int *)calloc(n_features, sizeof(int));

  if (!m->intercept || !m->term_n_dims || !m->term_features ||
      !m->term_bin_counts || !m->term_scores || !m->term_flat_sizes ||
      !m->bin_edges || !m->n_cuts || !m->feature_types) {
    /* Partial alloc cleanup handled by free_model */
    return m;
  }
  return m;
}

void wl_ebm_set_intercept(WlEbmModel *m, int idx, double val) {
  if (m && m->intercept && idx >= 0 && idx < m->n_scores) {
    m->intercept[idx] = val;
  }
}

void wl_ebm_set_feature(WlEbmModel *m, int fi, int type, int n_cuts_val) {
  if (!m || fi < 0 || fi >= m->n_features) return;
  m->feature_types[fi] = type;
  m->n_cuts[fi] = n_cuts_val;
  if (n_cuts_val > 0) {
    m->bin_edges[fi] = (double *)calloc(n_cuts_val, sizeof(double));
  }
}

void wl_ebm_set_feature_edge(WlEbmModel *m, int fi, int ci, double val) {
  if (!m || fi < 0 || fi >= m->n_features) return;
  if (!m->bin_edges[fi] || ci < 0 || ci >= m->n_cuts[fi]) return;
  m->bin_edges[fi][ci] = val;
}

void wl_ebm_set_term(WlEbmModel *m, int ti, int n_dims, int flat_size) {
  if (!m || ti < 0 || ti >= m->n_terms) return;
  m->term_n_dims[ti] = n_dims;
  m->term_flat_sizes[ti] = flat_size;
  m->term_features[ti] = (int *)calloc(n_dims, sizeof(int));
  m->term_bin_counts[ti] = (int *)calloc(n_dims, sizeof(int));
  m->term_scores[ti] = (double *)calloc((size_t)flat_size * m->n_scores, sizeof(double));
}

void wl_ebm_set_term_feature(WlEbmModel *m, int ti, int di, int feature_idx) {
  if (!m || ti < 0 || ti >= m->n_terms) return;
  if (!m->term_features[ti] || di < 0 || di >= m->term_n_dims[ti]) return;
  m->term_features[ti][di] = feature_idx;
}

void wl_ebm_set_term_bin_count(WlEbmModel *m, int ti, int di, int count) {
  if (!m || ti < 0 || ti >= m->n_terms) return;
  if (!m->term_bin_counts[ti] || di < 0 || di >= m->term_n_dims[ti]) return;
  m->term_bin_counts[ti][di] = count;
}

void wl_ebm_set_term_score(WlEbmModel *m, int ti, int idx, double val) {
  if (!m || ti < 0 || ti >= m->n_terms) return;
  if (!m->term_scores[ti] || idx < 0 || idx >= m->term_flat_sizes[ti] * m->n_scores) return;
  m->term_scores[ti][idx] = val;
}

void wl_ebm_free_model(WlEbmModel *m) {
  if (!m) return;
  free(m->intercept);
  free(m->term_n_dims);
  free(m->term_flat_sizes);

  for (int i = 0; i < m->n_terms; i++) {
    free(m->term_features[i]);
    free(m->term_bin_counts[i]);
    free(m->term_scores[i]);
  }
  free(m->term_features);
  free(m->term_bin_counts);
  free(m->term_scores);

  for (int i = 0; i < m->n_features; i++) {
    free(m->bin_edges[i]);
  }
  free(m->bin_edges);
  free(m->n_cuts);
  free(m->feature_types);

  free(m);
}

/* ------------------------------------------------------------------ */
/* int32 wrappers for libebm int64 API                                 */
/*                                                                     */
/* Emscripten legalizes i64 to (lo32, hi32) pairs at the JS boundary.  */
/* These wrappers accept int32 params and cast to int64 internally.    */
/* ------------------------------------------------------------------ */

int32_t wl_ebm_measure_rng(void) {
  return (int32_t)MeasureRNG();
}

void wl_ebm_init_rng(int32_t seed, void *rngOut) {
  InitRNG((SeedEbm)seed, rngOut);
}

int32_t wl_ebm_generate_seed(void *rng, int32_t *seedOut) {
  SeedEbm seed;
  ErrorEbm err = GenerateSeed(rng, &seed);
  if (seedOut) *seedOut = (int32_t)seed;
  return (int32_t)err;
}

int32_t wl_ebm_get_histogram_cut_count(int32_t countSamples, const double *featureVals) {
  return (int32_t)GetHistogramCutCount((IntEbm)countSamples, featureVals);
}

int32_t wl_ebm_cut_quantile(int32_t countSamples, const double *featureVals,
                             int32_t minSamplesBin, int32_t isRounded,
                             int32_t *countCutsInOut, double *cutsOut) {
  IntEbm cuts = (IntEbm)*countCutsInOut;
  ErrorEbm err = CutQuantile((IntEbm)countSamples, featureVals,
                              (IntEbm)minSamplesBin, (BoolEbm)isRounded,
                              &cuts, cutsOut);
  *countCutsInOut = (int32_t)cuts;
  return (int32_t)err;
}

int32_t wl_ebm_discretize(int32_t countSamples, const double *featureVals,
                           int32_t countCuts, const double *cuts,
                           int32_t *binIndexesOut) {
  /* Discretize expects IntEbm* output, but we have int32_t* from JS.
     Allocate a temp IntEbm array and copy back. */
  IntEbm *tmp = (IntEbm *)malloc((size_t)countSamples * sizeof(IntEbm));
  if (!tmp) { set_error("OOM in discretize"); return -1; }

  ErrorEbm err = Discretize((IntEbm)countSamples, featureVals,
                             (IntEbm)countCuts, cuts, tmp);

  for (int32_t i = 0; i < countSamples; i++) {
    binIndexesOut[i] = (int32_t)tmp[i];
  }
  free(tmp);
  return (int32_t)err;
}

int32_t wl_ebm_measure_dataset_header(int32_t nFeatures, int32_t nWeights, int32_t nTargets) {
  return (int32_t)MeasureDataSetHeader((IntEbm)nFeatures, (IntEbm)nWeights, (IntEbm)nTargets);
}

int32_t wl_ebm_measure_feature(int32_t countBins, int32_t isMissing, int32_t isUnseen,
                                int32_t isNominal, int32_t countSamples,
                                const int32_t *binIndexes) {
  /* binIndexes is int32 from JS, need IntEbm */
  IntEbm *tmp = (IntEbm *)malloc((size_t)countSamples * sizeof(IntEbm));
  if (!tmp) return -1;
  for (int32_t i = 0; i < countSamples; i++) tmp[i] = (IntEbm)binIndexes[i];

  IntEbm result = MeasureFeature((IntEbm)countBins, (BoolEbm)isMissing,
                                  (BoolEbm)isUnseen, (BoolEbm)isNominal,
                                  (IntEbm)countSamples, tmp);
  free(tmp);
  return (int32_t)result;
}

int32_t wl_ebm_measure_classification_target(int32_t countClasses, int32_t countSamples,
                                              const int32_t *targets) {
  IntEbm *tmp = (IntEbm *)malloc((size_t)countSamples * sizeof(IntEbm));
  if (!tmp) return -1;
  for (int32_t i = 0; i < countSamples; i++) tmp[i] = (IntEbm)targets[i];

  IntEbm result = MeasureClassificationTarget((IntEbm)countClasses, (IntEbm)countSamples, tmp);
  free(tmp);
  return (int32_t)result;
}

int32_t wl_ebm_measure_regression_target(int32_t countSamples, const double *targets) {
  return (int32_t)MeasureRegressionTarget((IntEbm)countSamples, targets);
}

int32_t wl_ebm_fill_dataset_header(int32_t nFeatures, int32_t nWeights, int32_t nTargets,
                                    int32_t countBytesAllocated, void *fillMem) {
  return (int32_t)FillDataSetHeader((IntEbm)nFeatures, (IntEbm)nWeights,
                                     (IntEbm)nTargets, (IntEbm)countBytesAllocated, fillMem);
}

int32_t wl_ebm_fill_feature(int32_t countBins, int32_t isMissing, int32_t isUnseen,
                             int32_t isNominal, int32_t countSamples,
                             const int32_t *binIndexes,
                             int32_t countBytesAllocated, void *fillMem) {
  IntEbm *tmp = (IntEbm *)malloc((size_t)countSamples * sizeof(IntEbm));
  if (!tmp) return -1;
  for (int32_t i = 0; i < countSamples; i++) tmp[i] = (IntEbm)binIndexes[i];

  ErrorEbm err = FillFeature((IntEbm)countBins, (BoolEbm)isMissing,
                              (BoolEbm)isUnseen, (BoolEbm)isNominal,
                              (IntEbm)countSamples, tmp,
                              (IntEbm)countBytesAllocated, fillMem);
  free(tmp);
  return (int32_t)err;
}

int32_t wl_ebm_fill_classification_target(int32_t countClasses, int32_t countSamples,
                                           const int32_t *targets,
                                           int32_t countBytesAllocated, void *fillMem) {
  IntEbm *tmp = (IntEbm *)malloc((size_t)countSamples * sizeof(IntEbm));
  if (!tmp) return -1;
  for (int32_t i = 0; i < countSamples; i++) tmp[i] = (IntEbm)targets[i];

  ErrorEbm err = FillClassificationTarget((IntEbm)countClasses, (IntEbm)countSamples,
                                           tmp, (IntEbm)countBytesAllocated, fillMem);
  free(tmp);
  return (int32_t)err;
}

int32_t wl_ebm_fill_regression_target(int32_t countSamples, const double *targets,
                                       int32_t countBytesAllocated, void *fillMem) {
  return (int32_t)FillRegressionTarget((IntEbm)countSamples, targets,
                                        (IntEbm)countBytesAllocated, fillMem);
}

int32_t wl_ebm_sample_without_replacement(void *rng, int32_t nTrain, int32_t nVal,
                                           int8_t *bagOut) {
  return (int32_t)SampleWithoutReplacement(rng, (IntEbm)nTrain, (IntEbm)nVal, (BagEbm *)bagOut);
}

int32_t wl_ebm_create_booster(void *rng, const void *dataSet, const double *intercept,
                               const int8_t *bag, const double *initScores,
                               int32_t countTerms,
                               const int32_t *dimensionCounts,
                               const int32_t *featureIndexes,
                               int32_t countInnerBags,
                               int32_t flags, int32_t acceleration,
                               const char *objective,
                               void **boosterHandleOut) {
  /* Convert int32 arrays to IntEbm */
  IntEbm *dimCounts64 = (IntEbm *)malloc((size_t)countTerms * sizeof(IntEbm));
  if (!dimCounts64) { set_error("OOM"); return -1; }
  for (int32_t i = 0; i < countTerms; i++) dimCounts64[i] = (IntEbm)dimensionCounts[i];

  /* Count total feature indexes */
  int32_t totalFi = 0;
  for (int32_t i = 0; i < countTerms; i++) totalFi += dimensionCounts[i];

  IntEbm *fi64 = (IntEbm *)malloc((size_t)totalFi * sizeof(IntEbm));
  if (!fi64) { free(dimCounts64); set_error("OOM"); return -1; }
  for (int32_t i = 0; i < totalFi; i++) fi64[i] = (IntEbm)featureIndexes[i];

  BoosterHandle handle = NULL;
  ErrorEbm err = CreateBooster(rng, dataSet, intercept, (const BagEbm *)bag,
                                initScores, (IntEbm)countTerms, dimCounts64, fi64,
                                (IntEbm)countInnerBags, (CreateBoosterFlags)flags,
                                (AccelerationFlags)acceleration, objective, NULL, &handle);

  free(dimCounts64);
  free(fi64);
  *boosterHandleOut = handle;
  return (int32_t)err;
}

void wl_ebm_free_booster(void *handle) {
  FreeBooster((BoosterHandle)handle);
}

int32_t wl_ebm_generate_term_update(void *rng, void *handle, int32_t indexTerm,
                                     int32_t flags, double learningRate,
                                     int32_t minSamplesLeaf, double minHessian,
                                     double regAlpha, double regLambda,
                                     double maxDeltaStep,
                                     int32_t maxLeaves,
                                     double *avgGainOut) {
  IntEbm leavesMax = (IntEbm)maxLeaves;
  return (int32_t)GenerateTermUpdate(rng, (BoosterHandle)handle,
                                      (IntEbm)indexTerm, (TermBoostFlags)flags,
                                      learningRate, (IntEbm)minSamplesLeaf,
                                      minHessian, regAlpha, regLambda,
                                      maxDeltaStep,
                                      0, 0.0, 0, 0.0, /* categorical params */
                                      &leavesMax, NULL, avgGainOut);
}

int32_t wl_ebm_apply_term_update(void *handle, double *avgValMetricOut) {
  return (int32_t)ApplyTermUpdate((BoosterHandle)handle, avgValMetricOut);
}

int32_t wl_ebm_get_best_term_scores(void *handle, int32_t indexTerm, double *scoresOut) {
  return (int32_t)GetBestTermScores((BoosterHandle)handle, (IntEbm)indexTerm, scoresOut);
}

int32_t wl_ebm_get_current_term_scores(void *handle, int32_t indexTerm, double *scoresOut) {
  return (int32_t)GetCurrentTermScores((BoosterHandle)handle, (IntEbm)indexTerm, scoresOut);
}

int32_t wl_ebm_create_interaction_detector(const void *dataSet,
                                            const int8_t *bag,
                                            int32_t flags, int32_t acceleration,
                                            const char *objective,
                                            void **handleOut) {
  InteractionHandle handle = NULL;
  ErrorEbm err = CreateInteractionDetector(dataSet, NULL, (const BagEbm *)bag, NULL,
                                            (CreateInteractionFlags)flags,
                                            (AccelerationFlags)acceleration,
                                            objective, NULL, &handle);
  *handleOut = handle;
  return (int32_t)err;
}

void wl_ebm_free_interaction_detector(void *handle) {
  FreeInteractionDetector((InteractionHandle)handle);
}

int32_t wl_ebm_calc_interaction_strength(void *handle, int32_t countDimensions,
                                          const int32_t *featureIndexes,
                                          int32_t flags, int32_t maxCardinality,
                                          int32_t minSamplesLeaf,
                                          double *avgStrengthOut) {
  IntEbm *fi64 = (IntEbm *)malloc((size_t)countDimensions * sizeof(IntEbm));
  if (!fi64) { set_error("OOM"); return -1; }
  for (int32_t i = 0; i < countDimensions; i++) fi64[i] = (IntEbm)featureIndexes[i];

  ErrorEbm err = CalcInteractionStrength((InteractionHandle)handle,
                                          (IntEbm)countDimensions, fi64,
                                          (CalcInteractionFlags)flags,
                                          (IntEbm)maxCardinality,
                                          (IntEbm)minSamplesLeaf,
                                          0.0, 0.0, 0.0, 0.0,
                                          avgStrengthOut);
  free(fi64);
  return (int32_t)err;
}

int32_t wl_ebm_measure_weight(int32_t countSamples, const double *weights) {
  return (int32_t)MeasureWeight((IntEbm)countSamples, weights);
}

int32_t wl_ebm_fill_weight(int32_t countSamples, const double *weights,
                            int32_t countBytesAllocated, void *fillMem) {
  return (int32_t)FillWeight((IntEbm)countSamples, weights,
                              (IntEbm)countBytesAllocated, fillMem);
}
