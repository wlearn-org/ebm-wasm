import { loadEBM, getWasm } from './wasm.js'
import {
  normalizeY,
  encodeBundle, decodeBundle,
  register,
  DisposedError, NotFittedError
} from '@wlearn/core'

// FinalizationRegistry safety net -- warns if dispose() was never called
const leakRegistry = typeof FinalizationRegistry !== 'undefined'
  ? new FinalizationRegistry(({ ref, freeFn }) => {
    if (ref[0]) {
      console.warn('@wlearn/ebm: EBMModel was not disposed -- calling free() automatically. This is a bug in your code.')
      freeFn(ref[0])
    }
  })
  : null

// Internal sentinel for load path
const LOAD_SENTINEL = Symbol('load')

// --- Helper: get last error from C ---
function getLastError() {
  return getWasm().ccall('wl_ebm_get_last_error', 'string', [], [])
}

// --- Helper: C string allocation ---
function withCString(wasm, str, fn) {
  const bytes = new TextEncoder().encode(str + '\0')
  const ptr = wasm._malloc(bytes.length)
  wasm.HEAPU8.set(bytes, ptr)
  try {
    return fn(ptr)
  } finally {
    wasm._free(ptr)
  }
}

// --- EBMModel ---

export class EBMModel {
  #handle = null      // WlEbmModel* for predict
  #freed = false
  #handleRef = null
  #params = {}
  #fitted = false
  #nClasses = 0
  #classes = null
  #nTerms = 0
  #nScores = 0
  #modelData = null   // JS-side model state
  #termNames = null
  #featureNames = null
  #isRegressor = false

  constructor(sentinel, arg1, arg2) {
    if (sentinel === LOAD_SENTINEL) {
      // Load path: arg1 = modelData, arg2 = { params, nClasses, classes, ... }
      this.#modelData = arg1
      this.#params = arg2.params || {}
      this.#nClasses = arg2.nClasses || 0
      this.#classes = arg2.classes ? new Int32Array(arg2.classes) : null
      this.#nTerms = arg1.terms.length
      this.#nScores = arg1.nScores
      this.#termNames = arg2.termNames || null
      this.#featureNames = arg2.featureNames || null
      this.#isRegressor = arg1.task === 'regression'
      this.#fitted = true
      this.#freed = false
      this.#handle = this.#buildCModel(arg1)
      this.#handleRef = [this.#handle]
      if (leakRegistry) {
        leakRegistry.register(this, {
          ref: this.#handleRef,
          freeFn: (h) => { try { getWasm()._wl_ebm_free_model(h) } catch {} }
        }, this)
      }
    } else {
      // Normal construction from create()
      this.#params = sentinel || {}
      this.#freed = false
    }
  }

  static async create(params = {}) {
    await loadEBM()
    return new EBMModel(params)
  }

  // --- Estimator interface ---

  fit(X, y) {
    this.#ensureFitted(false)

    // Dispose previous model if refitting
    if (this.#handle) {
      getWasm()._wl_ebm_free_model(this.#handle)
      this.#handle = null
      if (this.#handleRef) this.#handleRef[0] = null
      if (leakRegistry) leakRegistry.unregister(this)
    }
    this.#modelData = null

    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)
    const yNorm = normalizeY(y)
    if (yNorm.length !== rows) {
      throw new Error(`y length (${yNorm.length}) does not match X rows (${rows})`)
    }

    // Determine task
    const isRegressor = this.#params.objective === 'regression' || this.#detectRegression(yNorm)
    this.#isRegressor = isRegressor

    let nClasses = 0
    let classes = null
    let yInt = null

    if (!isRegressor) {
      const unique = new Set()
      for (let i = 0; i < yNorm.length; i++) {
        const v = yNorm[i]
        if (v !== Math.floor(v)) throw new Error(`Classifier labels must be integers, got ${v} at index ${i}`)
        unique.add(v)
      }
      classes = [...unique].sort((a, b) => a - b)
      nClasses = classes.length
      // Remap to 0-based contiguous
      const classMap = new Map()
      classes.forEach((c, i) => classMap.set(c, i))
      yInt = new Int32Array(rows)
      for (let i = 0; i < rows; i++) yInt[i] = classMap.get(yNorm[i])
      this.#classes = new Int32Array(classes)
      this.#nClasses = nClasses
    } else {
      this.#classes = null
      this.#nClasses = 0
    }

    // nScores: binary=1, multiclass=nClasses, regression=1
    const nScores = (!isRegressor && nClasses > 2) ? nClasses : 1
    this.#nScores = nScores

    // Parameters
    const maxBins = this.#params.maxBins || 256
    const minSamplesBin = this.#params.minSamplesBin || 1
    const maxRounds = this.#params.maxRounds || 5000
    const earlyStoppingRounds = this.#params.earlyStoppingRounds || 50
    const learningRate = this.#params.learningRate || 0.01
    const maxLeaves = this.#params.maxLeaves || 3
    const minSamplesLeaf = this.#params.minSamplesLeaf || 2
    const maxInteractions = this.#params.maxInteractions || 10
    const outerBags = this.#params.outerBags || 8
    const innerBags = this.#params.innerBags || 0
    const regAlpha = this.#params.regAlpha || 0
    const regLambda = this.#params.regLambda || 0
    const seed = this.#params.seed ?? 42

    // Step 1: Bin each feature
    const featureCuts = []
    const featureBinIndexes = []
    const featureTypes = new Array(cols).fill(0) // all continuous for now

    for (let f = 0; f < cols; f++) {
      // Extract feature column
      const colData = new Float64Array(rows)
      for (let r = 0; r < rows; r++) colData[r] = xData[r * cols + f]

      const colPtr = wasm._malloc(rows * 8)
      wasm.HEAPF64.set(colData, colPtr / 8)

      // Get suggested cut count
      let suggestedCuts = wasm._wl_ebm_get_histogram_cut_count(rows, colPtr)
      if (suggestedCuts > maxBins - 1) suggestedCuts = maxBins - 1
      if (suggestedCuts < 1) suggestedCuts = 1

      // Allocate cuts output
      const cutsPtr = wasm._malloc(suggestedCuts * 8)
      const countCutsPtr = wasm._malloc(4)
      wasm.HEAP32[countCutsPtr / 4] = suggestedCuts

      const err = wasm._wl_ebm_cut_quantile(rows, colPtr, minSamplesBin, 1, countCutsPtr, cutsPtr)
      const actualCuts = wasm.HEAP32[countCutsPtr / 4]

      const cuts = new Float64Array(actualCuts)
      for (let i = 0; i < actualCuts; i++) cuts[i] = wasm.HEAPF64[cutsPtr / 8 + i]

      featureCuts.push(cuts)

      // Discretize
      const binIdxPtr = wasm._malloc(rows * 4)
      wasm._wl_ebm_discretize(rows, colPtr, actualCuts, cutsPtr, binIdxPtr)

      const binIdx = new Int32Array(rows)
      for (let i = 0; i < rows; i++) binIdx[i] = wasm.HEAP32[binIdxPtr / 4 + i]
      featureBinIndexes.push(binIdx)

      wasm._free(colPtr)
      wasm._free(cutsPtr)
      wasm._free(countCutsPtr)
      wasm._free(binIdxPtr)
    }

    // Step 2: Build dataset
    const nBinsPerFeature = featureCuts.map(c => c.length + 2) // +1 missing +1 unseen

    let totalSize = wasm._wl_ebm_measure_dataset_header(cols, 0, 1)

    for (let f = 0; f < cols; f++) {
      const binIdxPtr = wasm._malloc(rows * 4)
      const arr = featureBinIndexes[f]
      for (let i = 0; i < rows; i++) wasm.HEAP32[binIdxPtr / 4 + i] = arr[i]
      totalSize += wasm._wl_ebm_measure_feature(nBinsPerFeature[f], 1, 1, 0, rows, binIdxPtr)
      wasm._free(binIdxPtr)
    }

    if (isRegressor) {
      const yPtr = wasm._malloc(rows * 8)
      wasm.HEAPF64.set(new Float64Array(yNorm), yPtr / 8)
      totalSize += wasm._wl_ebm_measure_regression_target(rows, yPtr)
      wasm._free(yPtr)
    } else {
      const yPtr = wasm._malloc(rows * 4)
      for (let i = 0; i < rows; i++) wasm.HEAP32[yPtr / 4 + i] = yInt[i]
      totalSize += wasm._wl_ebm_measure_classification_target(nClasses, rows, yPtr)
      wasm._free(yPtr)
    }

    // Allocate and fill dataset
    const dsPtr = wasm._malloc(totalSize)
    wasm._wl_ebm_fill_dataset_header(cols, 0, 1, totalSize, dsPtr)

    for (let f = 0; f < cols; f++) {
      const binIdxPtr = wasm._malloc(rows * 4)
      const arr = featureBinIndexes[f]
      for (let i = 0; i < rows; i++) wasm.HEAP32[binIdxPtr / 4 + i] = arr[i]
      wasm._wl_ebm_fill_feature(nBinsPerFeature[f], 1, 1, 0, rows, binIdxPtr, totalSize, dsPtr)
      wasm._free(binIdxPtr)
    }

    if (isRegressor) {
      const yPtr = wasm._malloc(rows * 8)
      wasm.HEAPF64.set(new Float64Array(yNorm), yPtr / 8)
      wasm._wl_ebm_fill_regression_target(rows, yPtr, totalSize, dsPtr)
      wasm._free(yPtr)
    } else {
      const yPtr = wasm._malloc(rows * 4)
      for (let i = 0; i < rows; i++) wasm.HEAP32[yPtr / 4 + i] = yInt[i]
      wasm._wl_ebm_fill_classification_target(nClasses, rows, yPtr, totalSize, dsPtr)
      wasm._free(yPtr)
    }

    // Step 3: Create train/validation bag
    const rngSize = wasm._wl_ebm_measure_rng()
    const rngPtr = wasm._malloc(rngSize)
    wasm._wl_ebm_init_rng(seed, rngPtr)

    const nTrain = Math.floor(rows * 0.85)
    const nVal = rows - nTrain
    const bagPtr = wasm._malloc(rows)
    wasm._wl_ebm_sample_without_replacement(rngPtr, nTrain, nVal, bagPtr)

    // Step 4: Detect interactions
    let interactionPairs = []
    if (maxInteractions > 0 && cols > 1) {
      const intHandlePtr = wasm._malloc(4)
      const objective = isRegressor ? 'rmse' : 'log_loss'
      const intErr = withCString(wasm, objective, (objPtr) => {
        return wasm._wl_ebm_create_interaction_detector(dsPtr, bagPtr, 0, 0, objPtr, intHandlePtr)
      })

      if (intErr === 0) {
        const intHandle = wasm.getValue(intHandlePtr, 'i32')
        const strengthPtr = wasm._malloc(8)
        const pairIdxPtr = wasm._malloc(8) // 2 * int32

        const candidates = []
        for (let i = 0; i < cols; i++) {
          for (let j = i + 1; j < cols; j++) {
            wasm.HEAP32[pairIdxPtr / 4] = i
            wasm.HEAP32[pairIdxPtr / 4 + 1] = j
            const calcErr = wasm._wl_ebm_calc_interaction_strength(
              intHandle, 2, pairIdxPtr, 0, 0, 0, strengthPtr
            )
            if (calcErr === 0) {
              candidates.push({ features: [i, j], strength: wasm.HEAPF64[strengthPtr / 8] })
            }
          }
        }

        candidates.sort((a, b) => b.strength - a.strength)
        interactionPairs = candidates.slice(0, maxInteractions)

        wasm._free(strengthPtr)
        wasm._free(pairIdxPtr)
        wasm._wl_ebm_free_interaction_detector(intHandle)
      }
      wasm._free(intHandlePtr)
    }

    // Step 5: Build term definitions
    // All single features + selected interaction pairs
    const termDefs = []
    for (let f = 0; f < cols; f++) {
      termDefs.push({ features: [f], nDims: 1 })
    }
    for (const pair of interactionPairs) {
      termDefs.push({ features: pair.features, nDims: 2 })
    }

    const nTerms = termDefs.length
    this.#nTerms = nTerms

    // Flatten for C: dimensionCounts and featureIndexes
    const dimCountsArr = new Int32Array(nTerms)
    const fiArr = []
    for (let t = 0; t < nTerms; t++) {
      dimCountsArr[t] = termDefs[t].nDims
      for (const fi of termDefs[t].features) fiArr.push(fi)
    }
    const featureIndexesArr = new Int32Array(fiArr)

    const dimCountsPtr = wasm._malloc(nTerms * 4)
    for (let i = 0; i < nTerms; i++) wasm.HEAP32[dimCountsPtr / 4 + i] = dimCountsArr[i]

    const fiPtr = wasm._malloc(featureIndexesArr.length * 4)
    for (let i = 0; i < featureIndexesArr.length; i++) wasm.HEAP32[fiPtr / 4 + i] = featureIndexesArr[i]

    // Step 6: Create booster
    const boosterHandlePtr = wasm._malloc(4)
    const objective = isRegressor ? 'rmse' : 'log_loss'

    const createErr = withCString(wasm, objective, (objPtr) => {
      return wasm._wl_ebm_create_booster(
        rngPtr, dsPtr, 0, bagPtr, 0,
        nTerms, dimCountsPtr, fiPtr,
        innerBags, 0, 0,
        objPtr, boosterHandlePtr
      )
    })

    wasm._free(dimCountsPtr)
    wasm._free(fiPtr)

    if (createErr !== 0) {
      wasm._free(bagPtr)
      wasm._free(rngPtr)
      wasm._free(dsPtr)
      throw new Error(`CreateBooster failed: error ${createErr}, ${getLastError()}`)
    }

    const boosterHandle = wasm.getValue(boosterHandlePtr, 'i32')
    wasm._free(boosterHandlePtr)

    // Step 7: Boosting loop
    const gainPtr = wasm._malloc(8)
    const metricPtr = wasm._malloc(8)
    let bestMetric = Infinity
    let roundsSinceImprovement = 0

    for (let round = 0; round < maxRounds; round++) {
      for (let t = 0; t < nTerms; t++) {
        wasm._wl_ebm_generate_term_update(
          rngPtr, boosterHandle, t,
          0, learningRate, minSamplesLeaf, 0.0,
          regAlpha, regLambda, 0.0,
          maxLeaves, gainPtr
        )
        wasm._wl_ebm_apply_term_update(boosterHandle, metricPtr)
      }

      const metric = wasm.HEAPF64[metricPtr / 8]
      if (metric < bestMetric - 1e-10) {
        bestMetric = metric
        roundsSinceImprovement = 0
      } else {
        roundsSinceImprovement++
      }
      if (roundsSinceImprovement >= earlyStoppingRounds) break
    }

    wasm._free(gainPtr)
    wasm._free(metricPtr)

    // Step 8: Extract model scores
    const modelData = {
      format: 'ebm-json-v1',
      task: isRegressor ? 'regression' : 'classification',
      nFeatures: cols,
      nTerms,
      nScores,
      intercept: new Array(nScores).fill(0),
      features: [],
      terms: []
    }

    for (let f = 0; f < cols; f++) {
      modelData.features.push({
        type: 'continuous',
        cuts: Array.from(featureCuts[f])
      })
    }

    for (let t = 0; t < nTerms; t++) {
      const td = termDefs[t]
      // Calculate bin counts per dimension for this term
      const binCounts = td.features.map(fi => nBinsPerFeature[fi])
      const flatSize = binCounts.reduce((a, b) => a * b, 1)

      const scoresPtr = wasm._malloc(flatSize * nScores * 8)
      wasm._wl_ebm_get_best_term_scores(boosterHandle, t, scoresPtr)

      const scores = new Array(flatSize * nScores)
      for (let i = 0; i < flatSize * nScores; i++) {
        scores[i] = wasm.HEAPF64[scoresPtr / 8 + i]
      }
      wasm._free(scoresPtr)

      modelData.terms.push({
        features: td.features,
        binCounts,
        scores
      })
    }

    // Cleanup booster and dataset
    wasm._wl_ebm_free_booster(boosterHandle)
    wasm._free(bagPtr)
    wasm._free(rngPtr)
    wasm._free(dsPtr)

    // Build term names
    this.#termNames = termDefs.map(td => {
      if (td.features.length === 1) return `feature_${td.features[0]}`
      return td.features.map(f => `feature_${f}`).join(' x ')
    })
    this.#featureNames = Array.from({ length: cols }, (_, i) => `feature_${i}`)

    // Store model data and build C prediction handle
    this.#modelData = modelData
    this.#handle = this.#buildCModel(modelData)
    this.#fitted = true
    this.#handleRef = [this.#handle]
    if (leakRegistry) {
      leakRegistry.register(this, {
        ref: this.#handleRef,
        freeFn: (h) => { try { getWasm()._wl_ebm_free_model(h) } catch {} }
      }, this)
    }

    return this
  }

  predict(X) {
    this.#ensureFitted()
    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)
    const outPtr = wasm._malloc(rows * 8)

    const ret = wasm._wl_ebm_predict_classes(this.#handle, xPtr, rows, cols, outPtr)
    if (ret !== 0) {
      wasm._free(xPtr)
      wasm._free(outPtr)
      throw new Error(`Predict failed: ${getLastError()}`)
    }

    const result = new Float64Array(rows)
    for (let i = 0; i < rows; i++) result[i] = wasm.HEAPF64[outPtr / 8 + i]

    // Remap back to original class labels if classification
    if (!this.#isRegressor && this.#classes) {
      for (let i = 0; i < rows; i++) {
        const idx = Math.round(result[i])
        if (idx >= 0 && idx < this.#classes.length) {
          result[i] = this.#classes[idx]
        }
      }
    }

    wasm._free(xPtr)
    wasm._free(outPtr)
    return result
  }

  predictProba(X) {
    this.#ensureFitted()
    if (this.#isRegressor) {
      throw new Error('predictProba is not available for regression')
    }

    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)
    const ns = this.#nScores

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)
    const scoresPtr = wasm._malloc(rows * ns * 8)

    const ret = wasm._wl_ebm_predict_scores(this.#handle, xPtr, rows, cols, scoresPtr)
    if (ret !== 0) {
      wasm._free(xPtr)
      wasm._free(scoresPtr)
      throw new Error(`PredictProba failed: ${getLastError()}`)
    }

    let result
    if (ns === 1) {
      // Binary: expand to [P(0), P(1)] per sample
      result = new Float64Array(rows * 2)
      for (let r = 0; r < rows; r++) {
        const logit = wasm.HEAPF64[scoresPtr / 8 + r]
        const p1 = 1.0 / (1.0 + Math.exp(-logit))
        result[r * 2] = 1.0 - p1
        result[r * 2 + 1] = p1
      }
    } else {
      // Multiclass: softmax
      result = new Float64Array(rows * ns)
      for (let r = 0; r < rows; r++) {
        let maxScore = -Infinity
        for (let c = 0; c < ns; c++) {
          const s = wasm.HEAPF64[scoresPtr / 8 + r * ns + c]
          if (s > maxScore) maxScore = s
        }
        let sumExp = 0
        for (let c = 0; c < ns; c++) {
          const e = Math.exp(wasm.HEAPF64[scoresPtr / 8 + r * ns + c] - maxScore)
          result[r * ns + c] = e
          sumExp += e
        }
        for (let c = 0; c < ns; c++) {
          result[r * ns + c] /= sumExp
        }
      }
    }

    wasm._free(xPtr)
    wasm._free(scoresPtr)
    return result
  }

  score(X, y) {
    const preds = this.predict(X)
    const yArr = normalizeY(y)

    if (this.#isRegressor) {
      // R-squared
      let ssRes = 0, ssTot = 0, yMean = 0
      for (let i = 0; i < yArr.length; i++) yMean += yArr[i]
      yMean /= yArr.length
      for (let i = 0; i < yArr.length; i++) {
        ssRes += (yArr[i] - preds[i]) ** 2
        ssTot += (yArr[i] - yMean) ** 2
      }
      return ssTot === 0 ? 0 : 1 - ssRes / ssTot
    }

    // Accuracy
    let correct = 0
    for (let i = 0; i < preds.length; i++) {
      if (preds[i] === yArr[i]) correct++
    }
    return correct / preds.length
  }

  // --- Explainability ---

  explain(X) {
    this.#ensureFitted()
    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)
    const ns = this.#nScores
    const nt = this.#nTerms

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)
    const outPtr = wasm._malloc(rows * nt * ns * 8)

    const ret = wasm._wl_ebm_explain_local(this.#handle, xPtr, rows, cols, outPtr)
    if (ret !== 0) {
      wasm._free(xPtr)
      wasm._free(outPtr)
      throw new Error(`Explain failed: ${getLastError()}`)
    }

    const contributions = new Float64Array(rows * nt * ns)
    for (let i = 0; i < rows * nt * ns; i++) {
      contributions[i] = wasm.HEAPF64[outPtr / 8 + i]
    }

    wasm._free(xPtr)
    wasm._free(outPtr)

    return {
      intercept: Array.from(this.#modelData.intercept),
      contributions,
      termNames: this.#termNames ? [...this.#termNames] : [],
      nTerms: nt,
      nSamples: rows,
      nScores: ns
    }
  }

  featureImportances() {
    this.#ensureFitted()
    const md = this.#modelData
    const importances = new Float64Array(md.nTerms)

    for (let t = 0; t < md.nTerms; t++) {
      const scores = md.terms[t].scores
      let sum = 0
      for (let i = 0; i < scores.length; i++) {
        sum += Math.abs(scores[i])
      }
      importances[t] = sum / scores.length
    }

    return importances
  }

  getShapeFunction(termIndex) {
    this.#ensureFitted()
    const md = this.#modelData
    if (termIndex < 0 || termIndex >= md.nTerms) {
      throw new Error(`termIndex ${termIndex} out of range [0, ${md.nTerms})`)
    }

    const term = md.terms[termIndex]
    const ns = md.nScores

    if (term.features.length === 1) {
      const fi = term.features[0]
      const cuts = md.features[fi].cuts
      // Bin centers: left edge of bin (for plotting)
      const nBins = term.binCounts[0]
      const x = new Float64Array(nBins)
      // bin 0: below first cut, bin 1..nCuts: between cuts, bin nCuts+1: above last cut
      // plus missing/unseen bins at the end
      for (let i = 0; i < nBins; i++) {
        if (i === 0 && cuts.length > 0) x[i] = cuts[0] - 1
        else if (i <= cuts.length) x[i] = cuts[i - 1]
        else x[i] = cuts[cuts.length - 1] + 1
      }

      if (ns === 1) {
        const y = new Float64Array(nBins)
        for (let i = 0; i < nBins; i++) y[i] = term.scores[i]
        return { x, y }
      } else {
        const y = new Float64Array(nBins * ns)
        for (let i = 0; i < nBins * ns; i++) y[i] = term.scores[i]
        return { x, y, nScores: ns }
      }
    }

    // Interaction: 2+ dimensions
    return {
      features: [...term.features],
      binCounts: [...term.binCounts],
      scores: new Float64Array(term.scores),
      nScores: ns
    }
  }

  // --- Model I/O ---

  save() {
    this.#ensureFitted()
    const jsonStr = JSON.stringify(this.#modelData)
    const jsonBytes = new TextEncoder().encode(jsonStr)
    const typeId = this.#isRegressor
      ? 'wlearn.ebm.regressor@1'
      : 'wlearn.ebm.classifier@1'
    return encodeBundle(
      {
        typeId,
        params: this.getParams(),
        metadata: {
          nClasses: this.#nClasses,
          classes: this.#classes ? Array.from(this.#classes) : [],
          termNames: this.#termNames,
          featureNames: this.#featureNames
        }
      },
      [{ id: 'model', data: jsonBytes }]
    )
  }

  static async load(bytes) {
    const { manifest, toc, blobs } = decodeBundle(bytes)
    return EBMModel._fromBundle(manifest, toc, blobs)
  }

  static async _fromBundle(manifest, toc, blobs) {
    await loadEBM()

    const entry = toc.find(e => e.id === 'model')
    if (!entry) throw new Error('Bundle missing "model" artifact')
    const raw = blobs.subarray(entry.offset, entry.offset + entry.length)
    const jsonStr = new TextDecoder().decode(raw)
    const modelData = JSON.parse(jsonStr)

    const meta = manifest.metadata || {}
    return new EBMModel(LOAD_SENTINEL, modelData, {
      params: manifest.params || {},
      nClasses: meta.nClasses || 0,
      classes: meta.classes || null,
      termNames: meta.termNames || null,
      featureNames: meta.featureNames || null
    })
  }

  dispose() {
    if (this.#freed) return
    this.#freed = true

    if (this.#handle) {
      try { getWasm()._wl_ebm_free_model(this.#handle) } catch {}
    }

    if (this.#handleRef) this.#handleRef[0] = null
    if (leakRegistry) leakRegistry.unregister(this)

    this.#handle = null
    this.#fitted = false
    this.#modelData = null
  }

  // --- Params ---

  getParams() {
    return { ...this.#params }
  }

  setParams(p) {
    Object.assign(this.#params, p)
    return this
  }

  static defaultSearchSpace() {
    return {
      learningRate: { type: 'log_uniform', low: 0.001, high: 0.1 },
      maxRounds: { type: 'int_uniform', low: 1000, high: 10000 },
      maxLeaves: { type: 'int_uniform', low: 2, high: 5 },
      maxInteractions: { type: 'int_uniform', low: 0, high: 20 },
      maxBins: { type: 'categorical', values: [128, 256, 512] },
      outerBags: { type: 'int_uniform', low: 4, high: 16 },
      minSamplesLeaf: { type: 'int_uniform', low: 1, high: 10 }
    }
  }

  // --- Inspection ---

  get nrClass() { return this.#nClasses }
  get classes() { return this.#classes ? Int32Array.from(this.#classes) : new Int32Array(0) }
  get nTerms() { return this.#nTerms }
  get termNames() { return this.#termNames ? [...this.#termNames] : [] }
  get featureNames() { return this.#featureNames ? [...this.#featureNames] : [] }
  get isFitted() { return this.#fitted && !this.#freed }

  get capabilities() {
    return {
      classifier: !this.#isRegressor,
      regressor: this.#isRegressor,
      predictProba: !this.#isRegressor,
      decisionFunction: false,
      sampleWeight: false,
      csr: false,
      earlyStopping: true,
      explain: true,
      featureImportances: true
    }
  }

  // Internal: expose handle for tests
  _getHandle() { return this.#handle }

  get probaDim() {
    if (!this.isFitted) return 0
    if (this.#isRegressor) return 0
    return this.#nClasses <= 2 ? 2 : this.#nClasses
  }

  // --- Private helpers ---

  #normalizeX(X) {
    // Fast path: typed matrix { data, rows, cols }
    if (X && typeof X === 'object' && !Array.isArray(X) && X.data) {
      const { data, rows, cols } = X
      if (data instanceof Float64Array) return { data, rows, cols }
      return { data: new Float64Array(data), rows, cols }
    }

    // Slow path: number[][]
    if (Array.isArray(X) && Array.isArray(X[0])) {
      const rows = X.length
      const cols = X[0].length
      const data = new Float64Array(rows * cols)
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          data[i * cols + j] = X[i][j]
        }
      }
      return { data, rows, cols }
    }

    throw new Error('X must be number[][] or { data: TypedArray, rows, cols }')
  }

  #ensureFitted(requireFit = true) {
    if (this.#freed) throw new DisposedError('EBMModel has been disposed.')
    if (requireFit && !this.#fitted) throw new NotFittedError('EBMModel is not fitted. Call fit() first.')
  }

  #detectRegression(y) {
    // If y contains non-integer values, assume regression
    for (let i = 0; i < y.length; i++) {
      if (y[i] !== Math.floor(y[i])) return true
    }
    return false
  }

  #buildCModel(modelData) {
    const wasm = getWasm()
    const task = modelData.task === 'regression' ? 1 : 0
    const nf = modelData.nFeatures
    const nt = modelData.nTerms
    const ns = modelData.nScores

    const handle = wasm._wl_ebm_alloc_model(task, nf, nt, ns)
    if (!handle) throw new Error('Failed to allocate C model')

    // Set intercept
    for (let s = 0; s < ns; s++) {
      wasm._wl_ebm_set_intercept(handle, s, modelData.intercept[s])
    }

    // Set features
    for (let f = 0; f < nf; f++) {
      const feat = modelData.features[f]
      const type = feat.type === 'nominal' ? 1 : 0
      const nCuts = feat.cuts ? feat.cuts.length : 0
      wasm._wl_ebm_set_feature(handle, f, type, nCuts)
      for (let c = 0; c < nCuts; c++) {
        wasm._wl_ebm_set_feature_edge(handle, f, c, feat.cuts[c])
      }
    }

    // Set terms
    for (let t = 0; t < nt; t++) {
      const term = modelData.terms[t]
      const nDims = term.features.length
      const flatSize = term.binCounts.reduce((a, b) => a * b, 1)

      wasm._wl_ebm_set_term(handle, t, nDims, flatSize)
      for (let d = 0; d < nDims; d++) {
        wasm._wl_ebm_set_term_feature(handle, t, d, term.features[d])
        wasm._wl_ebm_set_term_bin_count(handle, t, d, term.binCounts[d])
      }
      for (let i = 0; i < flatSize * ns; i++) {
        wasm._wl_ebm_set_term_score(handle, t, i, term.scores[i])
      }
    }

    return handle
  }
}

// --- Register loaders with @wlearn/core ---

register('wlearn.ebm.classifier@1', async (m, t, b) => EBMModel._fromBundle(m, t, b))
register('wlearn.ebm.regressor@1', async (m, t, b) => EBMModel._fromBundle(m, t, b))
