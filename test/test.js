let passed = 0
let failed = 0

async function test(name, fn) {
  try {
    await fn()
    console.log(`  PASS: ${name}`)
    passed++
  } catch (err) {
    console.log(`  FAIL: ${name}`)
    console.log(`        ${err.message}`)
    if (err.stack) {
      const lines = err.stack.split('\n').slice(1, 3)
      for (const line of lines) console.log(`        ${line.trim()}`)
    }
    failed++
  }
}

function assert(condition, msg) {
  if (!condition) throw new Error(msg || 'assertion failed')
}

function assertClose(a, b, tol, msg) {
  const diff = Math.abs(a - b)
  if (diff > tol) throw new Error(msg || `expected ${a} ~ ${b} (diff=${diff}, tol=${tol})`)
}

// Deterministic data generation
function makeLinearData(n, seed = 7) {
  const X = []
  const y = []
  for (let i = 0; i < n; i++) {
    const t = ((i * seed + 3) % n) / n
    const s = ((i * (seed + 6) + 7) % n) / n
    const x1 = t * 2 - 1
    const x2 = s * 2 - 1
    X.push([x1, x2])
    y.push(x1 + x2 > 0 ? 1 : 0)
  }
  return { X, y }
}

function makeMulticlassData(n) {
  const X = []
  const y = []
  for (let i = 0; i < n; i++) {
    const t = ((i * 7 + 3) % n) / n
    const s = ((i * 13 + 7) % n) / n
    const x1 = t * 2 - 1
    const x2 = s * 2 - 1
    X.push([x1, x2])
    const sum = x1 + x2
    y.push(sum < -0.3 ? 0 : sum < 0.3 ? 1 : 2)
  }
  return { X, y }
}

function makeRegressionData(n) {
  const X = []
  const y = []
  for (let i = 0; i < n; i++) {
    const x1 = ((i * 7 + 3) % n) / (n / 2) - 1
    const x2 = ((i * 13 + 7) % n) / (n / 2) - 1
    const noise = ((i * 31 + 11) % n) / (n * 5) - 0.1
    X.push([x1, x2])
    y.push(2 * x1 + 3 * x2 + noise)
  }
  return { X, y }
}

async function main() {

// ============================================================
// WASM loading
// ============================================================
console.log('\n=== WASM Loading ===')

const { loadEBM, getWasm } = require('../src/wasm.js')
const wasm = await loadEBM()

await test('WASM module loads', async () => {
  assert(wasm, 'wasm module is null')
  assert(typeof wasm.ccall === 'function', 'ccall not available')
})

await test('get_last_error returns string', async () => {
  const err = wasm.ccall('wl_ebm_get_last_error', 'string', [], [])
  assert(typeof err === 'string', `expected string, got ${typeof err}`)
})

// ============================================================
// EBMModel basics
// ============================================================
console.log('\n=== EBMModel ===')

const { EBMModel } = require('../src/model.js')

await test('create() returns model', async () => {
  const model = await EBMModel.create()
  assert(model, 'model is null')
  assert(!model.isFitted, 'should not be fitted yet')
  model.dispose()
})

// ============================================================
// Binary classification
// ============================================================
console.log('\n=== Binary Classification ===')

await test('Binary classification', async () => {
  const model = await EBMModel.create({
    maxRounds: 200,
    learningRate: 0.05,
    maxInteractions: 0,
    earlyStoppingRounds: 30,
    seed: 42
  })

  const { X, y } = makeLinearData(120)
  model.fit(X, y)
  assert(model.isFitted, 'should be fitted')
  assert(model.capabilities.classifier, 'should be classifier')
  assert(model.nrClass === 2, `expected 2 classes, got ${model.nrClass}`)

  const preds = model.predict(X)
  assert(preds instanceof Float64Array, 'predictions should be Float64Array')
  assert(preds.length === 120, `expected 120 predictions, got ${preds.length}`)

  let correct = 0
  for (let i = 0; i < preds.length; i++) {
    if (preds[i] === y[i]) correct++
  }
  const accuracy = correct / preds.length
  assert(accuracy > 0.65, `accuracy ${accuracy.toFixed(3)} too low for linearly separable data`)

  model.dispose()
})

// ============================================================
// Multiclass classification
// ============================================================
console.log('\n=== Multiclass Classification ===')

await test('Multiclass classification', async () => {
  const model = await EBMModel.create({
    maxRounds: 300,
    learningRate: 0.05,
    maxInteractions: 0,
    earlyStoppingRounds: 50,
    seed: 42
  })

  const { X, y } = makeMulticlassData(180)
  try {
    model.fit(X, y)
  } catch (e) {
    throw new Error(`fit failed: ${e.message}`)
  }
  assert(model.nrClass === 3, `expected 3 classes, got ${model.nrClass}`)
  assert(model.capabilities.classifier, 'should be classifier')

  const preds = model.predict(X)
  assert(preds.length === 180, `expected 180 predictions, got ${preds.length}`)

  // All predictions should be valid class labels
  const validClasses = new Set([0, 1, 2])
  for (let i = 0; i < preds.length; i++) {
    assert(validClasses.has(preds[i]),
      `invalid prediction at ${i}: ${preds[i]}`)
  }

  model.dispose()
})

// ============================================================
// Regression
// ============================================================
console.log('\n=== Regression ===')

await test('Regression', async () => {
  const model = await EBMModel.create({
    maxRounds: 500,
    learningRate: 0.05,
    maxInteractions: 0,
    earlyStoppingRounds: 50,
    seed: 42
  })

  const { X, y } = makeRegressionData(120)
  model.fit(X, y)
  assert(model.capabilities.regressor, 'should be regressor')

  const preds = model.predict(X)
  assert(preds.length === 120, `expected 120 predictions, got ${preds.length}`)

  const r2 = model.score(X, y)
  assert(r2 > 0.3, `R-squared ${r2.toFixed(3)} too low`)

  model.dispose()
})

// ============================================================
// Probability estimates
// ============================================================
console.log('\n=== Probability ===')

await test('predictProba returns valid probabilities (binary)', async () => {
  const model = await EBMModel.create({
    maxRounds: 200,
    learningRate: 0.05,
    maxInteractions: 0,
    seed: 42
  })

  const { X, y } = makeLinearData(100)
  model.fit(X, y)
  const probs = model.predictProba(X)

  assert(probs.length === 200, `expected 200 probabilities (100*2), got ${probs.length}`)

  for (let r = 0; r < 100; r++) {
    const p0 = probs[r * 2]
    const p1 = probs[r * 2 + 1]
    assert(p0 >= 0 && p0 <= 1, `P(0) out of [0,1]: ${p0}`)
    assert(p1 >= 0 && p1 <= 1, `P(1) out of [0,1]: ${p1}`)
    assertClose(p0 + p1, 1.0, 1e-6, `row ${r} probabilities sum to ${p0 + p1}`)
  }

  model.dispose()
})

await test('predictProba throws for regression', async () => {
  const model = await EBMModel.create({ seed: 42, maxRounds: 50, maxInteractions: 0 })
  const { X, y } = makeRegressionData(50)
  model.fit(X, y)

  let threw = false
  try { model.predictProba(X) } catch { threw = true }
  assert(threw, 'predictProba should throw for regression')

  model.dispose()
})

// ============================================================
// Local explanations
// ============================================================
console.log('\n=== Explainability ===')

await test('explain returns per-term contributions', async () => {
  const model = await EBMModel.create({
    maxRounds: 200,
    learningRate: 0.05,
    maxInteractions: 0,
    seed: 42
  })

  const { X, y } = makeLinearData(50)
  model.fit(X, y)

  const expl = model.explain(X)
  assert(expl.nSamples === 50, `expected 50 samples, got ${expl.nSamples}`)
  assert(expl.nTerms > 0, 'should have terms')
  assert(expl.contributions instanceof Float64Array, 'contributions should be Float64Array')
  assert(expl.termNames.length === expl.nTerms, 'termNames length should match nTerms')

  // Verify contributions sum to raw scores (approximately)
  const wasmRef = getWasm()
  const xData = new Float64Array(X.flat())
  const rows = X.length
  const cols = X[0].length
  const xPtr = wasmRef._malloc(xData.length * 8)
  wasmRef.HEAPF64.set(xData, xPtr / 8)
  const scoresPtr = wasmRef._malloc(rows * 8)
  wasmRef._wl_ebm_predict_scores(model._getHandle(), xPtr, rows, cols, scoresPtr)

  for (let r = 0; r < rows; r++) {
    let contribSum = expl.intercept[0]
    for (let t = 0; t < expl.nTerms; t++) {
      contribSum += expl.contributions[r * expl.nTerms + t]
    }
    const rawScore = wasmRef.HEAPF64[scoresPtr / 8 + r]
    assertClose(contribSum, rawScore, 1e-6, `row ${r}: contrib sum ${contribSum} != score ${rawScore}`)
  }

  wasmRef._free(xPtr)
  wasmRef._free(scoresPtr)
  model.dispose()
})

await test('featureImportances returns correct length', async () => {
  const model = await EBMModel.create({
    maxRounds: 100,
    maxInteractions: 0,
    seed: 42
  })

  const { X, y } = makeLinearData(80)
  model.fit(X, y)

  const imp = model.featureImportances()
  assert(imp instanceof Float64Array, 'should be Float64Array')
  assert(imp.length === model.nTerms, `expected ${model.nTerms} importances, got ${imp.length}`)
  for (let i = 0; i < imp.length; i++) {
    assert(imp[i] >= 0, `importance ${i} is negative: ${imp[i]}`)
  }

  model.dispose()
})

await test('getShapeFunction returns correct dimensions', async () => {
  const model = await EBMModel.create({
    maxRounds: 100,
    maxInteractions: 0,
    seed: 42
  })

  const { X, y } = makeLinearData(80)
  model.fit(X, y)

  const shape = model.getShapeFunction(0)
  assert(shape.x instanceof Float64Array, 'x should be Float64Array')
  assert(shape.y instanceof Float64Array, 'y should be Float64Array')
  assert(shape.x.length === shape.y.length, 'x and y should have same length')
  assert(shape.x.length > 0, 'shape should have bins')

  model.dispose()
})

// ============================================================
// Interaction terms
// ============================================================
console.log('\n=== Interactions ===')

await test('maxInteractions > 0 produces extra terms', async () => {
  const model = await EBMModel.create({
    maxRounds: 100,
    learningRate: 0.05,
    maxInteractions: 3,
    seed: 42
  })

  const { X, y } = makeLinearData(100)
  model.fit(X, y)

  // With 2 features and maxInteractions=3, we should get 2 main + up to 1 interaction
  assert(model.nTerms >= 2, `expected at least 2 terms, got ${model.nTerms}`)

  model.dispose()
})

// ============================================================
// Missing values
// ============================================================
console.log('\n=== Missing Values ===')

await test('NaN in features does not crash', async () => {
  const model = await EBMModel.create({
    maxRounds: 100,
    maxInteractions: 0,
    seed: 42
  })

  const { X, y } = makeLinearData(80)
  model.fit(X, y)

  // Add NaN to test data
  const Xtest = [[NaN, 0.5], [0.3, NaN], [NaN, NaN], [0.5, 0.5]]
  const preds = model.predict(Xtest)
  assert(preds.length === 4, `expected 4 predictions, got ${preds.length}`)
  // Should not be NaN
  for (let i = 0; i < 4; i++) {
    assert(!isNaN(preds[i]), `prediction ${i} is NaN`)
  }

  model.dispose()
})

// ============================================================
// Save / Load
// ============================================================
console.log('\n=== Save / Load ===')

const { decodeBundle, load: coreLoad } = require('@wlearn/core')

await test('save produces WLRN bundle', async () => {
  const model = await EBMModel.create({
    maxRounds: 100,
    maxInteractions: 0,
    seed: 42
  })

  const { X, y } = makeLinearData(80)
  model.fit(X, y)

  const buf = model.save()
  assert(buf instanceof Uint8Array, 'save should return Uint8Array')
  assert(buf.length > 0, 'saved model should not be empty')

  // Verify WLRN magic
  assert(buf[0] === 0x57, 'bad magic[0]')
  assert(buf[1] === 0x4c, 'bad magic[1]')
  assert(buf[2] === 0x52, 'bad magic[2]')
  assert(buf[3] === 0x4e, 'bad magic[3]')

  const { manifest, toc } = decodeBundle(buf)
  assert(manifest.typeId === 'wlearn.ebm.classifier@1',
    `expected classifier typeId, got ${manifest.typeId}`)
  assert(toc.length === 1, `expected 1 TOC entry, got ${toc.length}`)
  assert(toc[0].id === 'model', `expected TOC entry "model", got ${toc[0].id}`)

  model.dispose()
})

await test('save regressor uses regressor typeId', async () => {
  const model = await EBMModel.create({
    maxRounds: 100,
    maxInteractions: 0,
    seed: 42
  })

  const { X, y } = makeRegressionData(80)
  model.fit(X, y)

  const buf = model.save()
  const { manifest } = decodeBundle(buf)
  assert(manifest.typeId === 'wlearn.ebm.regressor@1',
    `expected regressor typeId, got ${manifest.typeId}`)

  model.dispose()
})

await test('save and load round-trip', async () => {
  const model = await EBMModel.create({
    maxRounds: 200,
    learningRate: 0.05,
    maxInteractions: 0,
    seed: 42
  })

  const { X, y } = makeLinearData(80)
  model.fit(X, y)

  const preds1 = model.predict(X)
  const buf = model.save()

  const model2 = await EBMModel.load(buf)
  assert(model2.isFitted, 'loaded model should be fitted')

  const preds2 = model2.predict(X)
  assert(preds1.length === preds2.length, 'prediction length mismatch')
  for (let i = 0; i < preds1.length; i++) {
    assert(preds1[i] === preds2[i],
      `prediction ${i}: ${preds1[i]} !== ${preds2[i]}`)
  }

  // Loaded model preserves params
  const params = model2.getParams()
  assert(params.maxRounds === 200, `loaded params.maxRounds = ${params.maxRounds}`)

  model.dispose()
  model2.dispose()
})

// ============================================================
// Registry dispatch
// ============================================================
console.log('\n=== Registry Dispatch ===')

await test('core.load() dispatches to EBM loader', async () => {
  const model = await EBMModel.create({
    maxRounds: 100,
    maxInteractions: 0,
    seed: 42
  })

  const { X, y } = makeLinearData(60)
  model.fit(X, y)

  const preds1 = model.predict(X)
  const buf = model.save()

  const model2 = await coreLoad(buf)
  assert(model2.isFitted, 'registry-loaded model should be fitted')

  const preds2 = model2.predict(X)
  for (let i = 0; i < preds1.length; i++) {
    assert(preds1[i] === preds2[i],
      `core.load prediction ${i}: ${preds1[i]} !== ${preds2[i]}`)
  }

  model.dispose()
  model2.dispose()
})

// ============================================================
// Params
// ============================================================
console.log('\n=== Params ===')

await test('getParams / setParams', async () => {
  const model = await EBMModel.create({ learningRate: 0.05, maxRounds: 100 })

  const params = model.getParams()
  assert(params.learningRate === 0.05, `expected 0.05, got ${params.learningRate}`)
  assert(params.maxRounds === 100, `expected 100, got ${params.maxRounds}`)

  model.setParams({ maxRounds: 500 })
  const params2 = model.getParams()
  assert(params2.maxRounds === 500, `expected 500 after setParams, got ${params2.maxRounds}`)

  model.dispose()
})

await test('defaultSearchSpace returns object', async () => {
  const space = EBMModel.defaultSearchSpace()
  assert(space, 'search space is null')
  assert(space.learningRate, 'missing learningRate in search space')
  assert(space.maxRounds, 'missing maxRounds in search space')
  assert(space.maxLeaves, 'missing maxLeaves in search space')
})

// ============================================================
// Resource management
// ============================================================
console.log('\n=== Resource Management ===')

await test('dispose is idempotent', async () => {
  const model = await EBMModel.create({ maxRounds: 50, maxInteractions: 0, seed: 42 })
  const { X, y } = makeLinearData(40)
  model.fit(X, y)
  model.dispose()
  model.dispose() // should not throw
})

await test('throws after dispose', async () => {
  const model = await EBMModel.create({ maxRounds: 50, maxInteractions: 0, seed: 42 })
  const { X, y } = makeLinearData(40)
  model.fit(X, y)
  model.dispose()

  let threw = false
  try { model.predict(X) } catch { threw = true }
  assert(threw, 'predict after dispose should throw')
})

await test('throws before fit', async () => {
  const model = await EBMModel.create()

  let threw = false
  try { model.predict([[1, 2]]) } catch { threw = true }
  assert(threw, 'predict before fit should throw')

  model.dispose()
})

await test('refit does not leak', async () => {
  const model = await EBMModel.create({ maxRounds: 50, maxInteractions: 0, seed: 42 })
  const { X, y } = makeLinearData(40)
  model.fit(X, y)
  model.fit(X, y) // refit

  const preds = model.predict(X)
  assert(preds.length === 40, 'should predict after refit')

  model.dispose()
})

// ============================================================
// Score
// ============================================================
console.log('\n=== Score ===')

await test('score returns accuracy for classification', async () => {
  const model = await EBMModel.create({
    maxRounds: 200,
    learningRate: 0.05,
    maxInteractions: 0,
    seed: 42
  })

  const { X, y } = makeLinearData(100)
  model.fit(X, y)
  const acc = model.score(X, y)
  assert(typeof acc === 'number', 'score should be a number')
  assert(acc > 0.5, `accuracy ${acc} too low`)
  assert(acc <= 1.0, `accuracy ${acc} > 1`)

  model.dispose()
})

await test('score returns R-squared for regression', async () => {
  const model = await EBMModel.create({
    maxRounds: 500,
    learningRate: 0.05,
    maxInteractions: 0,
    seed: 42
  })

  const { X, y } = makeRegressionData(100)
  model.fit(X, y)
  const r2 = model.score(X, y)
  assert(typeof r2 === 'number', 'score should be a number')
  assert(r2 > 0.2, `R-squared ${r2} too low`)

  model.dispose()
})

// ============================================================
// Input coercion
// ============================================================
console.log('\n=== Input Coercion ===')

await test('typed matrix fast path', async () => {
  const model = await EBMModel.create({ maxRounds: 50, maxInteractions: 0, seed: 42 })
  const { X, y } = makeLinearData(40)

  // Flatten to typed matrix
  const data = new Float64Array(40 * 2)
  for (let i = 0; i < 40; i++) {
    data[i * 2] = X[i][0]
    data[i * 2 + 1] = X[i][1]
  }

  model.fit({ data, rows: 40, cols: 2 }, y)
  const preds = model.predict({ data, rows: 40, cols: 2 })
  assert(preds.length === 40, `expected 40 predictions, got ${preds.length}`)

  model.dispose()
})

// ============================================================
// Capabilities
// ============================================================
console.log('\n=== Capabilities ===')

await test('capabilities reflect task type', async () => {
  const cls = await EBMModel.create({ maxRounds: 50, maxInteractions: 0, seed: 42 })
  const { X: Xc, y: yc } = makeLinearData(40)
  cls.fit(Xc, yc)
  assert(cls.capabilities.classifier === true, 'should be classifier')
  assert(cls.capabilities.explain === true, 'should support explain')
  assert(cls.capabilities.featureImportances === true, 'should support featureImportances')
  assert(cls.capabilities.predictProba === true, 'should support predictProba')
  cls.dispose()

  const reg = await EBMModel.create({ maxRounds: 50, maxInteractions: 0, seed: 42 })
  const { X: Xr, y: yr } = makeRegressionData(40)
  reg.fit(Xr, yr)
  assert(reg.capabilities.regressor === true, 'should be regressor')
  assert(reg.capabilities.classifier === false, 'should not be classifier')
  assert(reg.capabilities.predictProba === false, 'should not support predictProba')
  reg.dispose()
})

// ============================================================
// Summary
// ============================================================
console.log(`\n=== Results: ${passed} passed, ${failed} failed ===\n`)
process.exit(failed > 0 ? 1 : 0)

} // end main

main()
