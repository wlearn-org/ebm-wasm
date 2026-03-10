# @wlearn/ebm

InterpretML's Explainable Boosting Machine (EBM) compiled to WebAssembly. Interpretable machine learning in browsers and Node.js -- no server required, data stays local.

EBM is a Generalized Additive Model (GAM) trained via cyclic gradient boosting, one feature at a time. It produces per-feature shape functions that are inherently interpretable while achieving accuracy competitive with black-box models.

Part of [wlearn](https://wlearn.org) ([GitHub](https://github.com/wlearn-org), [all packages](https://github.com/wlearn-org/wlearn#repository-structure)). Based on [InterpretML v0.7.5](https://github.com/interpretml/interpret) (MIT license). Zero dependencies. CommonJS.

## Install

```bash
npm install @wlearn/ebm
```

## Quick start

```js
const { EBMModel } = require('@wlearn/ebm')

const model = await EBMModel.create({
  task: 'classification',  // or 'regression'; auto-detected from labels if omitted
  maxRounds: 500,
  seed: 42
})

// Train -- accepts number[][] or { data: Float64Array, rows, cols }
model.fit(
  [[1, 2], [3, 4], [5, 6], [7, 8]],
  [0, 0, 1, 1]
)

// Predict
const preds = model.predict([[2, 3], [6, 7]])        // Float64Array
const probs = model.predictProba([[2, 3], [6, 7]])    // Float64Array (nrow * nclass)
const accuracy = model.score([[2, 3], [6, 7]], [0, 1])

// Explain
const explanations = model.explain([[2, 3]])
// { intercept, contributions, termNames, nTerms, nSamples, nScores }
// prediction = intercept + sum(contributions)

const importances = model.featureImportances()  // Float64Array per term
const shape = model.getShapeFunction(0)         // { x, y } for plotting

// Save / load
const buf = model.save()  // Uint8Array (WLRN bundle)
const model2 = await EBMModel.load(buf)

// Clean up -- required, WASM memory is not garbage collected
model.dispose()
model2.dispose()
```

## Explainability

EBM's primary advantage over black-box models is built-in interpretability.

**Local explanations** (`explain(X)`) return per-sample, per-term additive contributions. For each sample, the prediction equals `intercept + sum(contributions)`. This tells you exactly how much each feature contributed to every prediction.

**Global importances** (`featureImportances()`) return mean absolute scores across bins for each term, showing which features matter most overall.

**Shape functions** (`getShapeFunction(i)`) return the learned response curve for each feature. For univariate terms this gives `{ x, y }` arrays for direct plotting. For interaction terms it gives `{ features, binCounts, scores }`.

## API

### `EBMModel.create(params?)` -> `Promise<EBMModel>`

Async factory. Loads WASM module on first call, returns a ready-to-use model.

### `model.fit(X, y)` -> `this`

Train the model. Returns `this`.
- `X` -- `number[][]` or `{ data: Float64Array, rows, cols }`
- `y` -- `number[]` or `Float64Array`

Task is auto-detected: integer labels become classification, non-integer becomes regression. Override with `objective: 'regression'` param.

### `model.predict(X)` -> `Float64Array`

Predict class labels (classification) or values (regression).

### `model.predictProba(X)` -> `Float64Array`

Predict class probabilities. Returns flat array of shape `nSamples * nClasses` (row-major). Binary: `[P(0), P(1)]` per sample. Classification only.

### `model.score(X, y)` -> `number`

Accuracy (classification) or R-squared (regression).

### `model.explain(X)` -> `object`

Local explanations: per-sample, per-term additive contributions.

Returns `{ intercept, contributions, termNames, nTerms, nSamples, nScores }`. The `contributions` array is flat: `nSamples * nTerms * nScores`. For each sample: `prediction = intercept + sum(contributions[sample])`.

### `model.featureImportances()` -> `Float64Array`

Global feature importances: mean absolute score per term. Length equals number of terms.

### `model.getShapeFunction(termIndex)` -> `object`

Shape function for a single term, useful for visualization.

- Univariate: `{ x: Float64Array, y: Float64Array }` -- bin edges and scores
- Interaction: `{ features, binCounts, scores, nScores }` -- 2D grid data

### `model.save()` / `EBMModel.load(buffer)`

Save to / load from `Uint8Array` (WLRN bundle with JSON model blob).

### `model.dispose()`

Free WASM memory. Required. Idempotent.

### `model.getParams()` / `model.setParams(p)`

Get/set hyperparameters. Enables AutoML grid search and cloning.

### `EBMModel.defaultSearchSpace()`

Returns default hyperparameter search space for AutoML.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `objective` | auto | `'classification'` or `'regression'`. Auto-detected from y if not set |
| `learningRate` | 0.01 | Boosting learning rate |
| `maxRounds` | 5000 | Maximum boosting rounds |
| `earlyStoppingRounds` | 50 | Rounds without improvement before stopping |
| `maxLeaves` | 3 | Maximum leaves per boosting step |
| `minSamplesLeaf` | 2 | Minimum samples per leaf |
| `maxInteractions` | 10 | Number of interaction terms (0 = no interactions) |
| `maxBins` | 256 | Maximum bins per feature |
| `minSamplesBin` | 1 | Minimum samples per bin |
| `outerBags` | 8 | Number of outer bags |
| `innerBags` | 0 | Number of inner bags |
| `regAlpha` | 0 | L1 regularization |
| `regLambda` | 0 | L2 regularization |
| `seed` | 42 | Random seed |

## Cross-runtime compatibility

Models saved in JS load and predict identically in the Python `wlearn` package. WLRN bundles round-trip between JS and Python (blob bytes are preserved). The Python wrapper uses pure numpy for prediction (no native InterpretML dependency needed at inference time).

```python
import wlearn.ebm
from wlearn import load

model = load(open('model.wlrn', 'rb').read())
preds = model.predict(X)
model.save()  # produces identical bundle
```

The Python wrapper also supports training via the `interpret` package:

```python
from wlearn.ebm import EBMModel

model = EBMModel.create({'seed': 42, 'maxRounds': 5000})
model.fit(X_train, y_train)  # requires: pip install interpret
preds = model.predict(X_test)
bundle = model.save()  # WLRN bundle loadable from JS
```

## Resource management

WASM heap memory is not garbage collected. Call `.dispose()` on every model when done. A `FinalizationRegistry` safety net warns if you forget, but do not rely on it.

## Build from source

Requires [Emscripten](https://emscripten.org/) (emsdk) activated.

```bash
git clone --recurse-submodules https://github.com/wlearn-org/ebm-wasm
cd ebm-wasm
bash scripts/build-wasm.sh
node --experimental-vm-modules test/test.js
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init
```

## Upstream

Based on [InterpretML](https://github.com/interpretml/interpret) libebm v0.7.5 (MIT license). See UPSTREAM.md for details.

## License

MIT (same as upstream InterpretML)
