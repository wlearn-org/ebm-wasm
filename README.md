# @wlearn/ebm

InterpretML's Explainable Boosting Machine (EBM) compiled to WebAssembly. Interpretable machine learning in browsers and Node.js -- no server required, data stays local.

EBM is a Generalized Additive Model (GAM) trained via cyclic gradient boosting. It produces per-feature shape functions that are inherently interpretable while achieving accuracy competitive with black-box models.

## Quick start

```js
import { EBMModel } from '@wlearn/ebm'

const model = await EBMModel.create({ maxRounds: 500 })
model.fit(X, y)

const predictions = model.predict(X)
const probabilities = model.predictProba(X)
const explanations = model.explain(X)
const importances = model.featureImportances()

// Save and load
const bundle = model.save()
const loaded = await EBMModel.load(bundle)

model.dispose()
```

## API

### `EBMModel.create(params?)` -> `Promise<EBMModel>`

Creates a new model. WASM is loaded on first call.

### `model.fit(X, y)` -> `this`

Train the model. X is `number[][]` or `{ data: Float64Array, rows, cols }`. y is `number[]` or `Float64Array`.

### `model.predict(X)` -> `Float64Array`

Predict class labels (classification) or values (regression).

### `model.predictProba(X)` -> `Float64Array`

Predict class probabilities. Returns `[nSamples * nClasses]` array.

### `model.score(X, y)` -> `number`

Accuracy (classification) or R-squared (regression).

### `model.explain(X)` -> `object`

Local explanations: per-sample, per-term additive contributions.

Returns `{ intercept, contributions, termNames, nTerms, nSamples }`.

For each sample: `prediction = intercept + sum(contributions[sample])`.

### `model.featureImportances()` -> `Float64Array`

Global feature importances: mean absolute score per term.

### `model.getShapeFunction(termIndex)` -> `object`

Per-feature shape function for visualization.

Returns `{ x, y }` for univariate terms, `{ x1, x2, scores }` for interactions.

### `model.save()` -> `Uint8Array`

Save to WLRN bundle format.

### `EBMModel.load(bytes)` -> `Promise<EBMModel>`

Load from WLRN bundle.

### `model.dispose()`

Free WASM resources. Must be called when done.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learningRate` | 0.01 | Boosting learning rate |
| `maxRounds` | 5000 | Maximum boosting rounds |
| `earlyStoppingRounds` | 50 | Rounds without improvement before stopping |
| `maxLeaves` | 3 | Maximum leaves per tree |
| `minSamplesLeaf` | 2 | Minimum samples per leaf |
| `maxInteractions` | 10 | Number of interaction terms (0 = no interactions) |
| `maxInteractionBins` | 32 | Maximum bins for interaction terms |
| `outerBags` | 8 | Number of outer bags |
| `innerBags` | 0 | Number of inner bags |
| `maxBins` | 256 | Maximum bins per feature |
| `minSamplesBin` | 1 | Minimum samples per bin |
| `seed` | 42 | Random seed |

## Upstream

Based on [InterpretML](https://github.com/interpretml/interpret) (MIT license). See UPSTREAM.md for details.
