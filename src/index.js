const { loadEBM, getWasm } = require('./wasm.js')
const { EBMModel } = require('./model.js')

// Convenience: create, fit, return fitted model
async function train(params, X, y) {
  const model = await EBMModel.create(params)
  model.fit(X, y)
  return model
}

// Convenience: load WLRN bundle and predict, auto-disposes model
async function predict(bundleBytes, X) {
  const model = await EBMModel.load(bundleBytes)
  const result = model.predict(X)
  model.dispose()
  return result
}

module.exports = { loadEBM, getWasm, EBMModel, train, predict }
