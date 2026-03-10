const { loadEBM, getWasm } = require('./wasm.js')
const { EBMModel: EBMModelImpl } = require('./model.js')
const { createModelClass } = require('@wlearn/core')

const EBMModel = createModelClass(EBMModelImpl, EBMModelImpl, { name: 'EBMModel', load: loadEBM })

// Convenience: create, fit, return fitted model
async function train(params, X, y) {
  const model = await EBMModel.create(params)
  await model.fit(X, y)
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
