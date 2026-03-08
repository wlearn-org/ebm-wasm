// WASM loader -- loads the libebm WASM module (singleton, lazy init)

let wasmModule = null
let loading = null

async function loadEBM(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    // SINGLE_FILE=1: .wasm is embedded in the .js file, no locateFile needed
    const createEBM = require('../wasm/ebm.js')
    wasmModule = await createEBM(options)
    return wasmModule
  })()

  return loading
}

function getWasm() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadEBM() first')
  return wasmModule
}

module.exports = { loadEBM, getWasm }
