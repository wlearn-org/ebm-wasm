// WASM loader -- loads the libebm WASM module (singleton, lazy init)

import { createRequire } from 'module'

let wasmModule = null
let loading = null

export async function loadEBM(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    // SINGLE_FILE=1: .wasm is embedded in the .js file, no locateFile needed
    // Emscripten output is CJS, use createRequire for ESM compatibility
    const require = createRequire(import.meta.url)
    const createEBM = require('../wasm/ebm.cjs')
    wasmModule = await createEBM(options)
    return wasmModule
  })()

  return loading
}

export function getWasm() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadEBM() first')
  return wasmModule
}
