# Changelog

## 0.1.0 (unreleased)

- Initial release
- EBM classifier and regressor via InterpretML libebm v0.7.5 compiled to WASM
- Per-feature shape functions and local explanations (explainability)
- Feature importances from learned shape functions
- Interaction term detection and training
- Save/load via WLRN bundle format (JSON model blob)
- Python wrapper: pure numpy inference (no native dependency for predict/save/load)
- Python fit() via interpret package (optional dependency)
- Cross-language golden fixture tests (JS trains, Python loads, predictions match)
- 27 JS tests, 22 Python fit tests, cross-language compat tests in wlearn core repo
