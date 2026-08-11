# TARO O1R reducer integration runtime

`reducer_integration.py` consumes one sealed R6 prospective factor bundle plus its bound candidate depth, confidence, Apple intrinsics, and the factory-bound R3 fit-only uncertainty model. It derives nine source-only uncertainty lookups at registered Apple centers and executes the single interval reducer that may produce `CLEAR_OBSERVED`, `OCCUPIED_OBSERVED`, or `UNKNOWN`.

`locked_uncertainty.py` verifies, hydrates, and factory-binds the exact persisted model (`3FB93A...5365`) from 8 `ADAPTER_FIT` parents and 211 frames; it does not refit or access eval residuals.

The runtime never accepts result-side geometry or metrics. Structural/hash drift aborts the bundle; ordinary missing factors, support, or uncertainty remain query-local `UNKNOWN`. It performs no training, network request, App change, or device action.

Focused validation:

```powershell
python -m scripts.research.taro_o1r_reducer_integration_runtime.validate_protocol_lock
python -m unittest scripts.research.taro_o1r_reducer_integration_runtime.test_reducer_integration
```
