# TARO O1R reducer integration runtime

状态：`current / TARO_RESEARCH_MODULE / R6_REDUCER_INTEGRATION_COMPLETE / R6_NOT_EVALUABLE_ALL_UNKNOWN / HISTORICAL_EVIDENCE_READ_ONLY / NO_ACTIVE_EXECUTION`

## 稳定 Interface

- `reducer_integration.py`：接收一个封存的 R6 prospective factor bundle，生成九个 source-only uncertainty lookup，并调用唯一 deterministic interval reducer。
- `locked_uncertainty.py`：只装载并验签固定的 8-parent / 211-frame fit-only uncertainty model；不重新拟合。
- `validate_*.py` 与 `test_reducer_integration.py`：验证 protocol、implementation、execution、evidence replay、R7 task lock 和 fail-closed mechanics。

## 输出

正式 R6 evidence root 是 `artifacts.local/evidence/taro/o1r-r6-reducer-integration-r0/`，已消费且只读；测试依赖的 hash-bound uncertainty blobs 位于 `artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/`。这些本地证据不进入 Git。

## 安全边界

public runtime 不接受 FARO、truth、outcome 或 task metric。缺失 factor、support 或 uncertainty 只产生 query-local `UNKNOWN`；结构或 hash 漂移则中止。该 Module 不训练、不访问网络、不修改 App、不操作设备，也不产生部署、产品或安全证明。

## 停止条件

任一绑定输入、receipt、parent/frame/query identity、九-slot contract、source/result firewall 或证据 hash 不匹配即停止。R6 one-shot 已消费且 2,151/2,151 final states 为 `UNKNOWN`，不得重跑同一 reducer 或事后缩小 uncertainty；动态权限只由 [`docs/research/taro/README.md`](../../../docs/research/taro/README.md) 维护。

`reducer_integration.py` consumes one sealed R6 prospective factor bundle plus its bound candidate depth, confidence, Apple intrinsics, and the factory-bound R3 fit-only uncertainty model. It derives nine source-only uncertainty lookups at registered Apple centers and executes the single interval reducer that may produce `CLEAR_OBSERVED`, `OCCUPIED_OBSERVED`, or `UNKNOWN`.

`locked_uncertainty.py` verifies, hydrates, and factory-binds the exact persisted model (`3FB93A...5365`) from 8 `ADAPTER_FIT` parents and 211 frames; it does not refit or access eval residuals.

The runtime never accepts result-side geometry or metrics. Structural/hash drift aborts the bundle; ordinary missing factors, support, or uncertainty remain query-local `UNKNOWN`. It performs no training, network request, App change, or device action.

Focused validation:

```powershell
python -m scripts.research.taro_o1r_reducer_integration_runtime.validate_protocol_lock
python -m unittest scripts.research.taro_o1r_reducer_integration_runtime.test_reducer_integration
```
