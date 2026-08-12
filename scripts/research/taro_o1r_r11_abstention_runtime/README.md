# TARO O1R R11 abstention runtime

状态：`current / TARO_RESEARCH_MODULE / R11_WEAK_DISTAL_ABSTENTION_DEVELOPMENT_ONLY / R11_PROTOCOL_LOCKED / R11_SCIENTIFIC_NOT_RUN / EXECUTION_FALSE / DEFAULT_APP_UNCHANGED`

## 稳定 Interface

- `abstention_candidate.py`：只接受 sealed R7 source feature；2-pixel base positive 还必须在 16-pixel、0.15m height 或 1.5m forward 三个相邻强度 cell 中至少命中一个，否则只变为 `UNKNOWN`，不输出 `CLEAR`。
- `development_replay.py`：只读重放已消费 R10 的 sealed source/label evidence，量化候选，不读取原始 FARO、模型或新数据。
- `fresh_pool.py`：从固定 Git exclusion snapshot 与 R10 全 32-parent source pool 重算 exact R11 metadata-only pool。
- `validate_protocol_lock.py`：验证 development lineage、算法、fresh roster、source/FARO firewall、双类门和 execution=false 权限。

## 输出

协议、development 小型结果和 validator 写入 Git；任何后续 source/evidence 只能写入未来 execution lock 指定的 `artifacts.local/` exclusive root。当前没有 R11 evidence root。

## 安全边界

R10 只作为 consumed development evidence，不能被改门、重跑或写成 confirmation。R11 候选只有 `OCCUPIED/UNKNOWN`；`UNKNOWN` 永远不是 negative。当前不授权 HEAD、source body、模型、FARO、训练、设备、部署、产品或安全主张。

## 停止条件

任一 R10 hash、source-only API、16-pixel cell、fresh parent exclusion、selector、dual-class gate、phase firewall 或 authority 漂移即停止。动态状态以 [`docs/research/taro/README.md`](../../../docs/research/taro/README.md) 为唯一真源。
