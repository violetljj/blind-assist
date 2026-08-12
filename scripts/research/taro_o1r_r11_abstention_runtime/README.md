# TARO O1R R11 abstention runtime

状态：`current / TARO_RESEARCH_MODULE / R11_WEAK_DISTAL_ABSTENTION_DEVELOPMENT_ONLY / R11_PROTOCOL_LOCKED / R11_EXACT_DATA_USE_AUTHORIZED / R11_HEAD_PASS_ONE_SHOT_CONSUMED / R11_DOWNLOAD_ATTEMPT_01_PRESTART_SUPERSEDED / R11_DOWNLOAD_ATTEMPT_02_IMPLEMENTATION_READY / R11_SOURCE_UNOPENED / R11_SCIENTIFIC_NOT_RUN / DEFAULT_APP_UNCHANGED`

## 稳定 Interface

- `abstention_candidate.py`：只接受 sealed R7 source feature；2-pixel base positive 还必须在 16-pixel、0.15m height 或 1.5m forward 三个相邻强度 cell 中至少命中一个，否则只变为 `UNKNOWN`，不输出 `CLEAR`。
- `development_replay.py`：只读重放已消费 R10 的 sealed source/label evidence，量化候选，不读取原始 FARO、模型或新数据。
- `fresh_pool.py`：从固定 Git exclusion snapshot 与 R10 全 32-parent source pool 重算 exact R11 metadata-only pool。
- `validate_protocol_lock.py`：验证 development lineage、算法、fresh roster、source/FARO firewall、双类门和 execution=false 权限。
- `run_pool_head.py`：只接受单独提交且 hash-bound 的 R11 execution lock；对 exact 144 URL 执行 no-redirect、zero-body HEAD，并封存逐 attempt receipt。它不能下载正文、解码 source、运行模型或读取 FARO。
- `run_pool_download.py`：只接受另行提交、绑定 HEAD evidence 与 implementation commit 的 one-shot execution lock；按 144-row 冻结顺序执行受限 GET，逐文件校验 HEAD 长度/validator、SHA-256 与 CRC32。最多三次仅限 transient transport retry；archive/source decode、模型与 FARO 始终关闭。

## 输出

协议、授权 receipt、development 小型结果和 validator 写入 Git；HEAD evidence 已封存在 consumed `artifacts.local/` exclusive root。Attempt 01 因 direct-script import 在 root/GET 前停止；Attempt 02 implementation 已改为稳定 `python -m` 入口，但新 execution lock 尚未提交，source/evidence root 仍不存在。

## 安全边界

R10 只作为 consumed development evidence，不能被改门、重跑或写成 confirmation。R11 候选只有 `OCCUPIED/UNKNOWN`；`UNKNOWN` 永远不是 negative。HEAD 已以 144/144 PASS 消费；download Attempt 01 未消费但已 supersede，Attempt 02 lock 尚不存在，source body、模型与 FARO 仍未打开。

## 停止条件

任一 R10 hash、source-only API、16-pixel cell、fresh parent exclusion、selector、dual-class gate、phase firewall 或 authority 漂移即停止。动态状态以 [`docs/research/taro/README.md`](../../../docs/research/taro/README.md) 为唯一真源。
