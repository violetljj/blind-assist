# TARO O1R R11 abstention runtime

状态：`current / TARO_RESEARCH_MODULE / R11_WEAK_DISTAL_ABSTENTION_DEVELOPMENT_ONLY / R11_PROTOCOL_LOCKED / R11_EXACT_DATA_USE_AUTHORIZED / R11_HEAD_PASS_ONE_SHOT_CONSUMED / R11_DOWNLOAD_ATTEMPT_01_PRESTART_SUPERSEDED / R11_SOURCE_DOWNLOAD_144_OF_144_INTEGRITY_PASS_ONE_SHOT_CONSUMED / R11_INVENTORY_IMPLEMENTATION_LOCK_PASS / R11_INVENTORY_FORMAL_NOT_RUN / R11_FORMAL_ZIP_MEMBER_PAYLOAD_READS_ZERO / R11_SCIENTIFIC_NOT_RUN / DEFAULT_APP_UNCHANGED`

## 稳定 Interface

- `abstention_candidate.py`：只接受 sealed R7 source feature；2-pixel base positive 还必须在 16-pixel、0.15m height 或 1.5m forward 三个相邻强度 cell 中至少命中一个，否则只变为 `UNKNOWN`，不输出 `CLEAR`。
- `development_replay.py`：只读重放已消费 R10 的 sealed source/label evidence，量化候选，不读取原始 FARO、模型或新数据。
- `fresh_pool.py`：从固定 Git exclusion snapshot 与 R10 全 32-parent source pool 重算 exact R11 metadata-only pool。
- `validate_protocol_lock.py`：验证 development lineage、算法、fresh roster、source/FARO firewall、双类门和 execution=false 权限。
- `run_pool_head.py`：只接受单独提交且 hash-bound 的 R11 execution lock；对 exact 144 URL 执行 no-redirect、zero-body HEAD，并封存逐 attempt receipt。它不能下载正文、解码 source、运行模型或读取 FARO。
- `run_pool_download.py`：只接受另行提交、绑定 HEAD evidence 与 implementation commit 的 one-shot execution lock；按 144-row 冻结顺序执行受限 GET，逐文件校验 HEAD 长度/validator、SHA-256 与 CRC32。最多三次仅限 transient transport retry；archive/source decode、模型与 FARO 始终关闭。
- `run_pool_inventory.py`：只接受后继独立 one-shot lock；先创建 exclusive evidence root，再重验 144 个下载文件。ZIP 只索引 central directory 的路径、声明尺寸与声明 CRC，不调用 `testzip/open/read`，不读取或解压任何 member payload；trajectory 只用于 exact-ns pose-bounded frame plan。

## 输出

协议、授权 receipt、development 小型结果和 validator 写入 Git；HEAD evidence 与 Attempt 02 download evidence 均封存在 consumed `artifacts.local/` roots。144 个 source blobs 已完整下载并重验；正式 inventory 尚未运行，ZIP member payload 和 source frame 仍未由正式阶段打开。

## 安全边界

R10 只作为 consumed development evidence，不能被改门、重跑或写成 confirmation。R11 候选只有 `OCCUPIED/UNKNOWN`；`UNKNOWN` 永远不是 negative。download Attempt 02 已 144/144 PASS 并消费。后继 inventory 把 highres/FARO member 的 central-directory 名称、声明尺寸和声明 CRC 视为 metadata；任何 member 解压或 payload byte 都算 FARO read 并在 sealed top-24 前禁止。声明 CRC 不冒充 payload CRC 验证。

## 停止条件

任一 R10 hash、source-only API、16-pixel cell、fresh parent exclusion、selector、dual-class gate、phase firewall 或 authority 漂移即停止。动态状态以 [`docs/research/taro/README.md`](../../../docs/research/taro/README.md) 为唯一真源。
