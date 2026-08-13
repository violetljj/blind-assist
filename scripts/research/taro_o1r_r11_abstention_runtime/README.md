# TARO O1R R11 abstention runtime

状态：`current / TARO_RESEARCH_MODULE / R11_WEAK_DISTAL_ABSTENTION_FALSIFIED_ON_FRESH_COHORT / R11_FRESH_48_PARENT_PIPELINE_COMPLETE / R11_PHASE_A_INDEPENDENT_VALIDATION_PASS / R11_SOURCE_ONLY_TOP24_INDEPENDENT_VALIDATION_PASS / R11_SELECTED_TOP24_FARO_PHASE_B_ONE_SHOT_CONSUMED / R11_PHASE_B_INDEPENDENT_VALIDATION_PASS / R11_NOT_EVALUABLE_DUAL_CLASS_COVERAGE / R11_NO_PROMOTION / DEFAULT_APP_UNCHANGED`

## 稳定 Interface

- `abstention_candidate.py`：只接受 sealed R7 source feature；2-pixel base positive 还必须在 16-pixel、0.15m height 或 1.5m forward 三个相邻强度 cell 中至少命中一个，否则只变为 `UNKNOWN`，不输出 `CLEAR`。
- `development_replay.py`：只读重放已消费 R10 的 sealed source/label evidence，量化候选，不读取原始 FARO、模型或新数据。
- `fresh_pool.py`：从固定 Git exclusion snapshot 与 R10 全 32-parent source pool 重算 exact R11 metadata-only pool。
- `validate_protocol_lock.py`：验证 development lineage、算法、fresh roster、source/FARO firewall、双类门和 execution=false 权限。
- `run_pool_head.py`：只接受单独提交且 hash-bound 的 R11 execution lock；对 exact 144 URL 执行 no-redirect、zero-body HEAD，并封存逐 attempt receipt。它不能下载正文、解码 source、运行模型或读取 FARO。
- `run_pool_download.py`：只接受另行提交、绑定 HEAD evidence 与 implementation commit 的 one-shot execution lock；按 144-row 冻结顺序执行受限 GET，逐文件校验 HEAD 长度/validator、SHA-256 与 CRC32。最多三次仅限 transient transport retry；archive/source decode、模型与 FARO 始终关闭。
- `run_pool_inventory.py`：只接受后继独立 one-shot lock；先创建 exclusive evidence root，再重验 144 个下载文件。ZIP 只索引 central directory 的路径、声明尺寸与声明 CRC，不调用 `testzip/open/read`，不读取或解压任何 member payload；trajectory 只用于 exact-ns pose-bounded frame plan。
- `validate_pool_phase_a.py`：原始、hash-bound 的独立 Phase-A validator；它在 CUDA 恢复后因把 producer 的 round-12 JSON pose/gravity 与重建的 float64 值作序列化前精确比较而停止，文件保持不变。
- `audit_pool_phase_a_round12_terminal.py`：新绑定的只读薄修复；保留原 validator 全部 root/hash/source/candidate/lineage/ledger/resource 检查，只把独立重建的 `camera_to_world_4x4` 和 `gravity_up_camera_xyz` 按冻结 canonical JSON 规则 round 到 12 位后作精确比较。Attempt 02 另将 exact repo-relative CLI path 与同一授权 junction target 作 resolved exact equality；alternate target 仍拒绝。无 epsilon、无模型重跑、无 source scoring/FARO 权限。
- `run_top24_selection.py` / `validate_top24_selection.py`：从 48-parent sealed source-only features 冻结 eligible-count 排名与 top24；正式 one-shot 及独立重算均已 PASS，未读取 FARO。
- `phase_b_metrics.py`：实现 R11 dual-class evaluability、candidate/R7 recall-loss、clear-frame/macro/Wilson 与 abstention-effect 固定 reducer；truth `UNKNOWN` 不进任何负类分母。
- `run_selected_phase_b.py`：仅允许 sealed selected 24 的 674 个 highres-depth payload，并以 single-frame geometry、terminal-last sibling partial root 原子发布结果。
- `validate_selected_phase_b.py`：不导入 Phase B producer/reducer，不重读 FARO payload；从 sealed R7/R11/label records 独立重算 6,066-query metrics、gates、terminal 与 exact root receipts。

## 输出

协议、授权 receipt、implementation/lock/result 与 Phase-A validator repair receipt 写入 Git；HEAD、download、inventory、Phase A、top24 与 Phase B evidence 均封存在 consumed `artifacts.local/` roots。Phase A 48/1,043/9,387、top24 24/674/6,066 和 selected FARO 674/674 均已独立复核。Phase B 为 `NOT_EVALUABLE_DUAL_CLASS_COVERAGE`：28 个 definite-CLEAR queries 只覆盖 10 个 physical frames；R11 在 definite labels 上与 R7 完全相同，仅额外 abstain 1 个 truth-UNKNOWN query。

## 安全边界

R10 与 R11 都是 consumed evidence，不能被改门、重跑或写成 PASS。R11 候选只有 `OCCUPIED/UNKNOWN`；truth `UNKNOWN` 永远不是 negative。top24 已不可变，unselected FARO 保持 0。后继仅可把 sealed R11 evidence 用于明确标注的 Development discovery；新的 confirmation 必须使用 untouched parents。

## 停止条件

任一 R10 hash、source-only API、16-pixel cell、fresh parent exclusion、selector、dual-class gate、phase firewall 或 authority 漂移即停止。动态状态以 [`docs/research/taro/README.md`](../../../docs/research/taro/README.md) 为唯一真源。
