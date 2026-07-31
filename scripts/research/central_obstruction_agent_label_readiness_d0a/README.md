# Central obstruction Agent label readiness D0-A

状态：canary / D0-A0 complete valid / D0-A1 complete not reliable / D0-A successor R0 complete auxiliary-only

## 研究问题与版本

本 Module 执行 `CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A` 的 D0-A0 与
D0-A1，及其不继承 D0-A2 权限的固定 clip successor。D0-A0 当前 evidence instance 为
`CENTRAL_OBSTRUCTION_D0_A0_INPUT_UNIVERSE_R3`，只允许形成现有连续 RGB
source/session/frame/timestamp/ancestry 清单与 SHA-256 回执。D0-A1 当前为
`CENTRAL_OBSTRUCTION_D0_A1_LABELABILITY_PILOT_R2`，只使用 calibration-only
source 冻结 ROI、prompt、parent-event、audit 与 readiness 合同；其终态为
`AGENT_LABEL_PROTOCOL_NOT_RELIABLE`。successor R0 当前 evidence instance 为
`CENTRAL_OBSTRUCTION_D0_A_SUCCESSOR_FIXED_CLIP_CALIBRATION_R0`，只保留 observation-level
Agent 标签，并由程序从冻结时间窗/slot 生成固定 clip unit，不生成 natural-event boundary。

## 稳定 Interface

从仓库根目录运行：

```powershell
python -m scripts.research.central_obstruction_agent_label_readiness_d0a.freeze_input_universe
python -m scripts.research.central_obstruction_agent_label_readiness_d0a.validate_input_universe
python -m scripts.research.central_obstruction_agent_label_readiness_d0a.freeze_d0a1_pilot
python -m scripts.research.central_obstruction_agent_label_readiness_d0a.validate_d0a1_pilot
python -m scripts.research.central_obstruction_agent_label_readiness_d0a.transcribe_d0a1_primary_review
python -m scripts.research.central_obstruction_agent_label_readiness_d0a.validate_d0a1_primary_review
python -m scripts.research.central_obstruction_agent_label_readiness_d0a.validate_d0a1_isolated_review --review-path <fresh-isolated-review.json>
python -m scripts.research.central_obstruction_agent_label_readiness_d0a.finalize_d0a1_adjudication --review-path <fresh-adjudication-review.json>
python -m scripts.research.central_obstruction_agent_label_readiness_d0a.validate_d0a1_final_readiness
python -m scripts.research.central_obstruction_agent_label_readiness_d0a.freeze_d0a_successor_calibration
python -m scripts.research.central_obstruction_agent_label_readiness_d0a.validate_d0a_successor_calibration
```

输入由 `source_universe_r3.json` 继承 R2/R1/R0 source inventory，并冻结当前 evidence
identity。Producer 和 validator 都要求源账本、逐帧 payload、协议、workflow、
reuse-role ledger 与 fitness review 哈希闭合；正式输出已存在时拒绝覆盖。

## 输出

只写入：

```text
artifacts.local/evidence/central-obstruction-agent-label-readiness-d0-a-r3/
```

正式文件为 `input-universe-manifest.json`、`input-universe-receipt.json`、
`reuse-role-ledger.jsonl`、`reuse-fitness-review.json` 和
`input-universe-validation.json`。逐帧图像不复制。

D0-A1 只写入：

```text
artifacts.local/evidence/central-obstruction-agent-label-readiness-d0-a1-r2/
```

它保存 pilot input/receipt/validation、三份不可覆盖 raw review、各自 parent-event、
初始 agreement、冻结 adjudication packet、canonical calibration ledger 与最终 readiness。
R2 终态为 `AGENT_LABEL_PROTOCOL_NOT_RELIABLE`：observation agreement 过门，但 raw
parent-event match `0.6316 < 0.75`，D0-A2 不授权。

D0-A successor R0 只写入：

```text
artifacts.local/evidence/central-obstruction-agent-label-readiness-d0-a-successor-r0/
```

它保存 content-blind fixed-clip input、两个 observation-only review、capture receipt、
固定单位派生结果与最终 validation。当前结果为
`CENTRAL_OBSTRUCTION_AUXILIARY_FEATURE_ONLY`：边界转换通过，但 fresh observation
agreement/固定单位状态一致性失败；不启动 D0-A2、D0-A3 或 D0-A4。

successor R0 将 D0-A1 的 burned 11 clips 仅作为规则设计材料；fresh calibration 使用
3 个未出现在 burned calibration 中的 D0-A0 production-universe session，冻结 6 个
one-second fixed clips / 24 个 observation slots。unit boundary 不读取 label，转换函数
只输出 `STABLE_PRESENT`、`STABLE_NO_EVIDENCE`、`MIXED_OBSERVATION` 或 `NOT_EVALUABLE`。
successor 只运行两个 fresh isolated observation passes，不启动 D0-A3/A4；任一 hard gate
失败都将中央阻塞降为 `AUXILIARY_FEATURE_ONLY` 并停止本路线扩展。

## 安全边界

两个阶段只读取 RGB source ledger、逐帧 payload 和 provenance receipt。禁止读取
YOLO、分割、深度、融合、scheduler、risk、feedback、truth、review 或旧 candidate
effect 输出。D0-A1 raw primary label 只来自 source-only calibration view，并明确为
非隔离 pass；所有来源均披露既有 content/output access，因此不具有 unseen
Confirmation authority。

## 停止条件

任一源账本漂移、payload 缺失/哈希不符、frame index 不连续、timestamp 非严格递增、
路径越界、重复 session 或正式输出已存在时 fail closed；不缩短困难片段、不替换来源。
D0-A1 缺 fresh isolated pass、任一 denominator 为空或任一冻结门失败时不得启动
D0-A2；绝不把 primary-only pass 当 agreement=1，也不以第三方裁决覆盖 raw reviewer
agreement 或 event-match failure。

## 假设与规则质疑

本阶段不检验中央阻塞标签本身，只检验一个更窄的前置假设：当前 dual-loop natural-event
连续 RGB 来源能否被完整、可复算地冻结。若该输入身份无法闭合，应先修 source
inventory，而不是通过抽帧、裁剪或删除不便片段降低门槛。

## 失败资产复用

成功清单可作为 source characterization 与后续 D0-A 输入锁；失败 ledger、缺失 payload
或 timestamp 异常只作为数据完整性诊断/回归 fixture，不得包装为标签或模型效果证据。
