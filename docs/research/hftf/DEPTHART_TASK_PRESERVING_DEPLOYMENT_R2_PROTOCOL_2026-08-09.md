# DepthART task-preserving deployment R2 protocol

状态：`PROTOCOL_FROZEN / EXECUTION_NOT_ACTIVATED / NO_OUTCOME_ACCESSED`

机器合同：[`DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2_PROTOCOL_2026-08-09.json`](DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2_PROTOCOL_2026-08-09.json)

前置筛选：[D0 precision screen](DEPTHART_TASK_PRESERVING_D0_PRECISION_SCREEN_PROTOCOL_2026-08-09.md) ·
[common source/control lock](DEPTHART_TASK_PRESERVING_D0_SOURCE_CONTROL_LOCK_2026-08-09.json) ·
[ARKit roster lock](DEPTHART_TASK_PRESERVING_R2_ARKIT_ROSTER_LOCK_2026-08-09.json) ·
[media HEAD preflight](DEPTHART_TASK_PRESERVING_R2_ARKIT_MEDIA_PREFLIGHT_2026-08-09.json)

## 决策与不可变前提

当前 QAIRT 2.47、SM8650/v75 标准 HTP float primitive 路径已经得到
`CURRENT_QAIRT_2_47_SM8650_HTP_STANDARD_FLOAT_PATH_STRICT_G4D_NOT_SUPPORTED`。
R2 不修改、放宽、重跑或“修复”该终态，也不继续把标准 Conv、Norm 和 activation
逐族 custom 化。R2 回答的是另一个问题：允许 raw depth tensor 不满足逐元素 strict
FP32 parity 时，一个冻结的 HTP-friendly DepthART 图能否保持 BlindAssist 真正使用的
通行空间和风险决策。

本协议的增加理由是 `INFORMATION_GAIN / REPRODUCIBILITY / CLAIM_INTEGRITY`。当前只冻结
问题、输入合同和判定门；没有新 cohort、候选输出、任务结果、设备性能或晋级结论。

## 科学问题与两臂

- Reference：关闭 CUDA/CuDNN TF32 的 canonical PyTorch DepthART-S，保持
  `image,K -> depth` 和冻结任务后处理。
- Candidate：必须先在独立 R2 cohort 之外完成一个预冻结的 Development screen；screen receipt
  之后只允许冻结一个 HTP-friendly 图进入 R2。不得在新独立 cohort 上比较多个候选或调阈值。
- Truth：独立 RGB-D/几何事实只用于计算任务指标；reference 与 candidate 同帧、同 intrinsics、
  同任务后处理。`UNKNOWN_GROUND` 永远不是 negative，也不得从分母中静默删除。

R2 必须同时回答：candidate 相对 reference 是否任务非劣，以及 candidate 对独立 truth
是否满足绝对底线。只与 reference 接近但两者都错，不能 PASS。

## 数据与激活

正式执行采用 `EVIDENCE_TRACK / CONFIRMATION_STRICT`。激活前必须建立新 cohort，满足：

- parent 与 session 对开发、R0 consumed 120-frame TUM、G4-D synthetic canary 全部独立；
- 连续帧、每帧 intrinsics、三 band/既定 horizon 的 truth clearance 和有效性可配对；
- roster 在 outcome access 前冻结，明确 provenance、许可、parent/session 身份和缺失规则；
- reference/candidate graph、checkpoint、量化/精度配置、task postprocess 和 runtime 均以
  SHA-256 或等价不可变身份绑定；
- pre-outcome activation manifest 必须由
  `validate_depthart_task_preserving_r2_activation.py` 校验通过；该 PASS 仍不等于执行授权。

用户需要在有效 manifest 之后显式激活 outcome access。当前状态不允许运行 reference 或
candidate 生成该 cohort 的 claim-bearing 输出。

截至 2026-08-09，公共 source/control 已锁定；D0 FP16/W8A16/INT8 三臂均在 outcome 前技术
前门淘汰，未产生 selection receipt，因此 R2 单一 candidate 尚未选定。既有 G4-C
fixed-mixed 图只是诊断 control，若继续必须另立 D1，不能事后塞回 D0。ARKitScenes 元数据
已锁定 8 个与既有 HFTF 身份零重叠的 Validation visit/session，
32 个官方资产 HEAD 全部可用，总计 `2,725,771,890` bytes，但本 roster 尚未取得显式 license
scope extension，媒体 payload 仍为零读取。它不得参与 D0 calibration 或候选选择。

## 冻结任务门

所有门是 conjunction；undefined、非有限值、缺失 parent/session 宏平均或分母为零均 FAIL，
不得改成跳过。除 pooled 值外，必须报告 parent-macro、session-macro 和 worst-parent。

| 指标 | Candidate 对 truth 绝对门 | 相对 canonical reference 非劣门 |
|---|---:|---:|
| known coverage | `>= 0.90` | 下降 `<= 0.02` |
| clearance MAE | `<= 0.20 m` | 增加 `<= 0.025 m` |
| false-clear / all known | `<= 0.08` | 增加 `<= 0.01` |
| false-clear / truth occupied | 必须报告且 finite | 增加 `<= 0.02` |
| false-block / truth clear | `<= 0.02` | 增加 `<= 0.01` |
| temporal clearance-delta MAE | `<= 0.15 m` | 增加 `<= 0.025 m` |
| geometry transition agreement | `>= 0.90` | 下降 `<= 0.02` |
| valid-to-UNKNOWN rate | `<= 0.10` | 增加 `<= 0.02` |

false-clear 是主要风险门；false-block 和 known coverage 防止候选通过“全部 UNKNOWN/封路”
获得虚假安全。任何聚合 PASS 都不能覆盖 worst-parent 的 false-clear 绝对门失败；
worst-parent false-clear/all-known 必须 `<= 0.12`。

## 顺序与终态

1. `PRE_OUTCOME_CONTRACT_VALID`：只证明 manifest 完整；保持未激活。
2. 显式激活后，先生成并独立复核 task-quality receipt。
3. 任一任务门失败：`TASK_PRESERVING_R2_QUALITY_FAIL_STOP`，不得测性能来回救。
4. 任务门全部通过：`TASK_PRESERVING_R2_QUALITY_PASS_PERFORMANCE_ELIGIBLE`；只授权该候选
   的 partition purity、latency、RAM 和 thermal Development/Deployment 测量。
5. 性能通过也只产生 bounded device-route evidence；DA2 replacement、默认 App、生产与
   safety 始终 `NOT_AUTHORIZED`，需要另立 promotion/release gate。

近完整 custom-float32 engine 或不同 runtime/hardware 是不同立项，不属于本协议的候选搜索。
