# USTRF canonical observation authority / repairability audit R0 结果（2026-07-25）

状态：`SOURCE_AUTHORITY_ABSENT / VALID`

权限：`RESEARCH_ONLY / G1_CLOSED / SIGNAL_CLOSED / ANDROID_CLOSED / HUMAN_CLOSED / PRODUCTION_CLOSED`

## 结论

当前 41 条 sequence、62,229 帧的 source geometry、RGB、capture timestamp 和 frame membership 能从源侧重新核验，bbox coordinate frame 也能作为可验证 transform 追溯；但 62,229/62,229 帧均没有逐帧绑定的 canonical transform，263,680/263,680 个 observed person box 均没有 authoritative severe-truncation source fact。后者不是可从边界接触规则无假设推导的字段，因此按冻结优先级终止为：

`SOURCE_AUTHORITY_ABSENT`

这证明的是当前 evidence pack 无法承担 frozen pure-scale signal 的接受职责，不是 pure-scale 算法失败，也不是任务本身不可观测。G1 canonical repair 不得启动；合法后继只能是新的 authoritative annotation/source data pack，或停止让当前数据承担该 signal。

## 唯一研究问题与非目标

问题：现有冻结上游证据是否真实包含、且能无假设地绑定 pure-scale 与后续 ego-motion 所需的 canonical geometry、truncation、RGB continuity 和 timestamp authority？

本轮没有实现 signal、计算 slope、读取 truth outcome、生成 threshold/frontier、修 observation schema，亦没有改 Android/runtime/opener。A、B、validator 为三个不同 PID，所有 candidate/event/cell/negative/oracle/outcome/signal/truth 解码计数保持 0。

## 协议动态修正

审计中确认旧 scale producer 的所谓 candidate-blind 入口会先完整解码 candidate/lifecycle trace 再投影，不能充当真正的 G0-A；父 eligibility mask 与 denominator receipt 也含有 outcome/truth 分支。因此本轮没有复用这些对象作为 A 输入，而是：

1. A 只读取 source bundle、source frame ledger、canonical observation transport 和逐帧 RGB；逐帧冻结 orthogonal 的 `origin_authority / transform_status / value_state / scope` 与最终 authority state。
2. A 写完 inventory 并冻结 SHA 后退出。
3. B 在读取自己的配置与聚合分母前先复验 inventory SHA；只使用已人工冻结的 aggregate identity/membership projection，不打开 event/window/negative identifiers。
4. validator 在第三进程重算 41 个 frame-ledger SHA、字段状态、终态优先级和禁止解码计数。

这使“source-only inventory”与“denominator-only availability”成为真实的进程边界，而不是同进程中的逻辑约定。

## Authority matrix

| 字段族 | 逐帧状态 | 覆盖 | 判定 |
| --- | --- | ---: | --- |
| source geometry | `authoritative` | 62,229 | 39 条 CrowdBot 的 camera info 与 2 条 LILoc source bundle / PNG IHDR 可绑定 |
| canonical transform | `unknown` | 62,229 | 没有逐帧 orientation/rotation/crop/letterbox/flip receipt |
| bbox coordinate frame | `verifiable_transform` | 62,229 | canonical observation 中的 bbox 可核验，但不能倒推出缺失的完整 source-to-canonical receipt |
| severe truncation | `absent` | 62,229 | source/annotation/receipt 不提供；边界接触 heuristic 不能冒充 authority |
| RGB continuity | `authoritative` | 62,229 | 逐帧文件可读、PNG 尺寸可读且 SHA 与 source ledger 一致 |
| capture time | `authoritative` | 62,229 | source timestamp 与 observation timestamp 一致且逐序列单调 |
| frame membership | `authoritative` | 62,229 | 41/41 sequence、62,229/62,229 frame 可重建 |
| background feature input | `inferred` | 62,229 | 只形成 G3 planning information，不改变 G0 scale terminal |

39 条 CrowdBot 合计 57,635 帧，2 条 LILoc 合计 4,594 帧。观测 person box 为 263,680，与父 gap receipt 数量一致。

## Availability upper bound

由于 required family `severe_truncation` 在全部冻结帧上均为 absent，unknown/absent 必须弃权。这个全局缺口支配所有 event/window，因此无需、也不允许打开其标识符：

| 分母 | 可用 / 冻结 |
| --- | ---: |
| independent supported event | 0 / 11 |
| mechanically mapped candidate cell | 0 / 33 |
| negative interval | 0 / 836 |
| negative eligible pair | 0 / 3,801 |
| negative duration | 0 / 297,376,110,945 ns（4.95626851575 min） |
| sequence membership | 41 / 41 |
| frame membership | 62,229 / 62,229 |

33 个 cell 是 11 个独立 event 的三份机械映射，不能当作 33 个独立证据单元。父输入也未单独物化 per-track/reset first-opportunity inventory；这项限制被显式记录，但不影响 severe-truncation 全局缺失所给出的零上界。

## 判定优先级与边界

审计完整、进程隔离和禁止解码均通过，因此不是 `FAIL_CLOSED_AUDIT_INCOMPLETE`。required severe-truncation source authority 已证实 absent，所以必须在 availability insufficiency 之前命中 `SOURCE_AUTHORITY_ABSENT`，不得选择更乐观终态。

source size、RGB、timestamp 和 membership 是“已有权威、可重新绑定”的工程资产；canonical transform 是 unknown；severe truncation 是 source-authority absent。修复前四者不能创造最后一个字段的源事实。

90° rotation 不改变归一化 bbox area 的数值，这意味着原 scale contract 对 rotation 的要求可能对单独 area slope 过强；但 rotation/canonical mapping 仍决定 bbox 边界语义、truncation、G3 背景运动与未来统一 observation spine。本轮不能在看到缺口后改写冻结合同，后继新协议应把“scale 数值不变量”与“canonical boundary authority”分开预注册。

## 可复验入口

- A config SHA-256：`40a4f8b07c419c629ffdeb3368a4841e0bb052d38e187dddede53f03a86d4903`
- authority inventory SHA-256：`e383fd34e2c22ba734b4649897b7577333247db05933a8abb41099e2a2910213`
- B config SHA-256：`da08e602c497dbc1ada16bfb555fb18e36bfcdcd50fe1c773048cede6db94ffb`
- availability SHA-256：`8d0906fd339de5f4b45acdc8029a38a8f9fcd9846666bda201d34b56b6b11dcb`
- terminal SHA-256：`51fc1d78a7f6303b49f8d11eb477d6ea83c0e2ba3e1cf73658e3f2cb65e54a47`
- audit SHA-256：`46f3c6967e309e0ca07a00d3a716f375d9e3cd949356813ffd47d18f4ee98988`
- validation receipt：`4e542ec86a3f26826e4f43bab6d858900e104073c3d3963be99944538053ae28`
- A/B/validator PID：`59868 / 42744 / 3616`

验证：

```text
python scripts/run_research_tool.py ustrf-route-target-evidence-closure test_canonical_observation_authority_repairability_r0.py
........
Ran 8 tests
OK

python -m py_compile <6 G0 Python files>
PASS

python scripts/run_research_tool.py ustrf-route-target-evidence-closure validate_canonical_observation_authority_repairability_r0.py --repo . --config configs/ustrf_canonical_observation_denominator_availability_r0.json
SOURCE_AUTHORITY_ABSENT / VALID
```

## 下一合法边界

不要启动 G1，也不要再次对当前 11-event/4.956min pack 调 signal 或缩分母。若继续，下一独立目标应是 `CANONICAL_OBSERVATION_SOURCE_AUTHORITY_DATA_PACK_R0`：先定义 source-native severe-truncation/occlusion 与 frame-bound canonical transform 的最低准入合同，再对新公开来源或新采集方案做 availability-first 审计。未经新数据和相应权限，不得声称 blocker 已解除。
