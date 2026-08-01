# HFTF Stage C source feasibility result C0.1

日期：2026-08-01

终态：`C0_1_STAGE_C_SOURCE_TRANSPORT_FEASIBILITY_SUPPORTED`

## 1. 结论

C0.1 在同一 hash-bound consumed cohort 上通过最小 timebase schema repair：
container nominal 100 Hz 只记录，物理 timeline 由 parquet frame/timestamp 与
`meta/info.json=5` 定义。

两条 trajectory 的 parquet 有效 rate 都为精确 `5.0 Hz`，pose/RGB/depth frame count
精确一致，PTS 仍严格 constant-step；原 C0 的文件绑定、完整 decode、32-frame depth
support 和 UNKNOWN firewall 全部保持通过。

因此当前只得到
`CONSUMED_SOURCE_SCHEMA_REPAIR_AND_NATURAL_DEPTH_SURFACE_OBSERVABILITY_ONLY`。
它不是 ground hazard truth、student effect 或助盲事件证据。

## 2. 报告绑定

- report：
  `artifacts.local/evidence/hftf/stage-c-c0-1-egowalk-timebase-repair-20260801/timebase_repair.json`
- SHA-256：
  `071c8e9aa7fd36ee6682ef836f7dfed09120f2db24e5779b0c109cc55bc72024`
- C0.1 protocol commit：`dcc261e`
- C0.1 runner commit：`754719d`

## 3. 逐 source 结果

| trajectory | pose/RGB/depth | timestamp delta ms | effective rate | container nominal |
| --- | ---: | --- | ---: | --- |
| `2024_08_15__19_45_11` | `647/647/647` | `198/200/201` | `5.0 Hz` | `100/100 Hz` |
| `2024_07_11__12_33_57` | `664/664/664` | `198/200/201` | `5.0 Hz` | `100/100 Hz` |

两条的 C0.1 gate failures 均为空，unchanged surface observability 均通过。predecessor
C0 的唯一 failures 仍精确为 RGB/depth nominal-rate mismatch，没有被隐藏。

## 4. 权限

唯一新权限：

`FREEZE_STAGE_C_LABEL_AND_STUDENT_CANARY_PROTOCOL_ONLY`

尚未授权正式 geometry-label execution、student training/effect、研究主线切换、
Android/App、生产或安全 claim。后续协议必须先让 semantic-independent natural depth
reader 产生可审计的 known/UNKNOWN 与 opportunity，再允许 student 训练；解析 terrain
不能因没有 RGB 被静默当成 unified student source。
