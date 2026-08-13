# TARO O1R R11 fresh 48-parent source-only Phase A execution status

状态：`FORMAL_EXECUTION_PASS / ONE_SHOT_CONSUMED / CUDA_RECOVERED / ORIGINAL_VALIDATOR_NUMERIC_REPRESENTATION_STOP / ROUND12_REPAIR_ATTEMPT_01_PATH_ALIAS_PRESTART_SUPERSEDED / ROUND12_REPAIR_ATTEMPT_02_FROZEN_REVALIDATION_REQUIRED / PIPELINE_HOLD`

R11 all-48 source-only Phase A 已按冻结 module argv 正式消费。producer 原子终态为
`TARO_O1R_R11_FRESH_POOL_PHASE_A_SOURCE_ONLY_SEALED_PASS`：exact `48` parents、`1,043` frames、
`9,387` queries 与 `1,043` 次 DepthART inference 均已封存。正式 root 恰好 `5,219` files、
`959,553,693 bytes`；terminal 前为 `5,218` files、`958,520,288 bytes`。

R7 baseline 形成 `7,315 OCCUPIED / 2,072 UNKNOWN / 0 CLEAR`；R11 candidate 形成
`7,313 OCCUPIED / 2,074 UNKNOWN / 0 CLEAR`，即弱远端 abstention 把两个 R7 positive 改为
`UNKNOWN`。这只是 source-only factor landscape，不是 FARO/task outcome。

## Firewall 与资源

- 四种允许 payload 各完成 `1,043` 次读取：color、intrinsics、lowres depth、confidence；
- highres-depth member、FARO value、truth、label、outcome、training、network 均为 `0`；
- R9 parent scoring 与 top-24 selection 均未执行，`UNKNOWN` 从未作为 negative；
- wall `7,983.922 s`、OS peak RSS `1,342,758,912 bytes`、CUDA peak allocated
  `140,934,144 bytes`、terminal 前 evidence `958,520,288 bytes`，producer 报告均在冻结上限内。

## 独立验签边界

首次独立 validator 已只读重建 exact `5,219`-file root set，重哈希全部 `5,218` 个 terminal binding，
并验证 control seals、64 个 execution-lock binding 与冻结 authority。随后 GPU 进入 Windows
`Code 43 / CM_PROB_FAILED_POST_START`，验签以 `R11_PHASE_A_VALIDATION_CUDA` 环境失败停止。

主机于 2026-08-13 重启后 RTX 5060 / CUDA 12.8 恢复，原命令针对同一 root 继续运行，并在首个
candidate source receipt 以 `R11_PHASE_A_VALIDATION_SOURCE_BINDING` 停止。定位证明 producer 通过冻结的
canonical JSON 将 trajectory pose/gravity round 到 12 位，而原 validator 将独立重建的 float64 值在序列化前
作 Python exact equality；例如 `-0.403235180695` 对 `-0.4032351806954706`。重建值按冻结规则规范化后与
stored pose/gravity 的 canonical SHA 同为 `B3CCB272ACACF3EA7C41CAD6DC196AA5536523EF32293118D44553CEC49574C7`。
该停止属于 numeric representation defect，不是 source/evidence corruption。原 validator、execution lock、
terminal 和正式 root 均未改写，模型、scoring、highres/FARO/truth/label/outcome 均未重跑或读取。

因此当前仍不能进入 source-only top-24 或 selected-only FARO。新的 protocol-only repair 已冻结：保留原
validator 的全部 5,219-file/root/source/candidate/lineage/count/ledger/resource 检查，只将独立重建的
`camera_to_world_4x4` 与 `gravity_up_camera_xyz` 按 producer 的 canonical JSON round-12 规则规范化后精确比较。
它不使用 epsilon/tolerance，不修改任何旧 byte；只有 repaired audit PASS 后才可另立 top-24 lock。

Attempt 01 repair 推送后首次调用在 output-root/payload 前以 `R11_PHASE_A_REPAIR_PATH` fail closed：repo lexical
`artifacts.local` path 与同一授权 junction target 的 resolved spelling 被误作不同。正式/partial root 均未创建，
Phase-A frame payload/model/FARO 读取均为 0。Attempt 02 只在 exact CLI 与 exact authorized path 两侧都 resolve
后作 exact equality；不接受 alternate target。Attempt 02 推送后唯一允许的续验命令为：

```powershell
E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe -m scripts.research.taro_o1r_r11_abstention_runtime.audit_pool_phase_a_round12_terminal --repair-receipt docs/research/taro/TARO_O1R_R11_PHASE_A_INDEPENDENT_VALIDATOR_ROUND12_REPRESENTATION_REPAIR_ATTEMPT_02_2026-08-13.json --output-root artifacts.local/evidence/taro/o1r-r11-fresh-pool-phase-a-validator-round12-repair-r0
```

本状态不产生 task effectiveness、路线晋级、部署、设备、产品或安全主张。
