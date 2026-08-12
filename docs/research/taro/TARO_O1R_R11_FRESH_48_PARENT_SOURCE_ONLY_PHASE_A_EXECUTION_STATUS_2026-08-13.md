# TARO O1R R11 fresh 48-parent source-only Phase A execution status

状态：`FORMAL_EXECUTION_PASS / ONE_SHOT_CONSUMED / INDEPENDENT_VALIDATION_ENV_BLOCKED / PIPELINE_HOLD`

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

独立 validator 已只读重建 exact `5,219`-file root set，重哈希全部 `5,218` 个 terminal binding，
并验证 control seals、64 个 execution-lock binding 与冻结 authority。随后 GPU 进入 Windows
`Code 43 / CM_PROB_FAILED_POST_START`，`torch.cuda.is_available()` 变为 false，验签以
`R11_PHASE_A_VALIDATION_CUDA` 环境失败停止；完整 1,043-frame lineage 与 4,172 次允许 payload
decode replay 尚未完成。validator 未改写正式 root，Phase A one-shot 不得重跑。

因此当前不能进入 source-only top-24 或 selected-only FARO。唯一下一步是重启主机恢复 CUDA，随后对同一
sealed root 只读重跑独立 validator；只有该验签 PASS 后，才可另立 top-24 implementation/execution lock。

重启后唯一允许的续验命令为：

```powershell
E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe -m scripts.research.taro_o1r_r11_abstention_runtime.validate_pool_phase_a --evidence-root artifacts.local/evidence/taro/o1r-r11-fresh-pool-phase-a-r0 --execution-lock docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_SOURCE_ONLY_PHASE_A_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json
```

本状态不产生 task effectiveness、路线晋级、部署、设备、产品或安全主张。
