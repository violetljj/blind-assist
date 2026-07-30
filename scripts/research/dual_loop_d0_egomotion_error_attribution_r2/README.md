# dual_loop_d0_egomotion_error_attribution_r2

状态：`IMPLEMENTED / NOT_LOCKED / NOT_RUN`

## 稳定 Interface

稳定入口是 `scripts/run_dual_loop_d0_egomotion_error_attribution_r2.py`。
R2 仅恢复 R1 缺失的 runtime envelope；`analysis.py`、`bindings.py` 与
`producer.py` 必须和 R1 byte-identical。

所有命令必须直接使用冻结解释器，并启用 isolated/no-bytecode：

```powershell
$py = 'E:\codex-tools\venvs\dual-loop-radial-geometry-lite-r0\Scripts\python.exe'
& $py -I -B scripts/run_dual_loop_d0_egomotion_error_attribution_r2.py runtime validate `
  --manifest artifacts.local/evidence/dual-loop/d0-egomotion-error-attribution-r2/runtime-freeze/runtime_environment_manifest.json
```

非正式测试：

```powershell
& $py -I -B scripts/run_dual_loop_d0_egomotion_error_attribution_r2.py test
```

实现提交、推送且工作树 clean 后：

```powershell
& $py -I -B scripts/run_dual_loop_d0_egomotion_error_attribution_r2.py `
  create-implementation-lock --repository-root E:\linnan\linnan --output <lock.json>

& $py -I -B scripts/run_dual_loop_d0_egomotion_error_attribution_r2.py `
  validate-implementation --implementation-lock <lock.json>

& $py -I -B scripts/run_dual_loop_d0_egomotion_error_attribution_r2.py `
  activate --repository-root E:\linnan\linnan --implementation-lock <lock.json> `
  --implementation-review <review.json> --output <activation.json>
```

只有 exact lock、独立 review 与 activation 完成后，才允许：

```powershell
& $py -I -B scripts/run_dual_loop_d0_egomotion_error_attribution_r2.py `
  produce --activation <activation.json> --implementation-lock <lock.json>

& $py -I -B scripts/run_dual_loop_d0_egomotion_error_attribution_r2.py `
  validate-execution --run-root <run-r2> --protocol <protocol.json> `
  --dependency-receipt <dependency_receipt.json> --write-results
```

## 输出

Runtime manifest 位于
`artifacts.local/evidence/dual-loop/d0-egomotion-error-attribution-r2/runtime-freeze/`。
实现锁、独立 review 与 activation 位于同一 R2 evidence 根下的
`implementation/`。正式 one-shot 只写 `run-r2/`：

- `formal_start.json`
- `event_table.jsonl`
- `analysis.json`
- `producer_receipt.json`
- `execution_validation.json`
- `execution_receipt.json`
- `progress.json`

marker 后失败只写互斥的 `failure_receipt.json`。R1 `run-r1/` 不得修改。

## 安全边界

- R1 永久是 `EXECUTION_INVALID / CONSUMED / NO_RERUN`。
- marker 前只允许协议指定的一条 Vicon operational probe；不得保留 pose、读取第二条
  message 或计算 D0 指标。
- 禁止 `PYTHONPATH`、user site、自动安装、升级或 fallback。
- 禁止旧 F-1B decision、production A/B trace、Confirmation、RCLE 和未冻结输入。
- 只有独立 validator 可以发布 `VALID` 与科学出口。
- 所有结论仅是 burned single-capture operational priority，不是因果、泛化、产品或
  安全证据。

## 停止条件

- runtime、R1 failure binding、输入、源码、仓库或 activation 任一漂移：
  `PRESTART_INVALID / NOT_RUN`，不得创建 `run-r2/`。
- marker 后任一失败：
  `EXECUTION_INVALID / CONSUMED / NO_RERUN / NO_SCIENTIFIC_EXIT`。
- validator PASS 后只允许：
  `EGO_CANARY_PRIORITY`、`TEMPORAL_TREND_PRIORITY` 或
  `NO_PRIORITY_IDENTIFIED`。
- D0 不自动授权后继 canary、Confirmation、Android、产品或安全工作。
