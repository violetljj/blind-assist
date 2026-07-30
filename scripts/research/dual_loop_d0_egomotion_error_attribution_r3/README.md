# dual_loop_d0_egomotion_error_attribution_r3

状态：`IMPLEMENTED / NOT_LOCKED / NOT_RUN`

## 稳定 Interface

稳定入口是 `scripts/run_dual_loop_d0_egomotion_error_attribution_r3.py`。
R3 只恢复 R2 缺失的 PyYAML runtime dependency 与完整性门；`analysis.py`、
`bindings.py` 与 `producer.py` 必须和 R1/R2 byte-identical。

所有命令必须直接使用冻结解释器，并启用 isolated/no-bytecode：

```powershell
$py = 'E:\codex-tools\venvs\dual-loop-d0-egomotion-r3\Scripts\python.exe'
& $py -I -B scripts/run_dual_loop_d0_egomotion_error_attribution_r3.py runtime validate `
  --manifest artifacts.local/evidence/dual-loop/d0-egomotion-error-attribution-r3/runtime-freeze/runtime_environment_manifest.json
```

非正式测试：

```powershell
& $py -I -B scripts/run_dual_loop_d0_egomotion_error_attribution_r3.py test
```

实现提交、推送且工作树 clean 后：

```powershell
& $py -I -B scripts/run_dual_loop_d0_egomotion_error_attribution_r3.py `
  create-implementation-lock --repository-root E:\linnan\linnan --output <lock.json>

& $py -I -B scripts/run_dual_loop_d0_egomotion_error_attribution_r3.py `
  validate-implementation --implementation-lock <lock.json>

& $py -I -B scripts/run_dual_loop_d0_egomotion_error_attribution_r3.py `
  activate --repository-root E:\linnan\linnan --implementation-lock <lock.json> `
  --implementation-review <review.json> --output <activation.json>
```

只有 exact lock、独立 review 与 activation 完成后，才允许：

```powershell
$runnerArguments = @(
  'produce', '--activation', '<activation.json>',
  '--implementation-lock', '<lock.json>'
)
pwsh -NoProfile -File scripts/run_guarded_host_research.ps1 `
  -PreflightReceipt <preflight.json> `
  -Script scripts/run_dual_loop_d0_egomotion_error_attribution_r3.py `
  -Python $py -PythonArguments @('-I', '-B') `
  -RunnerArguments $runnerArguments

& $py -I -B scripts/run_dual_loop_d0_egomotion_error_attribution_r3.py `
  validate-execution --run-root <run-r3> --protocol <protocol.json> `
  --dependency-receipt <dependency_receipt.json> --write-results
```

## 输出

Runtime manifest 位于
`artifacts.local/evidence/dual-loop/d0-egomotion-error-attribution-r3/runtime-freeze/`。
实现锁、独立 review 与 activation 位于同一 R3 evidence 根下的
`implementation/`。正式 one-shot 只写 `run-r3/`：

- `formal_start.json`
- `event_table.jsonl`
- `analysis.json`
- `producer_receipt.json`
- `execution_validation.json`
- `execution_receipt.json`
- `progress.json`

marker 后失败只写互斥的 `failure_receipt.json`。R1 `run-r1/` 与 R2
`run-r2/` 均不得修改。

## 安全边界

- R1 与 R2 永久是 `EXECUTION_INVALID / CONSUMED / NO_RERUN`。
- review、activation 与 runner marker 前只验证 control-plane/identity/runtime；
  `validate_scientific_inputs` 必须显式为 `False`，不得打开当前科学输入。
- marker 前不读取真实 calibration 或 bag message；只允许合成 YAML/calibration
  parser 与 rosbags API smoke，并绑定 R2 已消费的 operational probe。
- marker 与初始 progress 持久化后，runner 才能显式以
  `validate_scientific_inputs=True` 完成科学输入校验；随后才可装载 bundle、
  calibration 与 tracks。
- 禁止 `PYTHONPATH`、user site、自动安装、升级或 fallback。
- 禁止旧 F-1B decision、production A/B trace、Confirmation、RCLE 和未冻结输入。
- 只有独立 validator 可以发布 `VALID` 与科学出口。
- 所有结论仅是 burned single-capture operational priority，不是因果、泛化、产品或
  安全证据。

## 停止条件

- runtime、R1 failure binding、输入、源码、仓库或 activation 任一漂移：
  `PRESTART_INVALID / NOT_RUN`，不得创建 `run-r3/`。
- marker 后任一失败：
  `EXECUTION_INVALID / CONSUMED / NO_RERUN / NO_R4 / NO_SCIENTIFIC_EXIT`。
- validator PASS 后只允许：
  `EGO_CANARY_PRIORITY`、`TEMPORAL_TREND_PRIORITY` 或
  `NO_PRIORITY_IDENTIFIED`。
- D0 不自动授权后继 canary、Confirmation、Android、产品或安全工作。
