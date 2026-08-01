# HFTF Stage C D2 机械学执行合同

## 冻结问题

本合同只检验：history-only constant-velocity causal transport 能否比不移动的 current
field persistence 更准确地预测 `+0.4 / +0.8 s` 的 synthetic geometry-proxy signed
clearance。6 个 parent sessions 是独立单位；geometry teacher 不是人类事件或安全真值。

合同在任何 D2 candidate prediction、future truth 或 effect outcome 前冻结。D2.1 对
exact G0 全点 second-order proxy 与 ground-aligned SE2 的澄清具有优先级。

## Future-blind 阶段

preprocessor 只允许读取每个 anchor 的 `t-0.4 s / t` pose slices 与 `t` 的
depth/mask。输出固定为 6 parents × 7 anchors = 42 个 anchor predictions，每个包含
两个 horizons 和两个 arms，共 84 个 anchor-horizon records。

prediction root 必须原先不存在。CLI 在第一次 pose 或 current media read 前写入并
`fsync` 独占 attempt；每个 anchor 的 points 与 prediction 都在读取下一 anchor 前
durable。42/84 全部完成并复核 earlier hashes 后才写 completion。UNKNOWN 只能是 null，
不能变成数值 SAFE。

## One-shot truth 阶段

evaluator 首先离线验证 exact 42 predictions、points、order、hash 与 completion。随后在
第一次 future pose/depth/mask read 前，排他写入并 `fsync`
`truth_join_once_receipt`。receipt 一旦存在，即使后续 crash，也不允许第二次 truth
join。

future truth 使用 future pose/depth/mask 生成 exact G0 geometry proxy，但坐标基仍是
已经 durable 的 causal predicted SE2；future pose 不得改变 candidate origin 或方向。
输出必须精确为 84 个 truth records。

## 冻结门

24 个 `source × height × horizon` strata 全部必须达到 common-known coverage
`>=0.10`、risk `>=5`、safe `>=20`，且 UNKNOWN→SAFE violation 为 0；否则终态
`D2_NOT_EVALUABLE_OPPORTUNITY_INADEQUATE_NO_SOURCE_REPLACEMENT`。

opportunity 足够后，advected arm 必须同时满足：

- six-source macro MAE 相对下降至少 10%，绝对下降至少 0.03 m；
- 每个 height 与 horizon 不劣于 persistence（容差 `1e-12`）；
- 至少 5/6 parents 的 MAE 严格改善；
- parent-macro risk-sign F1 增量至少 0.03；
- UNKNOWN→SAFE violation 为 0。

任一 effect gate 失败即
`CAUSAL_SIGNED_CLEARANCE_TRANSPORT_NOT_SUPPORTED_STOP`；全部通过才是
`CAUSAL_SIGNED_CLEARANCE_TRANSPORT_SUPPORTED_FOR_RGB_STUDENT_PROTOCOL`。

## 执行门与命令

contract、common、preprocessor、evaluator、tests 与两个 mechanics dependencies
必须作为 exact bytes 提交推送。两个 CLI 都会检查所有绑定、test receipt、
tracked/clean 与 `HEAD == origin/master`。

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/preprocess_stage_c_d2_future_blind.py `
  --contract docs/research/hftf/HFTF_STAGE_C_D2_MECHANICS_EXECUTION_CONTRACT_2026-08-02.json `
  --output-root artifacts.local/evidence/hftf/stage-c-d2-future-blind-predictions-20260802

E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/evaluate_stage_c_d2_transport_effect.py `
  --contract docs/research/hftf/HFTF_STAGE_C_D2_MECHANICS_EXECUTION_CONTRACT_2026-08-02.json `
  --output artifacts.local/evidence/hftf/stage-c-d2-transport-effect-result-20260802/result.json
```

preprocessor 失败不得重跑；truth receipt 后中断不得再次 truth join。任何失败均不得
换源、追加或同 cohort 调参。若 Python exception 发生在 durable attempt 之后，
canonical failure artifact 必须以 `fsync` 写出 NOT_EVALUABLE；若进程被外部强杀，
只能依据既有 attempt/truth receipt 封存失败，不得重新读取 current 或 future inputs。

## 权限边界

成功只授权另冻 RGB student protocol，不直接训练 student。reserved official-test、
研究主线、默认 App、Android、生产与 safety 权限全部关闭。
