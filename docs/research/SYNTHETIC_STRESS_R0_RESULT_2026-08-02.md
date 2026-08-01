# Controlled synthetic stress R0 result

状态：`COMPLETE / VALID / MIXED_DEVELOPMENT_DIAGNOSTIC / NO_PROMOTION_AUTHORITY`

## 结论

4,448 个受控 synthetic case 已完成，独立 validator `VALID`，boundary canary
`32/32 PASS`。这证明 runner、hash ledger、UNKNOWN firewall、坐标方向和边界合同
可复现；不证明 RCLE 或 D2 在所有退化下稳健，也不授权新的 official media、student、
Android、默认 App、production 或 safety claim。

本轮最重要的负信号是：rotation leakage 存在 stress 长尾，RCLE coverage 在最坏行降到
`1/9`；front-approach 的全场 expansion sign 只有 `39.95%`（`433` evaluable），所以
不能把混合场的全局 expansion 直接当成接近语义。scale sign 单独为 `99.38%`
（`325` evaluable）。D2 causal advection 在本 synthetic field 上不是普遍增益：全部
evaluable strata 的 advected-better fraction 为 `33.78%`，按 D2 exact timebase 的
`234` strata 为 `32.91%`。

## 关键分层结果

| 项目 | 结果 | 解释 |
|---|---:|---|
| case 总数 | `4,448` | 704 motion catalog、2,592 one-factor、576 pairwise、576 stress-pack |
| RCLE evaluable | `3,312 / 4,448` | `1,136` 保留为 `NOT_EVALUABLE_INSUFFICIENT_TRACK_SUPPORT` |
| RCLE evaluable coverage | median `0.7778`，minimum `0.1111` | 不能用 pooled coverage 掩盖最坏 grid |
| rotation leakage raw → compensated median | `0.03353 → 0.01513 s^-1` | 只说明总体抑制倾向，不是全 stress gate |
| yaw / pitch / roll compensated median | `0.01320 / 0.01756 / 0.01463 s^-1` | 对应 max 为 `0.83563 / 0.71542 / 0.39641 s^-1` |
| rotation leakage reduced fraction | `77.55%` | roll 仅 `54.52%`；高 blur/低 texture/depth-gap 长尾必须保留 |
| TTC proxy evaluable | `1,544 / 4,448` | `2,904` non-closing；proxy 明确不是 physical TTC |
| field transport strata | `8,896` | 每 case 两个 horizon；全部有 common-known cell |
| field advected-better fraction | `33.78%` | median improvement 为 `0`，因为正负效果会抵消，需按 motion/条件分层 |
| D2 exact-time eligible | `117 / 4,448` cases，`234` strata | 其他为 `4,016` unsupported FPS、`234` timebase、`81` missing required frame |
| UNKNOWN→SAFE | `0` | numeric mutation 也被独立测试拒绝 |

### 运动/条件解释

- scale 正负的 RCLE sign 是当前最稳定的 synthetic mechanism cell；front approach
  全场 sign 不稳定，保留为混合场反例，不进入“接近可检测”结论。
- rotation compensation 在 clean/轻退化下有抑制倾向，但 yaw/pitch 的 P90 与三轴
  max 明显高于 median；应优先按 axis × blur/texture/depth-discontinuity 分层，而不
  继续扩大 pooled aggregate。
- field transport 在 rotation+translation、yaw、前向平移的一部分 cell 有改善，
  在 pitch/roll/vertical 或不匹配条件下可为零或退化；因此当前不支持“D2 causal
  advection 普遍优于 persistence”。
- 5/20 FPS 且 exact normalized timeline 的 D2 case 才有资格进入 D2 mechanics
  观察；10/15/30/60 FPS 不是自动补偿成 D2 evidence，jitter/缺帧也不能被零填。

## 证据与 hash

最终输出目录：

`artifacts.local/evidence/controlled_synthetic_stress_r0/20260802-run4/`

- protocol SHA-256：`1df629cd829a7bdd6a13a0593bf6d2574bbe88413ae28a25049a956b63f7cd2a`
- case manifest SHA-256：`e27fff9ba403ec7de063eeb5f9cec156f122aec8c1b1b522b546765d635a10de`
- case results SHA-256：`8ce213727f1eddd66657fb70dc5345d21342e029b1b89ce39e5481ff0ea6419e`
- boundary results SHA-256：`2e45c48941f7c56474c034dd242edf8bb4ec6707273e1ebaf42e77e0fbcaac8b`
- independent validation SHA-256：`C802B953926DB02A8A4EE16125C16E98A80F87172BFC8938D56E4BA2B9C424A6`

验证命令：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/synthetic_stress/validate_stress_r0.py `
  --output artifacts.local\evidence\controlled_synthetic_stress_r0\20260802-run4 `
  --receipt artifacts.local\evidence\controlled_synthetic_stress_r0\20260802-run4\independent_validation.json
```

## 行动边界

当前结论是 `HOLD_FOR_NEW_VERSIONED_MINIMAL_DISCRIMINATING_TEST`：不以当前 synthetic
结果替换 RCLE/D2 正式协议，不打开 fresh/official media，不调阈值救援同一结果，也不
把 TTC proxy 或 field teacher 当作安全 truth。若继续，只应另冻 axis×condition 的
最坏-cell 复核，明确 oracle/observed information representation 和每个
`NOT_EVALUABLE` terminal。
