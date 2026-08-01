# HFTF Stage C causal future-label mechanics D1

日期：2026-08-01

状态：`FROZEN_AFTER_CONSUMED_CALIBRATION_BEFORE_FORMAL_D1_REPORT`

## 1. 研究问题

D0 只证明 current depth 可产生 semantic-independent label proxy。D1 检验：
future depth observation 能否在不让真实 future path 选择 origin/方向的前提下，为
`.4/.8 s` anchor-grid teacher label 增加 known support。

这仍是 consumed Development mechanics，不训练 student。

## 2. Phone-causal origin

formal anchor 为 `5,10,15,...` 且必须有 `anchor+4`。history velocity 只用
`anchor-2 -> anchor` 的 `.4 s` pose；所有 horizon orientation 固定为 current yaw：

`origin(tau) = position(anchor) + history_velocity * tau`

真实 future pose 只允许把 future depth-derived section 变换到 world，禁止决定
origin、grid orientation 或输出 direction。速度超过 `3 m/s` 的 anchor 全 horizon
UNKNOWN。

## 3. Observation 与 baseline

- `0 s`：current depth profile；
- `.4 s`：current + frame `+2` profiles；
- `.8 s`：current + frame `+4` profiles。

每个已知 horizontal-support section 经 source pose 变换到 world，再相对 causal origin
投到固定 `5 directions x 5 distances`；方向误差最多 `7.5°`、距离误差最多 `.2 m`，
多 observation height 取 median。

future baseline 只重投影 current observation；candidate 再 union 对应 future
observation，禁止删除 baseline known cell。缺失和 unmatched 永远 UNKNOWN。

## 4. 校准披露与冻结门

同一 consumed sources 上，`.4/.8 s` motion-yaw circular resultant 为
outdoor `.899/.840`、indoor `.969/.962`，中位绝对偏差 `4.5–7.0°`。

candidate 相对 current-only 新增 known cells：

| source | `.4 s` | `.8 s` |
| --- | ---: | ---: |
| outdoor | 186 | 280 |
| indoor | 303 | 490 |

formal 要求每 source/horizon motion-yaw resultant `>=.8`、误差 `<=15°`、candidate
known fraction `>=.70`、新增 known `>=100`、known loss 0；cohort future risk proxy
至少 2 cells/2 frames/2 directions，且七个 causality/UNKNOWN/determinism canaries
全过。

## 5. 权限

D1 成功也只允许冻结 fresh session-disjoint teacher corpus + student canary protocol；
不授权 acquisition、corpus generation、training/effect、主线、Android/App 或安全
claim。

机器可读真源：
[HFTF_STAGE_C_CAUSAL_FUTURE_LABEL_MECHANICS_D1_2026-08-01.json](HFTF_STAGE_C_CAUSAL_FUTURE_LABEL_MECHANICS_D1_2026-08-01.json)
