# HFTF Stage B swept-envelope label mechanics canary D0

日期：2026-08-01

状态：`FROZEN_DEVELOPMENT_CANARY_RESULT_NOT_RUN`

## 1. 问题

在不训练 student、不消费 fresh evidence 的条件下，实际实现原始 HFTF 所要求的
standard synthetic human envelope，判断标签生成器能否产生：

- lateral body-width collision；
- foot/body/head 高度特异风险；
- candidate-direction-specific swept path；
- foot ground continuity、step/drop 与 UNKNOWN；
- 可审计、单调且不把不可见写成 safe 的输出。

本 canary 使用 R2 burned sessions，只能形成 label-mechanics Development evidence。

## 2. Swept path contract

六个 theta bin 的中心是六条候选 path direction。每个 distance bin 是 path 上的
longitudinal segment。对 height layer `z`，障碍点必须同时满足：

- along-track 位于该 distance segment；
- absolute cross-track 不超过 layer effective half-width；
- height 位于对应 `foot/body/head` interval。

standard synthetic envelope 的 effective half-width：

- foot：`0.20 + 0.10 = 0.30 m`
- body：`0.30 + 0.10 = 0.40 m`
- head：`0.18 + 0.10 = 0.28 m`

这些是待检验的标准代理，不是 participant anthropometry 或物理 camera-to-person
calibration。

## 3. Known 与 foot ground

每个 swept prism 使用 segment center 加
`longitudinal × lateral × height` 八角点，共 9 probes；至少 5/9 通过 camera-z、
image、semantic nonzero 与 depth-front `0.20 m` 才可 known。

foot layer 另外沿 centerline 取 5 个 longitudinal ground sections。每 section 至少
3 个 semantic-ground points，至少 4/5 sections supported 才可判 ground-known。
相邻 section 的 median ground height rise `>0.18 m` 或 drop `>0.15 m` 为 foot
ground risk。ground support 不足是 UNKNOWN，不是 safe。

## 4. Structural gates

synthetic fixtures 必须证明：

1. envelope 变宽只会增加、不会减少 collision；
2. head-only point 不污染 foot/body；
3. angular cell 外但身体宽度内的点会被 swept envelope 捕获；
4. 不相交 candidate direction 保持无风险；
5. step/drop 只进入 foot ground component；
6. missing ground support 不能变 safe；
7. obstacle count 为零只有 known 时才可 safe。

随后在四个 burned R2 sources 上报告 known、height disagreement、相对旧
point-support 的 unique collisions、foot ground risk/unknown、dynamic provenance 与
failure atlas。不得把任何比例当作 fresh confirmation 或阈值选择依据。

## 5. 下一权限

只有结构测试全过、四 sources 均可绑定、无 UNKNOWN→SAFE，并出现非退化的 known 与
height-specific outputs，才允许另行冻结 fresh-source formal H1 R3。D0 本身不授权
student、H2、主线、Android、提醒、默认 App、生产或安全。
