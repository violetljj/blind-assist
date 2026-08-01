# HFTF Stage B swept-envelope reference comparison source preparation R3

日期：2026-08-01

状态：`FROZEN_SOURCE_PREPARATION_ONLY_R3_OUTCOME_NOT_AUTHORIZED`

## 1. Formal question

在全新 SANPO-Synthetic sessions 上，swept human-envelope candidate 是否相对使用
相同 stride-8 points 的 angular point-support baseline，更准确地复现不相交
stride-4 dense swept geometry proxy reference？

这是 Stage B current-field obstacle/ground teacher 比较，不计算 future horizon，不训练
student。

## 2. Outcome-blind source selection

官方 train split generation 为 `1692794964120907`，文本 SHA-256 为
`f9c5dc4c289fa87342abc0d2cc49f112fcc78c7e02e0b6b081e296a99344173c`。

排除 R0/R1/R2 共 12 个 burned sessions 后，从完整 session ID 字典序开头扫描。前 19
个未烧毁 sessions 因没有 `camera_chest` 在只读 inventory preflight 中拒绝；没有读取
teacher field outcome。前四个具备 chest-left 且从 frame 0 可确定性取得 25 个
RGB/mask/depth 对齐帧的 sessions 固定为：

1. `043db91a506708587fce87dd9bd5f96c8d4480ba258b01927a58848907e266e5`
   （5 FPS，frames 0–24）；
2. `0460c41f987056b7a3b643d1fcd0554ef1fa4a2179c5b216c1b2c81ce04deb1b`
   （5 FPS，frames 0–24）；
3. `047a3307f975af5e9c653ce7666178b421e5ff2d933853a738bd7c78d13c25a5`
   （20→10 FPS，frames 0–48 step 2）；
4. `04bfa5b7f31e6b5ca7fb22556eb41600167053b0db57c81546d568854accc498`
   （33→10 FPS，deterministic resample frames 0–79）。

若下载后 source authority 不通过，只能记录失败并按同一字典序/eligibility 规则顺延，
不能根据 candidate/baseline 表现换 session。

## 3. Frozen gates

primary reference count threshold 为 `2`。4/4 source 与 reference readiness 过门后：

- cohort micro-F1 delta `>= +.10`；
- cohort precision delta `>= +.10`；
- cohort recall delta `>= -.02`；
- 4/4 session micro-F1 delta 各 `>= +.05`；
- foot/body/head candidate F1 均高于 baseline；
- thresholds `1/2/4/8` 的 cohort candidate F1 均高于 baseline；
- 四个 thresholds 的 paired candidate-only-correct 均多于
  baseline-only-correct；
- 每 session 每 height obstacle-known coverage `>= .10`。

任何 obstacle comparison gate 失败即终止为
`R3_SWEPT_ENVELOPE_REFERENCE_GAIN_NOT_SUPPORTED_STOP`，不进入 future。

## 4. Ground component

ground-known/risk/UNKNOWN 与 obstacle confusion 分开。每 session ground-known coverage
至少 `.10`；full Stage B terminal 还要求 fresh cohort 至少出现一个 reference ground
risk opportunity。

若 obstacle gates 全过但没有 ground risk opportunity，终态只能是
`R3_OBSTACLE_ENVELOPE_GAIN_SUPPORTED_GROUND_NOT_EVALUABLE`。这保留 obstacle 表示的
正结果，但不把 synthetic step/drop fixture 冒充真实 source 支持，也不授权 Stage C。

## 5. 当前权限

本合同只授权下载四个精确 sessions 并运行 frozen-canonical source authority。完整
authority、manifest、spec、pose 与实现 hashes 绑定到正式 R3 protocol 前，不得计算
candidate/baseline/reference outcome。

future Stage C、student/H2、研究主线、Android、提醒、默认 App、生产与安全均未授权。
