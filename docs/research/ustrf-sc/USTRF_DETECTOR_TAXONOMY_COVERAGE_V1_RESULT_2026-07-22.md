# USTRF detector taxonomy coverage v1 结果（2026-07-22）

状态：`G0_PASS / G1_EXACT_PARITY_FAIL / G2_PARTIAL_PASS / G3-G5_CLOSED / T0-T3_AND_H2_CLOSED`

## 结论

冻结 tracker/TTC R1 的“4,594 帧阈值内 person 为 0”不是 detector 数据域结论，而是 host raw tensor 布局解码错误。模型真实输出为 `[1,84,2100]`；Android 将其解释为 `84 channels × 2100 predictions`，旧 host 脚本却把 84 行当 prediction、把 prediction 维的前 80 项当类别。按 Android 语义修正后，host 在两来源分别有 `358/558`、`2281/4036` 帧输出阈值内 person；SM-S9280 真机为 `355/558`、`2262/4036`。

这否定了“tracker 之前完全没有 person observation”这一旧首因，但没有授权 T0–T3。Android Canvas 与 host PIL 在完整 4,594 帧上的输入 tensor 精确哈希匹配为 `0/4594`，raw 输出精确匹配也为 `0/4594`；两端 person 是否达到冻结 `.35` 的分歧为 30 帧。G1 因此 fail closed，taxonomy/domain-shift 归因和 detector 候选比较均关闭。H2 时序深度仍未开放。

## 冻结协议与执行

- 预注册：`configs/ustrf_detector_taxonomy_coverage_v1.json`；模型、COCO labels、父窗口、Android `ImagePreprocessor`/`YoloOutputDecoder` 实现、`.35/.45` 和 `[1,320,320,3] -> [1,84,2100]` 均哈希绑定。
- 输入：沿用 tracker/TTC R1 的 15 正/15 同源等长负窗口，共 4,594 个唯一 PNG；不改 frame、route、event、confidence 或 NMS。
- Host：新 Module 严格接受 channels-first 或 predictions-first 中唯一满足 `4 + labels` 的轴；保存每帧 image/input/raw SHA、阈值前 top predictions、全部 class-wise NMS 检测，不覆盖旧 ledger。
- Device：SM-S9280 / Android 16，生产同源 Android Canvas `ImagePreprocessor`、TFLite CPU 4 threads；4,594/4,594 帧完成、0 failure，运行约 344.7 秒。
- 受控 person canary：`dynamics_0/000150`，图像 SHA `5fdad9e…e54fa`；Android/host person 最大分分别为 `.918618/.917334`，均映射到 COCO zero-based index 0=`person` 并高于冻结 `.35`。合成双布局、person/其他类、class-wise NMS 回归通过；因 G1 失败且缺 target bbox，source-box/目标实例覆盖不计通过。

## 可复现结果

| 项目 | dynamics_0 | lt_changes_dynamics_0 | 合计/结论 |
| --- | ---: | ---: | ---: |
| 冻结帧 | 558 | 4036 | 4594 |
| host person 帧 | 358 | 2281 | 2639 |
| Android person 帧 | 355 | 2262 | 2617 |
| host-only / device-only person | 3 / 0 | 23 / 4 | 26 / 4 |
| 两端阈值状态分歧 | 3 | 27 | 30 |
| alertable interval 出现 raw person proposal 的事件 | 3/3 | 12/12 | 15/15 |

`15/15` 只说明事件区间内存在某个 person proposal，不证明它就是冻结目标 person。现有事件 consensus 没有逐帧 person bbox 或全 person 实例账本；同源负窗口也只是 first-fit 非重叠窗口，不是 person-absent truth。因此本轮不统计“person 被误识别成哪些类别”，不输出 detector target recall/FP，也不把背景共现类别称为 taxonomy confusion。

证据 SHA-256：host `dynamics_0` ledger `7a7d8c4…7b43a`；host `lt_changes_dynamics_0` ledger `c4bdeac…b913`；device receipt `17f3a77…ecfa`；重算 summary `3dcdb7f…c4bca`。完整 ignored evidence 位于 `artifacts.local/evidence/ustrf-detector-taxonomy-coverage-v1/`。

## 门与下一动作

- `G0 manifest`：pass。
- `G1 Android-host exact parity`：fail；PIL/Canvas 输入逐像素不等价，raw 输出随之不等价。
- `G2 controlled person canary`：partial pass；tensor shape、COCO index、labels、person raw score与合成 NMS 已闭合，目标 bbox/实例匹配未闭合。
- `G3 taxonomy attribution`、`G4 target coverage/negative FP`、`G5 candidate comparison`：closed。
- T0–T3、TTC 与 H2 D0/D1/D2：closed；App、训练与生产权限均 unchanged。

单一推荐下一动作是先冻结 candidate-hidden 的 person target/negative truth ledger，并让 host 消费 Android Canvas 导出的 canonical input（或复现其逐像素算法）后重跑 G1。只有 G1/G2 都通过，才区分预处理错误、taxonomy/domain shift、低于 `.35` 的 score coverage 与真实 localization miss；baseline 真正失败时才允许最多 3 个预注册候选共享隐藏窗口，禁止调 `.35`、NMS、route、event 或 tracker 回救。
