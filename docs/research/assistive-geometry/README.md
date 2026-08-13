# BlindAssist Assistive Geometry

状态：`current / R2_F0_SYNTHETIC_REDUCER_PASS / F1_SUPERVISION_FRONTDOOR_SATISFIED / AG_R2_SUPERTEACHER_TO_AG_FINAL_V2_SEAM_PASS / FACTORWISE_ORACLE_HEADROOM_PASS / FRAME_LCB_ZERO_HARM_BUT_PARENT_COVERAGE_FAIL / TUM14_RUNTIME_OBSERVABILITY_FAIL_STOP / SCIENTIFIC_NOT_RUN / CONFIRMATION_OUTCOMES_UNOPENED / DEFAULT_APP_UNCHANGED`

本页是路线日常操作真源。较早完整叙事保存在
[14d8ad7e 历史快照](archive/README_FULL_HISTORY_2026-08-13.md)，不能从中恢复旧权限。

## 当前主张

可替换轻量视觉 encoder 学习 metric-ish depth、support surface 与 obstacle boundary 连续因子；不同
factor 只在各自证据成立时选择性组合，再由确定性 body-swept reducer 构造 Clearance、Occupancy 与
`UNKNOWN`。DepthART-S 是 frozen depth prior/initialization，不是算法终点。

## 当前结论

- R2 F0 reducer、F1 supervision frontdoor 与 SuperTeacher → AG final V2 seam 已落地，但不证明真实
  跨传感器精度。
- consumed Development 已出现可复现的 angular-boundary 正信号；相反，ARKit-trained depth correction
  在 Bonn 有明显 negative transfer。下一步必须按 factor 分路，并让 correction 可 abstain/fallback，不能
  继续追求一个无条件 unified checkpoint。
- ETH3D calibration-control R0/R1 均 fail-closed consumed；R1 independent replay 只证明失败可复现，
  不把它改写为科学 PASS/FAIL。Confirmation outcomes 仍未打开。
- factor-wise oracle 在 11 个 consumed calibration parent 上确认安全 correction headroom；但 learned
  selector 的 Bonn/TUM 外部结果失败。新增的 pixel candidate + neural/kNN frame veto 在两组 consumed
  TUM 诊断上均做到零受伤，却都只覆盖 `1/3` parent，不能以 safe fallback 冒充机制成功。
- 把 14 个 consumed TUM parent 全部降级重分为 11 fit / 3 calibration 后，perfect oracle 仍有
  `39.27%` coverage，但当前 runtime observable vector 的所有 kNN-LCB 候选都只能全回退。这把瓶颈从
  “是否有可纠正区域”收敛为“纠正收益在运行时是否可观测”；默认 App、Android/HTP、产品和安全权限不变。

## 当前证据入口

- [R2 factorized hypothesis](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_FACTORIZED_GEOMETRY_HYPOTHESIS_PROTOCOL_2026-08-09.md)
- [F1 source-native supervision result](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SOURCE_NATIVE_LABEL_MATERIALIZATION_AND_FRONTDOOR_RESULT_2026-08-11.json)
- [SuperTeacher → AG final V2 result](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_SUPERTEACHER_TO_AG_LANDING_RESULT_2026-08-12.json)
- [Calibration-control R0 terminal](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_CALIBRATION_CONTROL_PREFLIGHT_ONE_SHOT_RESULT_2026-08-13.json)
- [R1 terminal](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_CALIBRATION_CONTROL_R1_ONE_SHOT_RESULT_2026-08-13.json)
- [Factor-wise no-regret + runtime observability result](BLINDASSIST_AG_FACTORWISE_NO_REGRET_AND_RUNTIME_OBSERVABILITY_RESULT_2026-08-13.json)
- [Unopened fresh-TUM source lock](BLINDASSIST_AG_FACTORWISE_NO_REGRET_FRESH_TUM_SOURCE_LOCK_R0_2026-08-13.json)
- [算法路线总表](../ALGORITHM_RESEARCH_CURRENT.md) · [研究脚本 Module](../../../scripts/research/assistive_geometry/README.md)

## 唯一 successor

`AG_RUNTIME_CORRECTION_GAIN_OBSERVABILITY_CANARY_R0`，可逆 Development：

1. 不再用相同的 feature mean/std、base/correction/gate/probability 统计重训 selector；先在 consumed TUM
   上增加只在推理时可获得的 temporal reprojection residual 与 model uncertainty observable。
2. 用 leave-one-parent-out 直接预测 correction 对 MAE 与 `>0.10 m` error 的 joint signed advantage；
   source depth 和 task outcome 只能在预测后评分，不能进入 runtime observable。
3. admission 继续要求每个可评价 parent 双指标 no-regret，且至少一半 parent 有非零 coverage；全回退、
   只在一个 parent 开门或只看 macro gain 都不算成功。
4. 只有在 consumed Development 冻结出非零候选后，才可物化已预锁的 fresh3 TUM payload 并执行一次
   parent-disjoint Development；R21 boundary、support validity、`UNKNOWN` 和 reducer 保持不变。

如果新增 observable 仍不能在 leave-one-parent-out 中形成非零 parent-safe gate，就停止 correction router，
保留已独立成立的 boundary 组件；不得为了消费 fresh3 或追求 coverage 放松逐 parent no-regret。

## 当前允许

- 运行上述 consumed Development runtime-observability canary 与逐 parent no-regret gate；
- 维护 fresh3 identity/source lock，但在候选全回退时不读取其 RGB/depth/model outcome；
- 只读复核已签署 evidence、失败链和 frozen negative terminals；
- 重放不访问真实 archive payload 的 focused tests。

## 当前禁止

- 重跑、resume、替换、覆盖或重新解释 R0/R1，或把 null 当 0/negative；
- 读取 ETH3D session/Confirmation、在已看 EVAL 调 threshold，或用 macro gain 掩盖任一 parent 退化；
- 用相同 observable 继续堆 selector/ensemble，或把两组 `1/3` parent coverage 的零伤害诊断写成 PASS；
- 让 depth selector 改写 boundary/support/`UNKNOWN`，或把全 fallback 写成机制成功；
- 把 Development 写成 Confirmation、默认 App、产品或安全证明。

## Claim ceiling

当前只证明 factor/reducer/supervision seam、组件级 angular-boundary 信号、factor-wise correction oracle
headroom 与当前 runtime observable 不足。它不证明 learned correction selector 已成功、跨传感器统一泛化、
完整 task superiority、部署可行性或助行安全。
