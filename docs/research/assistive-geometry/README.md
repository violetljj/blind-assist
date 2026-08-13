# BlindAssist Assistive Geometry

状态：`current / R2_F0_SYNTHETIC_REDUCER_PASS / F1_SUPERVISION_FRONTDOOR_SATISFIED / AG_R2_SUPERTEACHER_TO_AG_FINAL_V2_SEAM_PASS / AG_R2_CROSS_SENSOR_CALIBRATION_CONTROL_R0_AND_R1_FAIL_CLOSED_CONSUMED / R1_INDEPENDENT_REPLAY_CONFIRMED_PRODUCER_FAILURE / FACTORWISE_NO_REGRET_R0_ACTIVE / SCIENTIFIC_NOT_RUN / CONFIRMATION_OUTCOMES_UNOPENED / DEFAULT_APP_UNCHANGED`

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
- 当前新增机制是 factor-wise no-regret selector；默认 App、Android/HTP、产品和安全权限不变。

## 当前证据入口

- [R2 factorized hypothesis](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_FACTORIZED_GEOMETRY_HYPOTHESIS_PROTOCOL_2026-08-09.md)
- [F1 source-native supervision result](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SOURCE_NATIVE_LABEL_MATERIALIZATION_AND_FRONTDOOR_RESULT_2026-08-11.json)
- [SuperTeacher → AG final V2 result](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_SUPERTEACHER_TO_AG_LANDING_RESULT_2026-08-12.json)
- [Calibration-control R0 terminal](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_CALIBRATION_CONTROL_PREFLIGHT_ONE_SHOT_RESULT_2026-08-13.json)
- [R1 terminal](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_CALIBRATION_CONTROL_R1_ONE_SHOT_RESULT_2026-08-13.json)
- [算法路线总表](../ALGORITHM_RESEARCH_CURRENT.md) · [研究脚本 Module](../../../scripts/research/assistive_geometry/README.md)

## 唯一 successor

`AG_FACTORWISE_NO_REGRET_ORACLE_AND_PARENT_GATE_CANARY_R0`，可逆 Development：

1. 冻结 DepthART depth prior、现有 correction expert、R21 angular boundary 及 support/`UNKNOWN`
   validity；不调用 reducer，不读取 ETH3D protected outcome。
2. 在已披露的 `PROJECT_CONSUMED_DEVELOPMENT` 输出上比较 prior、expert、perfect signed-advantage
   oracle 与现有 selector。oracle 仅用于判断是否存在安全 correction coverage，不是部署策略。
3. threshold admission 要求每个可评价 parent 的 depth MAE 与 `>0.10 m` error 均不差于 prior，且至少
   一半 parent 有非零 correction coverage；只看 macro gain 不准入，零 coverage 只算 safe fallback。
4. 固定重放 R21 boundary；depth route 不得改写 boundary、support validity 或 `UNKNOWN`。

oracle 没有有意义的安全 coverage 就停止 depth router、保留 boundary 组件；oracle 有 headroom 而 selector
失败时，才允许下一版本训练 one-sided signed-advantage lower-confidence-bound router。不得在已看 EVAL
继续调 threshold。

## 当前允许

- 运行上述 consumed Development oracle/selector canary 与逐 parent no-regret gate；
- 只读复核已签署 evidence、失败链和 frozen negative terminals；
- 重放不访问真实 archive payload 的 focused tests。

## 当前禁止

- 重跑、resume、替换、覆盖或重新解释 R0/R1，或把 null 当 0/negative；
- 读取 ETH3D session/Confirmation、在已看 EVAL 调 threshold，或用 macro gain 掩盖任一 parent 退化；
- 让 depth selector 改写 boundary/support/`UNKNOWN`，或把全 fallback 写成机制成功；
- 把 Development 写成 Confirmation、默认 App、产品或安全证明。

## Claim ceiling

当前只证明 factor/reducer/supervision seam、组件级 angular-boundary 信号与 calibration-control failure
可复现；factor-wise selector 只是 active Development。它不证明跨传感器统一泛化、完整 task superiority、
部署可行性或助行安全。
