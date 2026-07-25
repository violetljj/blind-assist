# Independent person trajectory truth source authority and admission R0

状态：`FROZEN_BEFORE_CANDIDATE_OUTPUT_READ`

## 唯一研究问题

是否存在一个不依赖 JRDB 3D box、PCD point-in-box 或其质心输出的来源，能提供稳定 person track ID、米制 3D 位置、时间绑定、坐标语义、误差/标定说明，并在候选算法结果不可见时冻结可复算的实际距离分母？

## 冻结来源与选择

- JRDB annotation-derived geometry 因循环论证直接拒绝为真值。
- 已有 REveL Vicon 只保留为有限候选：独立 person/sensor marker 存在，但现有收据没有量化 Vicon 误差说明，且最大距离仅约 `10.41m`。
- 本轮 bounded canary 选择 THÖR people tracks v1。只按官方 Zenodo metadata，在五个 moving-robot `Exp_2_*_6D.tsv` 中选择声明字节数最小的 `Exp_2_run_2_6D.tsv`；冻结前不读取 payload 值。
- 窗口是整个文件；轨迹是文件中全部 `Helmet_2..Helmet_10` 与 moving-robot 刚体，冻结 payload header 将后者命名为 `Citi_1`。pre-payload placeholder `Robot_1` 只按 source schema 更正为 `Citi_1`；source、member、整文件窗口、person tracks、距离带、分母门和缺失策略均未改变，候选算法输出仍未读取。不得看结果后换 run、换轨迹或截窗。

## 距离与分母

同一 QTM 行内，以共享 Qualisys world frame 中 helmet rigid-body translation 到 `Citi_1` translation 的 3D 欧氏距离分带；只有 source unit 获得权威闭合时才可换算为米：

`0–5 / 5–10 / 10–20 / 20–40 / 40m+`

`5–20m` 是产品重点；每个 `5–10` 和 `10–20` band 预注册至少 `1,000` valid object-frame 且至少 `2` 个 person track ID。`40m+` 永久保留为能力边界；空分母不得写成通过，但不会单独否定来源的近中距 authority，必须在 authority scope 中明确关闭。

缺失 person 或 robot translation 只排除该 person-frame 并计入 missing；禁止插值、改 ID、换窗或人工挑轨迹。重复 frame、时间非单调、单位未知或 robot rigid body 缺失均 fail closed。

## 准入与权限

准入必须同时满足独立测量链、稳定 ID、metric 3D、共享时间、坐标语义、量化误差/标定、产品重点分母和 candidate-blind freeze。失败终止为 `INDEPENDENT_PERSON_TRAJECTORY_TRUTH_AUTHORITY_ABSENT`。

即使通过，也只准入 source-native helmet-to-robot offline metric trajectory truth。sensor-to-mocap 空间外参、helmet-to-body-center 变换、route/event、算法比较/选择、Android、人体、独立行走和生产权限全部关闭。
