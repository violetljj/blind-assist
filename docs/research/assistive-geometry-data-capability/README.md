# Assistive Geometry Data Capability Atlas

状态：`current / AG-DCA_R0_PROTOCOL_LOCKED_NOT_RUN / TRAIN_CAPABILITY_ATLAS_NOT_RUN / NO_ALGORITHM_AUTHORITY`

AG-DCA 是 Assistive Geometry 的数据能力基础设施，不是第三条算法支线。它回答：

> 冻结的现有数据是否真的观测了某个数学对象，并能否形成 parent-disjoint 的最低支持？

R0 将全量读取 B1 TRAIN `16 parent × 300 = 4,800 frames` 的 target receipts，只统计
truth/source capability，不读模型 outcome。atlas 同时区分 frame、parent、portrait/landscape、
joint parent-disjoint 与非数据 authority，避免 frame 数量冒充独立证据。

## 唯一 successor

`BLINDASSIST_ASSISTIVE_GEOMETRY_DCA_R0_FULL_TRAIN_ATLAS_EXECUTION`

已由 [machine protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_DCA_R0_PROTOCOL_2026-08-10.json) 冻结
input/implementation/requirements/authority facts；下一步运行一次全量 atlas。首批 checker
合同是 QSF H1 reopen、CBF R0-style grid 与 FCI-for-R2-decision。任何
`SUPPORTED_FOR_PROTOCOL_LOCK` 只允许另立协议；`NOT_SUPPORTED_*` 必须 fail-close，不能自动缩门。

## Capability 口径

R0 包含 clearance event/right-censor、ground plane、forward ground `0–2/0–5 m`、lateral
observation `±0.5/±1/±2 m`、full 2.5D grid、三档 occupancy、temporal pair、camera geometry、
truth-clear/truth-occupied，以及 depth/ground/support/obstacle factor bundle。完整 pose transform
没有物化到当前 target，必须诚实登记为 unsupported，不能从 frame index 猜测。R2 所需的
depth/support uncertainty 与连续 obstacle-boundary truth 也单列，不能用 crisp 派生值或零常数补齐。

## FCI 边界

AG-FCI 若要为 R2 资源分配提供选择证据，必须同时具备：joint clear/occupied factor truth、至少
8 个共同 parent、冻结的 reducer 和 oracle-injection interface，以及 fresh selection-eligible paired
outcome。B1 consumed Development 即使可做 post-hoc diagnostic，也不能满足 fresh authority，不能
通过 DCA checker 被重新包装成 R2 选择证据。

## Claim ceiling

AG-DCA 只建立数据/权限准入事实。它不定位 A0 的因果罪因，不执行 intervention，不选择 R2 factor，
不授权训练、Development、Confirmation、Android/HTP、默认 App、产品或助盲安全主张。
