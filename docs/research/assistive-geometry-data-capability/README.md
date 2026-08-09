# Assistive Geometry Data Capability Atlas

状态：`current / AG-DCA_R0_COMPLETE / THREE_HYPOTHESES_NOT_SUPPORTED / NO_ALGORITHM_AUTHORITY`

AG-DCA 是 Assistive Geometry 的数据能力基础设施，不是第三条算法支线。它回答：

> 冻结的现有数据是否真的观测了某个数学对象，并能否形成 parent-disjoint 的最低支持？

R0 已全量读取 B1 TRAIN `16 parent × 300 = 4,800 frames` 的 target receipts，只统计
truth/source capability，不读模型 outcome。atlas 同时区分 frame、parent、portrait/landscape、
joint parent-disjoint 与非数据 authority，避免 frame 数量冒充独立证据。

## R0 终态

`AG_DCA_R0_COMPLETE_THREE_HYPOTHESES_NOT_SUPPORTED`

全量 atlas 与机器判定已完成，见 [governed result](BLINDASSIST_ASSISTIVE_GEOMETRY_DCA_R0_RESULT_2026-08-10.md)
和 [machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_DCA_R0_RESULT_2026-08-10.json)。QSF H1 reopen 与
CBF R0-style grid 均为 `NOT_SUPPORTED_DATA`；FCI-for-R2-decision 为
`NOT_SUPPORTED_DATA_AND_AUTHORITY`，所以 AG-FCI 未创建、未启动。R0 无活动 successor；未来新
hypothesis 必须提交新的版本化 requirements 后重放不可变 atlas。

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
通过 DCA checker 被重新包装成 R2 选择证据。R0 实测 complete factor schema 与 clear bundle 均为
`0`，joint parent 为 `0`，oracle injection 与 fresh paired outcome 也未冻结。

## Claim ceiling

AG-DCA 只建立数据/权限准入事实。它不定位 A0 的因果罪因，不执行 intervention，不选择 R2 factor，
不授权训练、Development、Confirmation、Android/HTP、默认 App、产品或助盲安全主张。
