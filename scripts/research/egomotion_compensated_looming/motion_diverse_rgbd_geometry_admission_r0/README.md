# Motion-diverse RGB-D geometry admission R0

状态：`NOT_EVALUABLE_NO_RGB_NO_REPLACEMENT / VALID`

## 稳定 Interface

从仓库根目录通过 `scripts/run_research_tool.py
egomotion-compensated-looming` 调用
`run_motion_diverse_burned_fixture_smoke_r0.py` 或
`transport_motion_diverse_geometry_component_r0.py`。正式 geometry 与独立聚合
验证分别通过 `run_motion_diverse_eth3d_geometry_r0.py` 和
`validate_motion_diverse_eth3d_geometry_r0.py` 进入；外部合同不得绑定本目录
中的实现路径。

## 输出

burned fixture smoke 和后续 exact-component transport receipt 只写入
`artifacts.local/`。正式模板固定 Decimal 数值归一化、`rel_tol=1e-12`、
`abs_tol=1e-15`、有限值、10 秒窗、`2 positive + 2 below-reference` 与
默认 8 workers。R0 正式结果为 7 个窗全部
`AMBIGUOUS_OR_INELIGIBLE`，独立验证通过。

## 安全边界

本模块不读取 RGB、不运行 RGB 算法、不修改已消费的 Floor3 证据，不授权
`floor3_3`、候选替换、事后补窗、性能、Android、人体、安全或产品结论。
trajectory 只能用于 metadata 排序，不能授予几何角色。

## 停止条件

rank-one 候选缺少官方身份、精确 bytes、checksum、许可、同步
depth/pose/timestamp，或 geometry-only 无法冻结 2+2 四窗时，立即
`NOT_EVALUABLE`，不得取得 RGB 大载荷。
