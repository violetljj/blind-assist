# TUM fr2/rpy source-native geometry audit

状态：`single-source discovery`

本模块只读取官方 TUM `rgbd_dataset_freiburg2_rpy` 的 RGB、注册深度和
mocap color-camera pose。窗口、关联、覆盖与诊断带由
`RCLE_TUM_FR2_RPY_SOURCE_NATIVE_GEOMETRY_AUDIT_R0_CONTRACT_2026-07-26.json`
在下载目标 TGZ 和查看几何结果前冻结。

几何计算直接导入 `pb_h1_role_proxy/geometry.py`，不复制或修改 PB-H1
公式。输出保留全部 pair rows、连续分布、coverage 和弃权原因。这里不读取或
运行 RCLE RGB algorithm，不允许换 TUM sequence 回救，也不形成 confirmation、
安全或产品权限。
