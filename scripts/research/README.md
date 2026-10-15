# 研究辅助模块

这里保留被研究使用的辅助实现及其测试。当前路线、协议与结果由
[研究决定](../../docs/CURRENT_DECISION.md)和各路线 `CURRENT.md` 管理；
完整路线实现位于 [research/active](../../research/active/)。

| 模块 | 职责与入口 | 对应记录 |
| --- | --- | --- |
| [nearfield_opportunities](nearfield_opportunities/README.md) | 形状假设、主动观测、路径约束、共享偏差和边界可观测性的合成诊断；实现、runner、测试同目录 | 模块 README 链接各自协议和结果，保留原运行的适用范围 |
| [skysynth_audit_pilot.py](skysynth_audit_pilot.py) | 一次有界的 Auditor 提示对照；[test_skysynth_audit_pilot.py](test_skysynth_audit_pilot.py) 测试其评估辅助函数 | [试验记录](../../research/active/dtr-r0/nearfield/SKYSYNTH_AUDIT_PILOT_20260922.md) |
| [validate_skysynth_audit_tests.py](validate_skysynth_audit_tests.py) | 将已提交探针交给 SkySynth 原有测试验证器重放 | 同一试验记录；依赖已准备的实验产物与本机 SkyDiscover 环境 |

这些模块是研究诊断与复现代码；Android 运行时代码见
[代码地图](../../docs/CODE_MAP.md)。执行命令、依赖和输出位置按对应模块 README
或原试验记录选择。普通代码测试与重新运行已完成的科学试验分开处理。

项目维护和构建入口见 [scripts/README.md](../README.md)，
资产、后端和研究查询工具见 [tools/README.md](../../tools/README.md)。
