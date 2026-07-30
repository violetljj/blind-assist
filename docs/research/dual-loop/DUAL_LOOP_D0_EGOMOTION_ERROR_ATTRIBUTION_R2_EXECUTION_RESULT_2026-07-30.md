# D0 ego-motion error attribution R2 执行结果

状态：`EXECUTION_INVALID / CONSUMED / NO_RERUN / NO_SCIENTIFIC_EXIT`

日期：2026-07-30（Asia/Hong_Kong）

## 结论

D0 R2 修复了 R1 缺少 `rosbags` 的运行时问题，并在正式 marker 前通过了冻结
Python、依赖树、首条 Vicon message 反序列化 probe。唯一正式 authority 随后被
消费，但仍没有形成 D0 科学结果。

正式 marker 写入后，producer 已打开 Vicon bag 并读取轨迹；在读取冻结
`calibration.yaml` 时，继承自 R1 的 `read_camera_from_marker()` 动态导入
`yaml`，而 R2 冻结环境只包含 `ruamel.yaml`，因此抛出
`ModuleNotFoundError: No module named 'yaml'`。失败发生在构造任何 event row、
写入 event table 或计算 D0 指标之前；进度保持 `0 / 469`。

按冻结 one-shot 合同，R2 不能补装 PyYAML 后重跑，也不能从内存中已经读取但未
输出的 Vicon 轨迹推断或救援 `EGO_CANARY_PRIORITY`、
`TEMPORAL_TREND_PRIORITY` 或 `NO_PRIORITY_IDENTIFIED`。唯一合法终态仍是执行
无效、authority 已消费、无科学出口。

## 封存证据

- protocol SHA-256：
  `39627821c3da18bd896cae0458294c9d825830435692984f6f0c401211283dfe`
- repository：
  `HEAD == origin/master == a29a776e4e87dd452808aebe7d8ccb73b3c3faf2`
- implementation lock SHA-256：
  `dcb7f088ae531693edcb3f70355d078af3dd68c3a8b64685a073d04e7ec2dae3`
- independent review SHA-256：
  `4c206f3a36757d1a98684e7156559f4ebf95e3c999c2826db1131f7c0d473942`
- activation SHA-256：
  `30dd0b3d7b703316b7059d75a04476c13e039135c631bafa8b29dd2311e56320`
- formal start SHA-256：
  `730ec5dabf4a37716f589c363276e78c114fe26e51a7f656c33bf64aed776f63`
- progress SHA-256：
  `ea272a4712b642f917df271ade3986e6c0e03a2074f664d6b1c6d5246b92b8ef`
- failure receipt SHA-256：
  `057d4c9eb02992da32822cc3a8d18e5a4a5b055e43f8139951368b0115187bb1`
- completed events：`0 / 469`
- prestart probe：`VALID_OPERATIONAL_PROBE`
- `vicon_bag_messages_opened`：`true`
- `d0_metric_computation_pending`：`true`
- `scientific_exit`：`null`
- `rerun_authorized`：`false`

正式目录只包含 `formal_start.json`、`progress.json` 与
`failure_receipt.json`；不存在 `event_table.jsonl`、`analysis.json`、
`producer_receipt.json`、`execution_validation.json` 或
`execution_receipt.json`。

## 根因与后继边界

根因仍是执行环境/动态依赖门不完整，而不是算法、数据、统计出口或低支持负结果。
R2 的 runtime manifest 枚举了顶层模块与 distributions，但 activation 前 smoke
没有枚举 producer 和独立 validator 函数体内的动态 `import yaml`，也没有对真实
冻结 calibration 执行 output-blind parser smoke。

若继续，只能新建独立 D0 R3 合同与 namespace，并同时满足：

1. 精确绑定新的独立 Python environment；除 R2 全部依赖外显式包含
   `PyYAML` 及其模块源身份；
2. activation 前静态枚举所有正式 reachable imports，并在冻结环境中逐项导入；
3. marker 前允许只读解析冻结 calibration，仅验证类型、形状和有限性，不保留数值、
   不读取 truth、不构造 event row、不计算 D0 指标；
4. R3 producer、analysis、469-event 分母、统计规则、路由与互斥出口必须与 R1/R2
   科学合同完全相同；
5. R3 必须同时绑定 R1 与 R2 的 consumed failure terminal，明确不是 R1 或 R2
   rerun，并使用新的 lock、review、activation 和正式 namespace；
6. R3 仍为唯一一次执行；任何 marker 后失败都必须原样封存。

该恢复路线不授权 Confirmation、条件后继 canary、Android、产品或安全结论。
