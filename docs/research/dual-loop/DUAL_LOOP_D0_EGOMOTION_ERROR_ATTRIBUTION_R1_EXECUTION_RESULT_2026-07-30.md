# D0 ego-motion error attribution R1 执行结果

状态：`EXECUTION_INVALID / CONSUMED / NO_RERUN / NO_SCIENTIFIC_EXIT`

日期：2026-07-30（Asia/Hong_Kong）

## 结论

D0 R1 的唯一正式 authority 已被消费，但没有形成任何 D0 科学结果。锁、独立
implementation review 与 activation 均通过后，runner 排他创建
`formal_start.json`；首次准备解码 REveL Vicon bag 时，绑定的 Python 运行环境因
缺少 `rosbags` 抛出 `ModuleNotFoundError`。失败发生在读取任何 bag message、
构造 event table 或计算归因指标之前。

按冻结 one-shot 合同，本次不能以补环境变量的方式重跑 R1，也不能从已有 JSON
推断或救援 `EGO_CANARY_PRIORITY`、`TEMPORAL_TREND_PRIORITY` 或
`NO_PRIORITY_IDENTIFIED`。唯一合法终态是执行无效、authority 已消费、无科学出口。

## 封存证据

- protocol SHA-256：
  `87931369f912fdd054783db9decb2a1813080d0a961c3526b83ce686d1a48183`
- repository：
  `HEAD == origin/master == 1ed4e9b59f67d3729a842eb62d7350d4f2b59881`
- implementation lock SHA-256：
  `75bbaab0afdc77570a54d2f48990d7072b90c6be957203d26e0a9755856820b3`
- independent review SHA-256：
  `f148a91a12e1c957c87643eb32981230d06004ea8b6c9a399454965f8d60ee56`
- activation SHA-256：
  `b20fb8c5b62719390b26d94062a97497de36ecc5e82d251639ccf8504920de31`
- formal start SHA-256：
  `fca9f6db4fafc58e9fbe84e75af9399194cdb145741f5cd4dd012681fd9a3420`
- failure receipt SHA-256：
  `f51ea6ade52d01242b75640b615ba642f9875119114cf60b2a9ae00e2c34e62d`
- progress SHA-256：
  `185cca546660805c1f80ed212a79a3243007b1ea5c7fdcc9e86ba58559e76056`
- completed events：`0 / 469`
- `vicon_bag_messages_opened`：`false`
- `scientific_exit`：`null`
- `rerun_authorized`：`false`

正式目录只包含 `formal_start.json`、`progress.json` 与
`failure_receipt.json`；不存在 `event_table.jsonl`、`analysis.json`、
`producer_receipt.json`、`execution_validation.json` 或
`execution_receipt.json`。

## 根因与后继边界

根因是执行环境绑定不完整，而不是算法、数据或统计出口的负结果。仓库已有忽略目录
`artifacts.local/work/python-deps/rosbags-cpu-20260720`，其中 `rosbags==0.11.3`
可成功导入，但 R1 的 lock/activation 没有绑定该依赖树，marker 前预检也只哈希
bag，没有执行真实 Reader/typestore/message deserialize 探针。

若继续，只能新建独立 D0 R2 合同与 namespace，并同时满足：

1. 精确绑定 Python executable、`rosbags` 版本和依赖树身份；
2. 在正式 marker 前只读打开 bag，并反序列化协议指定的首条 Vicon message；
3. preflight 不聚合、不生成 source component、不构造 event table、不计算指标；
4. R2 算法、冻结输入、469-event 分母、统计规则和互斥出口与 R1 完全相同；
5. R2 必须绑定本 R1 failure terminal，且明确不构成 R1 rerun。

该修复路线仍不授权 Confirmation、后继 canary、Android、产品或安全结论。
