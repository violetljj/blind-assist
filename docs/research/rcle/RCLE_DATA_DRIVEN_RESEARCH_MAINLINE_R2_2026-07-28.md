# RCLE 数据能力驱动研究主线 R2

日期：2026-07-28
状态：`ADOPTED / CURRENT METHOD`

## 决策

RCLE 不再先制定理想数据合同，再寻找完全符合合同的数据。新顺序是：

```text
发现高质量、相关、可获得的数据
→ 低成本连续片段侦察
→ 根据实际能力分配证据角色
→ 从失败模式冻结 Development 问题
→ 调试前预留 session 级 sealed holdout
→ 冻结算法和指标后独立评估
```

该变化不是放宽证据结论，而是把严格度放到真正影响结论完整性的环节。

## 研究问题

当前问题是：

> 在自然相机运动的第一视角视频中，RCLE 和三连续 pair 机制相较未补偿
> expansion 与可用的简单尺度 baseline，呈现什么响应规律、优势和失败模式？

Discovery 可以得出 RCLE 有效、部分有效、无优势，或路线不值得继续。

## 三层隔离

### CAPABILITY_DISCOVERY

- 可以查看内容、信号和曲线；
- 可以改变诊断图和观察维度；
- 不计算或声称独立性能；
- 输出用于形成下一轮可证伪问题。

### DEVELOPMENT_DIAGNOSTIC

- 允许修实现、调参数、选择窗口或指标；
- 相关 observation unit 必须标为 `TUNED_ON`；
- 只能回答实现是否值得进入独立评估。

### SEALED_EVALUATION

- 算法、标签、指标、缺失处理和 split 在输出前冻结；
- 独立单位优先为 person、capture session、route、sequence；
- 同一来源的新 session 可以作为普通 holdout；
- 同一长视频的随机 frame/clip 切分不独立；
- 跨来源问题另立为 `EXTERNAL_TRANSFER`。

frame/pair 是纵向测量，不是独立推断单位。多个 session 到位前只做描述性分布、
曲线、支持率和固定分母触发密度；不得把 600 个相邻 pair 当成 `n=600` 独立样本。
未来分类或效应比较必须按 session/route 分层，报告 session 数、最差 session 和
跨 session 不确定性。

## 四级访问状态

```text
CONTENT_INSPECTED
OUTPUT_INSPECTED
TUNED_ON
SEALED_UNSEEN
```

只看 RGB 内容不自动烧毁 evaluation 资格。真正的污染来自查看算法结果后选择数据、
窗口、标签、指标或参数。历史状态只允许前向降级用途，不回写旧 terminal。

## 最小操作门

Discovery 只保留解码、时间顺序、基本身份、已知许可限制和成本上界。固定长度、
同源正负、精确闭合率、RGB/pose/depth 全模态和单来源全角色均不是默认门。

## 极简能力表

唯一 active 表使用以下 10 列：

```text
dataset_id
sequence_id
scene_motion
available_modalities
observation_unit
access_cost
outcome_access_state
assigned_role
claim_ceiling
notes
```

不得为该表新增通用管理框架、审批状态机、逐来源长报告或重复 adapter。历史 19 列
库存只作为 archive。

## Discovery 观察清单

运行前只冻结：

```text
观察接近、正常行走、转头、横穿、模糊、低纹理和步态振荡下，
各方法的响应分布、支持率、触发密度、时间一致性和失败案例。
```

无预设胜者，无 discovery 级 AUROC/F1 晋级门。

## 旧实验隔离

`RGB_SEGMENT_CONFIRMATION_R1_NOT_EVALUABLE` 永久保持。任何新数据、新提取方式或
新算法运行均必须属于新实验，不能救活、补写或重新包装旧 R1。

## 当前 evidence

ADVIO office03 sequence 15 的起始连续 600 pair 已作为
`CAPABILITY_DISCOVERY / OUTPUT_INSPECTED` 运行。它只支持单 session 响应描述，
不支持性能或泛化。下一步先审计旋转方向、畸变和运动分层，再扩自然 session。

## 停止与升级

- 如果机制审计发现坐标或实现错误：修复后进入新版本 Development；
- 如果 RCLE 在多个自然场景持续无增益：允许停止 RCLE 主路线并保留负结果；
- 如果出现稳定、可复现的场景优势：冻结实现并预留 session 级 evaluation；
- 不因 Discovery 正结果自动进入 Android、产品或安全路径。
