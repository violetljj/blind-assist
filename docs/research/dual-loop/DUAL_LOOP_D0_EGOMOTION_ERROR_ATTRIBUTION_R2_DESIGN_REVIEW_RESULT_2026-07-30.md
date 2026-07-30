# D0 ego-motion error attribution R2 设计复核结果

结论：`DESIGN_PASS / RUNTIME_RECOVERY_ONLY / NOT_RUN`

协议：
[D0 R2](DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R2_PROTOCOL_2026-07-30.json)

协议 SHA-256：
`39627821c3da18bd896cae0458294c9d825830435692984f6f0c401211283dfe`

## 修复范围

R2 只修复 R1 已证实缺失的执行环境绑定，不修改 D0 科学合同。469 个 parent
events、310 个 overlap components、六个 60 秒块、冻结输入、source/ROI/temporal
推导、4 个 preselected + 16 个 diagnostic-only 指标、missingness、Cliff、
person competitor、contradiction 和三个互斥出口均保持不变。

R1 永久保留
`EXECUTION_INVALID / CONSUMED / NO_RERUN / NO_SCIENTIFIC_EXIT`。R2 精确绑定
R1 result、formal start、failure receipt 与 progress 的哈希及语义，并要求 R1
五类科学/验证输出继续不存在。R2 使用新 protocol identity 与 `run-r2`
namespace，不覆盖、不删除、不重跑 `run-r1`。

## Runtime 门禁

冻结 runtime manifest SHA-256 为
`0faceae2077e87a90bc96da1a9e953dd81bd5c4baeec75779b23fd2f783e823a`，
内部 1,291 个有效文件的 tree SHA-256 为
`07227110ca3b91fb2445a13099bfd1a7c2f9df8f231ab77a84ed36113c6ebba4`。
它绑定：

- Python executable SHA-256
  `21bb438c0d4a6f1f164b9a646f6ee000340185e5871180aec06db8d3f07c0082`；
- `-I -B`、禁用 user site、禁止 `PYTHONPATH`；
- `numpy 2.4.2`、`rosbags 0.11.0`、`opencv-python-headless 4.13.0.90`、
  `lz4 4.4.5`、`ruamel.yaml 0.19.1`、`typing_extensions 4.16.0`、
  `zstandard 0.25.0` 的精确集合、来源和文件树；
- 禁止 formal 执行时自动安装、升级或 fallback。

marker 前只允许反序列化一条预指定的 sensor Vicon message：
`/vicon/event_lidar/event_lidar`、ordinal 0、timestamp
`1708490365692128652`、117 bytes、raw SHA-256
`55779b8473c8813aff6827669f42b97e230715aa88fbb80781b1454a1cea920b`。
只检查连接、类型、哈希与 7 个 transform 分量有限，不保存 pose、不扫描下一条
message、不计算 D0 指标。相同锁定进程随后才可排他创建 formal marker。

## 独立复核

统计与执行信封两路独立只读复核均为 `DESIGN_PASS`：

- R1 consumed failure 与 required-absent 集合 hash/semantic exact；
- R1 科学章节除追加 runtime-recovery claim ceiling 外保持 exact；
- runtime manifest、解释器、dependency tree、import provenance 与 probe 身份闭合；
- prestart 失败不创建 `run-r2`，marker 后失败为 consumed/no-rerun；
- 仅保留 `EGO_CANARY_PRIORITY`、`TEMPORAL_TREND_PRIORITY`、
  `NO_PRIORITY_IDENTIFIED` 三个科学出口；
- R2 不授权 successor、Confirmation、Android、产品或安全工作。

当前仅完成合同冻结与设计复核；implementation lock、activation 和正式 R2 均未运行。
