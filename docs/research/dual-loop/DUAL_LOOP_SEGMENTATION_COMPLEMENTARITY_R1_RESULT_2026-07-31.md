# YOLO + 语义分割图像空间互补性 R1 结果

状态：`DEVELOPMENT_COMPLETE / BURNED_SINGLE_SOURCE / IMAGE_SPACE_ONLY / NO_EFFECT_AUTHORITY`

日期：2026-07-31（Asia/Hong_Kong）
执行者：`violjjet`
授权：用户在本任务中明确解除此前 D0-B 机制诊断执行门槛

## 1. 研究问题与范围

本轮执行冻结的 `DUAL_LOOP_SEGMENTATION_COMPLEMENTARITY_R1`：在同一 RGB frame 和同一
YOLO box union 上，固定 semantic-segmentation reference 是否产生未被 YOLO 覆盖的
图像空间 class region，以及这些 region 的相邻帧稳定性和 host cost。

本轮不回答区域是否现实不可通行、是否是障碍风险、是否改善事件召回/误提醒/反馈、
是否应进入 Android 或生产。中央阻塞 Agent 标签、risk、feedback、event 字段均未读入。

## 2. 输入身份与角色

| 项目 | 值 |
| --- | --- |
| RGB manifest | `artifacts.local/evidence/dual-loop-r1-unseen-natural-event-r0/rank2-shiraz/input-10hz-r1/manifest.jsonl` |
| RGB manifest SHA256 | `af0ab3c735d96737f451a6e64d1784681966345c7849131ad51bd46c9d7e6571` |
| YOLO trace | `artifacts.local/evidence/dual-loop-r1-unseen-natural-event-r0/rank2-shiraz/device-r1/baseline-output/trace.jsonl` |
| YOLO trace SHA256 | `b9b1b55890e08fd268cb7d650954651a923fc75c9d537bb1e24721deb5753e9b` |
| segmentation reference | `artifacts.local/evidence/segmentation-candidate/sanpo-v3-pretrained-weighted-best-int8-20260713.tflite` |
| segmentation SHA256 | `88f0184d2671230c1f1f43192758689d286b530d7490e1d1ca0671f83b50b50c` |
| source role | 一个已被既有 Development route 消费的 Shiraz session；`burned`，不是 held-out/Confirmation |
| paired input | 4,891/4,891 frame，单一 source session，10 Hz |
| analysis grid | `256 × 256` |

## 3. 执行完整性

- RGB、YOLO trace 和 segmentation 输出按 `source_id + frame_id + image_sha256` 配对；
  timestamp 全部精确匹配，没有插值、最近帧修复或 `NOT_EVALUABLE` 行。
- YOLO 使用 trace 中的全部 object-detector rectangles，不重选 confidence、NMS 或风险阈值。
- segmentation 使用固定 INT8 reference 的四类 argmax mask；输出 finite，未发生单类塌缩。
- 独立 validator 返回 `VALID`，重新核对 4,891 行、时间顺序、四类像素分区、union 算术、
  输入哈希和禁止字段。
- frame artifact 未包含 risk、feedback 或 event 字段；报告声明 `risk_feedback_event_fields_read=false`。

本地产物：

目录：`artifacts.local/evidence/dual-loop-segmentation-complementarity-r1/`

| 产物 | SHA256 |
| --- | --- |
| `report.json` | `af62b5ff727ce807efb7ece6e79691966d6b4837a1c37c931d2496d6993749cf` |
| `frames.jsonl` | `f3b02eaedb421f8bfab84b2ce08fa94d5bded35983463acf3c24cfa7424a4d2b` |
| `validation.json` | `5a93b7ef6b8fee24475ce366acff71ebb2bf12fdcdd40ef177fdd580417f0fef` |

## 4. 图像空间结果

以下均为单一 burned session 的描述统计，不是 frame-independent uncertainty 或跨来源结论。

| class | mask fraction median | YOLO 未覆盖 fraction median | 未覆盖非空比例 | 相邻 IoU median |
| --- | ---: | ---: | ---: | ---: |
| `walkable` | 0.394974 | 0.359863 | 1.000000 | 0.816929 |
| `boundary_step_curb` | 0.004303 | 0.003860 | 0.996524 | 0.080014 |
| `obstacle` | 0.040115 | 0.023270 | 0.993253 | 0.249790 |
| `unknown_nonwalkable` | 0.546310 | 0.431168 | 0.999796 | 0.725020 |

YOLO box union 的平均覆盖率为 `0.122981`。按冻结定义，四类 argmax mask 的 union 天然覆盖
整个分析栅格，因此 `union_increment` 平均为 `0.877019` 只是 `1 - detector coverage`；
它不代表分割发现了障碍，也不作为主要互补结论。

## 5. 主机成本

- segmentation inference：P50 `7.1695 ms`，P95 `13.7080 ms`，MAX `18.6876 ms`；
- 含图像加载/预处理的 host total：P50 `11.0587 ms`，P95 `18.9226 ms`；
- 以上不是手机、Snapdragon、功耗或温度测量。

## 6. 解释与终态

本轮证明了一个更窄的机制事实：固定 reference 在该 source 上确实产生了非零的、未被
YOLO box 覆盖的 class-wise image-space 输出，且输出没有像初始 smoke artifact 那样完全
塌缩。但稳定性不是统一的：`unknown_nonwalkable` 相邻 IoU 中位数约 `0.73`，`obstacle`
约 `0.25`，`boundary_step_curb` 约 `0.08`。因此不能把非零区域直接称作有效障碍、风险、
可通行性增量或事件改善。

本轮终态为：

```text
IMAGE_SPACE_SIGNAL_OBSERVED
STABILITY_MIXED_BY_CLASS
CROSS_SOURCE_REPRODUCIBILITY_NOT_EVALUATED
NO_FUSION_EFFECT_AUTHORITY
```

本轮不进入风险融合，也不进入 Android。要判断互补性是否可跨来源复现，必须另有匹配的
RGB + YOLO reference source；当前这一份 burned session 不足以承担该结论。该结果保留为
Development mechanism evidence、可视化和回归 fixture，不升级为 Confirmation 或生产证据。
