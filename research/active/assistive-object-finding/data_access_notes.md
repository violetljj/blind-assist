# 寻物小规模数据可用性验证（Development，2026-10-15）

本记录只涉及数据取得、来源和规范化，不包含模型效果或题目成立结论。
数据、原始标签、下载包、哈希和预览均在
`artifacts.local/work/assistive-object-finding-pilot-20261015/data/`；
逻辑入口经过 canonical junction 落到 F:，未建立 E: 数据旁路。

## 实际取得的数据

- **RefEgo**：已取官方修复缩放问题的 `RefEgoAnnotations2.zip`，19,248,412 B；
  本地保留为 `RefEgoAnnotations2_ranged.zip`。三套公开注释均解析成功。
  未凭公开注释假定已经获得 Ego4D 视频使用许可。
- **MeViSv2 valid_u**：已取官方 FudanCVL Hugging Face 上的图像 tar、
  `meta_expressions_v2.json` 和 `mask_dict.json`，有全部帧的官方目标掩码。
  已实际物化 **12 个源视频、16 个 query 序列、528 个标注 query-frame**，
  对应 **456 张不同图像**。其中 present 327 行、absent 201 行；
  absent 中 96 行来自官方明确的 no-target 表达，另 105 行来自目标空掩码。

MeViS 图像 tar 为 746,655,744 B，通过两个连续 HTTP Range 请求取得并合并；
SHA-256 为 `bcb641213c9b6b09a73503e143f3aa57898320872bf1626207e68d6ed6464c21`，
与官方 HF LFS 对象哈希完全一致。掩码 JSON 为 22,761,929 B，表达 JSON 为
121,014 B。四个主要官方数据对象合计 **788,787,099 B**（含 RefEgo 注释）。
这是对象大小之和，不是精确网卡流量：另有小型 README/API 探测、一个 1 MiB
RefEgo 范围探测，以及被终止的慢速整包尝试与有限网络重试。依据请求上界，
总数据传输保守不超过 1.0 GB，低于 1.5 GB 子任务下载预算。

图像归档本身包含完整 50-video valid_u 媒体；实际模型评测清单仅使用上述
12 个源视频 / 528 query-frame。其余视频未进入本轮推理清单。

## RefEgo 的直接使用条件

官方入口：

- [RefEgo 官方仓库](https://github.com/shuheikurita/RefEgo)
- [修复版官方注释](https://refegopublic.s3.amazonaws.com/RefEgoAnnotations2.zip)
- [字段与视频抽帧说明](https://github.com/shuheikurita/RefEgo/blob/main/dataset/README.md)
- [Ego4D 官方获取流程](https://ego4d-data.org/docs/start-here/)

| 官方划分 | clip 数 | 独立图像数 | query-frame 注释行数 | 目标不在画面行数 | is_multi=multi 行数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 9,172 | 225,500 | 451,000 | 104,634 | 173,140 |
| val | 1,549 | 38,470 | 76,940 | 18,296 | 37,500 |
| public test | 1,317 | 31,560 | 63,120 | 15,492 | 32,680 |

每张图有两条描述，因此注释行数不是不同视频帧数。public test 只审计结构和
数量，没有在它上面运行模型或据其结果选择方法。原包保留，结果见
`refego_annotation_audit.json`。

实际标签含 `video_id`、`clip_id`、`img_id`、`img_clip_id`、`bbox`、`caption`、
`is_obj_in`、`stationary`、`is_multi`，直接支持显隐与同类干扰的分层诊断。
注意 `is_multi` 是 `multi` / `unique` / null，**不能用字符串真值转换**；
`bbox` 是 xywh，缺席时可为 `[-1,-1,-1,-1]`，应以 `is_obj_in` 为依据。
新版图像宽为 640，保持纵横比，需配官方 `fps=2,scale=640:-1` 流程；
train 实际有 640×480、640×360、640×635，不能统一硬拉伸到 640×480，
也不能直接把框套在 Ego4D 原分辨率图像上。官方两个抽帧 shell 脚本已在
数据目录保存。

本子任务开始时，只检查本机 `~/.aws/config`、`~/.aws/credentials` 和相关
AWS 环境变量的**存在性**，均无现成入口；没有读取或输出秘密值。
Ego4D 视频授权与原图取得由主线程另行处理。注释许可 CC BY-SA 4.0 不替代
Ego4D 视频许可。本子任务没有代签许可或绕过门槛。

## MeViS 清单如何形成

官方来源：

- [MeViS 官方仓库及许可说明](https://github.com/henghuiding/MeViS)
- [官方项目页](https://henghuiding.com/MeViS/)
- [FudanCVL/MeViSv2 valid_u](https://huggingface.co/datasets/FudanCVL/MeViSv2/tree/main/valid_u)

该发布供非商业研究，CC BY-NC-SA 4.0。本轮保留来源而不重新发布媒体。

`samples.jsonl` 是唯一推理清单；每行有 sequence_id、source_id、frame_idx、
image_path、query、timestamp_s、gt_present、gt_bbox_xyxy、width、height、
annotation_source、dataset、source_split、query_id 和 selection_reason。
所有行标为 Development。框为像素 xyxy，左上包含、右下不包含。
`source_id` 与 `query_id` 是官方 JSON 的精确 key，图像 basename 是官方
帧名；可用这三个字段严格关联官方发布的预测，不能按列表位置猜测。

采样使用固定 seed=20261015：先取 6 个不同源视频的显隐切换表达，再取 6 个
不同源的多标注对象表达，每段选连续 36 个标注帧；最后在已选视频上加入
4 条官方 no-target 表达，各 24 帧。窗口优先覆盖显隐切换，完整算法、所有
候选、所选表达与起止位置保存为 `candidate_selection.json`。
**这是利用 GT 富集难例的 Development 样本，不用于估计自然发生的错误率。**

空目标只依据完整官方目标掩码为空，或表达明确无目标；没有把缺少标注当作
目标不存在。掩码 RLE 解码包含长度和非负检查，528 行图片均通过 PIL 解码
完整性和 mask/image 尺寸匹配检查；另用 8 个目标的独立像素栅格化 getbbox
交叉核验 RLE→bbox。12 源的首个 present 样本已生成并查看
`qa_gt_montage.jpg`，未据预览手工更改标签或选取结果。

## 解释边界

- MeViS 主要是第三人称、多动物与人员的视频，并非视障用户日常寻物；
  它能检验语言候选选择和显隐处理机制，不能单独证明视障应用收益。
- 官方表达常依赖动作、前后顺序或完整片段。36 帧窗口和逐帧定位模型可能
  看不到完整的语言消歧证据。应把这种任务不匹配与模型可修复错误分开。
- 发布元数据没有可靠源 FPS。本清单 `timestamp_s=null`，只能报告标注帧
  数量或延迟，不能把一帧擅自换算成固定秒数。
- 多标注对象不必然是同类干扰。`multiple_annotated_objects` 是自动采样
  条件，不能直接改名为“全部具有同类干扰”的真值子集。
- 基础模型可能已接触 MeViS 或其来源。任何在本样本上的收益都保持
  Development 身份，不能说成独立泛化或真实用户安全结论。

## 重现

```powershell
python `
  research/active/assistive-object-finding/acquire_pilot_data.py `
  --root artifacts.local/work/assistive-object-finding-pilot-20261015/data

python `
  research/active/assistive-object-finding/acquire_pilot_data.py `
  --root artifacts.local/work/assistive-object-finding-pilot-20261015/data `
  --refego-annotations-only
```

已有媒体时可用 `--materialize-only` 重做同一选择、物化和检查。
此脚本只有 HTTP、archive、标签与图像 I/O，CPU 放置理由为
`TASK_NOT_GPU_SUITABLE`。本子任务未运行大模型，也未占用 GPU。

## 下载中间块清理

完整 tar 和 RefEgo zip 的 SHA-256 复核后，按精确清单核对路径、文件哈希与
非 reparse 属性，使用 PowerShell `Remove-Item -LiteralPath` 删除了 22 个
重复 HTTP range 文件，共 **766,952,732 B**，并逐项确认不存在。完整 tar/zip、
官方标签、抽取图片、所有来源/哈希回执与清理回执均保留；明细见
`duplicate_parts_cleanup_receipt.json`。自动获取脚本会复用完整归档，不会因
中间块已清理而重新下载它们。
