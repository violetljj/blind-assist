# 语言指引寻物：小规模可行性验证

2026-10-15 · EXPLORE / Development · 保留方向，暂不定题。

本轮已把公开数据取得、标签转换、真实模型推理、候选重排、轻量学习、
时序回放和统一评价串通。主要工作可以自动完成，无需新增人工逐帧标注。
数据和代码可继续复用，但两个拟议研究点的简化实现尚无可靠收益，不能
据此认定创新点已成立。

当前决定见 [CURRENT](CURRENT.md)，范围与预算见 [PLAN](PLAN.md)，
执行记录见 [RUNS](RUNS.md)，数据来源细节见 [数据获取说明](data_access_notes.md)。

## 这次验证回答了什么

1. **数据能否直接进入自动实验：可以。**MeViSv2 的公开标签可直接驱动
   定位、显隐、候选覆盖和恢复评价；RefEgo 修复版注释也已实际解析。
2. **现代方法是否仍有困难：有。**可见目标选错、低分正确候选被过滤、
   目标离场后持续输出均已实际观察。强任务方法的官方预测也有这些错误。
3. **拟议方法是否已经改善：尚未。**简单 CLIP 重排、固定小型学习器及
   自适应时序策略没有给出兼顾定位、离场输出和恢复的可靠收益。

建议下一步优先取得 RefEgo 真实第一视角像素，检查候选保留与指定实例
存在性。先确认这一问题及可观察证据，再决定是否把时序调度作为第二章。
本轮不继续扩大模型、参数搜索或数据规模。

## 数据与评价单位

| 项目 | 本轮事实 |
| --- | --- |
| 实际推理数据 | MeViSv2 `valid_u` 的 12 个源视频 |
| 规模 | 16 个 query 序列，528 个 query-frame，456 张不同图像 |
| 可见 / 缺席 | 327 / 201 行；缺席含 96 行官方 no-target 表达 |
| 采样 | 固定 seed，显隐切换与多标注对象困难窗口富集 |
| 定位成功 | 可见帧 bbox IoU ≥ 0.5，分母始终为 327 |
| 默认存在门限 | 原始 DINO 最大候选分数 ≥ 0.25，未据评价结果选点 |
| 时间 | 原始 FPS 未提供，`timestamp_s=null`，只报告标注样本数 |
| 第一视角验证 | Ego4D 申请已提交，待邮件协议及官方审批；RefEgo 像素实验 **NOT_RUN** |

MeViS 多为动物、人员、动作描述及第三人称视频。部分表达依赖片段中的
动作先后关系，逐帧模型或短窗口可能看不到完整证据。这里的错误率不能
当作视障日常寻物错误率；多标注对象也不自动等于“同类干扰”真值。

框从完整官方目标掩码导出，采用像素 xyxy、右下边界不包含的约定。
空目标来自明确无目标表达或官方目标空掩码，未把未标注当作不存在。
统一评价使用 `(sequence_id, frame_idx)` 联接，缺失预测另报覆盖率。

## 研究点一：候选比较与实例辨别

Grounding DINO-base 实际推理，原候选池保留分数 ≥ 0.05，经过固定 NMS。
初始 CLIP 比较和小型学习比较只在原 DINO ≥ 0.25 池内选择；存在门限保持。
裁剪和特征均由候选框产生，推理时不读取 GT。

| 方法 | 定位成功 / 327 | 相对 DINO 救 / 损 |
| --- | ---: | ---: |
| 原始 DINO 分数排序 | 82（25.08%） | 参照 |
| 原 ≥ 0.25 池 CLIP 重排 | 70（21.41%） | 25 / 37 |
| 独立特征 L2 logistic | 65（19.88%） | 25 / 42 |
| 候选相对特征 L2 logistic | 81（24.77%） | 7 / 8 |
| 全 ≥ 0.05 池 CLIP，事后机制试验 | 85（25.99%） | 69 / 66 |

两个学习器使用 3-fold GroupKFold，以 source_id 分组。同一源的所有帧、
表达都在同一折；缩放和拟合只使用训练折，C=1 固定，无超参数搜索。
候选相对模型比独立模型多对 16 帧，但仍少于原 DINO 1 帧。
三折相对 DINO 净变化为 +1、−1、−1，不能写成超过基线。

全池 CLIP 是发现过滤问题后追加的 **posthoc Development** 试验，复用
已有特征，没有新训练或参数搜索。净多 3/327 帧伴随救 69、损 66，变化
幅度大且互相抵消，不能声称稳定提升；原存在门限下缺席仍输出 201/201。

候选 oracle 只供评价者诊断，不能作为方法成绩：

- 全缓存 ≥ 0.05 池内有正确框：273/327（83.49%）。
- 原 ≥ 0.25 池内有正确框：116/327（35.47%）。
- 原 top1 错误中，全池有正确框 191 帧；全池也缺框 54 帧。
- 保持 ≥ 0.25 池时，相对 82 帧最多再救 34 帧，不是 191 帧。

因此有候选选择空间，但“保留低分正确候选”与“避免无目标输出”需要
共同研究。只给低分框重新排序，尚未解决指定实例存在性。

## 强任务方法：官方 StreamingRVOS-1B 缓存

本轮取得官方 HF 固定 revision 的 `1b/MEVIS_U/results.json`，通过 Range
只取约 64.7 MB 压缩条目。原压缩体、JSON、URL/range/hash 收据保留。
这是**官方预测重评分**，没有在本机执行 StreamingRVOS 或测量其速度。

video ID、expression ID、帧名和完整表达精确匹配后，共 432/528 行可评分：
327 可见、105 离场。MeViSv2 新增的 96 行 no-target 表达在旧官方缓存中
没有预测，明确为 NOT_EVALUABLE，未当作正确的“无目标”。

| 指标 | 结果 |
| --- | ---: |
| 原生 mask 派生框 IoU ≥ 0.5 | 147/327（44.95%） |
| 离场时仍输出非空 mask | 88/105（83.81%） |
| 离场输出面积 ≥ 图像 0.1% | 83/105，描述性检查 |
| 可见帧原生 mask J / F | 39.08% / 44.12% |
| 全 432 帧原生 J&F | 35.42%，不是全量官方榜单成绩 |

独立栅格解码再次复算了 147/327、88/105；帧、尺寸、GT 框及查询映射一致。
官方 mask 没有置信度，统一 JSONL 中 score=1 仅表示非空状态，禁止解释成
模型置信度或套用 0.25 分数曲线。原生空 mask 是该缓存的存在性输出。
缓存只包含最终 mask，缺少候选池；**DINO 的 273/327 oracle 不适用于它**。

## 研究点二：按恢复需要分配定位计算

真实 OpenCV KLT 在标注图像上运行；重定位读取本轮已测 DINO 缓存。
自适应策略使用前后向误差、有效点数和最大间隔，不读取 GT。
以下给出实际动作数，不能把缓存回放耗时当完整在线系统耗时。

| 策略 | 定位调用 / 528 | 定位成功 / 327 | 缺席输出 / 201 |
| --- | ---: | ---: | ---: |
| 每帧 DINO | 528 | 82 | 201 |
| 每 2 标注样本重定位 + KLT | 264 | 77 | 188 |
| 每 4 标注样本重定位 + KLT | 132 | 67 | 147 |
| 每 8 标注样本重定位 + KLT | 72 | 59 | 120 |
| 自适应，最大间隔 8 | 119 | 72 | 187 |
| 每序列同调用数均匀对照，事后预算 | 119 | 64 | 168 |

同调用数对照的预算来自完整片段的自适应调用数，属于 posthoc 对照，
不是可直接在线获得的预算。自适应相对它救 13、损 5 个可见帧，净 +8；
但离场输出清掉 4、增加 23，净多 19。14 个重新可见段，自适应只恢复
1 段，对照 2 段，其余明确 censored。当前没有全面优势。

调用数、实际 KLT 时间、缓存中原 DINO 时间和组件成本估计分别记录。
减少调用与定位下降、停止输出、恢复失败存在取舍，不能只报调用节省。

## 自动化、算力与证据边界

本地 8 GB 显存环境完成 528/528 行 DINO + CLIP 推理并保存候选与特征。
GPU 进程墙钟记账共 321.516 秒（含 smoke），峰值显存约 2.13 GB；这不是
GPU kernel 时间或手机端速度。CPU KLT 与小型学习不新增 GPU/模型下载。
原始 Windows 计时分辨率有限，单次细粒度耗时只作粗测。

数据选择与方法迭代均保持 Development 身份；3-fold 分源也不是新增独立
确认。目标是判断下一步，不证明创新、真实视障体验、导航或安全收益。
原始失败、原候选池比较和后续全池试验分别保留，没有回写失败结果。

## 来源、脚本与轻量回执

- 数据：[RefEgo](https://github.com/shuheikurita/RefEgo)、[Ego4D 获取流程](https://ego4d-data.org/docs/start-here/)、[MeViS](https://github.com/henghuiding/MeViS)、[MeViSv2](https://huggingface.co/datasets/FudanCVL/MeViSv2)。
- 模型：[DINO-base](https://huggingface.co/IDEA-Research/grounding-dino-base)、[CLIP-B32](https://huggingface.co/openai/clip-vit-base-patch32)、[StreamingRVOS](https://github.com/wkzhang636/StreamingRVOS)。
- 数据脚本：[acquire](acquire_pilot_data.py)；模型：[grounding](run_grounding_pilot.py)；时序：[KLT](run_temporal_pilot.py)；学习：[ranking probe](probe_learned_reranking.py)。
- 评价：[analyze](analyze_pilot.py)；强方法缓存：[replay](replay_official_streamingrvos.py)；诊断图：[montage](make_diagnostic_montage.py)；事后全池：[pool probe](probe_candidate_pool.py)。
- 数据回执：[data_summary](../../../artifacts.local/work/assistive-object-finding-pilot-20261015/data/data_summary.json)、[download receipts](../../../artifacts.local/work/assistive-object-finding-pilot-20261015/data/download_receipts.jsonl)。
- 模型回执：[downloads](../../../artifacts.local/work/assistive-object-finding-pilot-20261015/model_download_receipts.json)、[inference hashes](../../../artifacts.local/work/assistive-object-finding-pilot-20261015/inference/hash_receipt.json)。
- 汇总：[analysis rollup](../../../artifacts.local/work/assistive-object-finding-pilot-20261015/analysis/analysis_rollup.json)、[learning](../../../artifacts.local/work/assistive-object-finding-pilot-20261015/learned_rank_probe/summary.json)、[pool change](../../../artifacts.local/work/assistive-object-finding-pilot-20261015/candidate_pool_probe/summary.json)。
- 缓存证据：[range receipt](../../../artifacts.local/work/assistive-object-finding-pilot-20261015/official_streamingrvos/receipt.json)、[mapping audit](../../../artifacts.local/work/assistive-object-finding-pilot-20261015/official_streamingrvos/independent_mapping_audit.json)。
- 时序回执：[terminal](../../../artifacts.local/work/assistive-object-finding-pilot-20261015/temporal/terminal.json)。媒体、模型和逐帧载荷均留在 ignored `artifacts.local/`。

## 复现

从仓库根执行，`python` 指向与运行回执相容的环境；需要 CUDA PyTorch、
Transformers、Pillow、requests、NumPy、OpenCV，学习探针另需 scikit-learn。
固定模型 revision、尺寸和阈值见回执；复用已有权重及数据。

```powershell
$pilot = "artifacts.local/work/assistive-object-finding-pilot-20261015"
$route = "research/active/assistive-object-finding"
python "$route/acquire_pilot_data.py" --root "$pilot/data" --materialize-only
python "$route/run_grounding_pilot.py" --manifest "$pilot/data/samples.jsonl" --model-dir "$pilot/models/grounding-dino-base" --clip-dir "$pilot/models/clip-vit-base-patch32" --output-dir "$pilot/inference"
python "$route/analyze_pilot.py" --self-test
python "$route/analyze_pilot.py" --manifest "$pilot/data/samples.jsonl" --predictions "$pilot/inference/base_predictions.jsonl" --output-dir "$pilot/analysis/base"
python "$route/analyze_pilot.py" --manifest "$pilot/data/samples.jsonl" --predictions "$pilot/inference/clip_predictions.jsonl" --output-dir "$pilot/analysis/clip"
python "$route/replay_official_streamingrvos.py" --root "$pilot" --fetch
python "$route/run_temporal_pilot.py" --manifest "$pilot/data/samples.jsonl" --predictions "$pilot/inference/base_predictions.jsonl" --output-dir "$pilot/temporal"
python "$route/probe_learned_reranking.py" --root "$pilot"
python "$route/probe_candidate_pool.py" --root "$pilot"
python "$route/analyze_pilot.py" --manifest "$pilot/data/samples.jsonl" --predictions "$pilot/candidate_pool_probe/all_pool_predictions.jsonl" --output-dir "$pilot/candidate_pool_probe/analysis"
```

数据首次取得时去掉 `--materialize-only`；RefEgo 注释审计可加
`--refego-annotations-only`。该命令不完成 Ego4D 视频许可或像素获取。
多个模型分别评价，目录自动合并仅适用于同一模型的分片，不能混合不同
模型的同键预测。强方法公平比较用其 `matched_manifest.jsonl` 限制相同
432 行；原生 J/F 与额外 96 行缺失状态保留，不填补或推断。

RefEgo 必须使用已修复缩放错误的 `RefEgoAnnotations2.zip`：框为官方
640 宽、保持纵横比的图像坐标；`is_multi` 为分类值而非字符串布尔值。
第一视角候选保留和存在性实验仍是下一步，不能把本轮机制诊断写成已验证。
