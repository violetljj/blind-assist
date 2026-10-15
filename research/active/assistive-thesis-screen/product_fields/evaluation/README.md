# 商品字段自动评价合同 v1

本合同评价**标注必需信息的词面覆盖错误**，不等于完整语义读取正确。错误预测器只可读取原图、模型预测字符串、OCR文字/框/置信度与训练集标签；不得把真值词在OCR里是否命中当成输入。

## 固定数据与文件

保留前轮 `image_receipts.json` 的96图顺序，前64图train，后32图dev_eval。`build_manifest.py`验证每张图片bytes和SHA256，输出到 `artifacts.local/work/product-fields-dev-20261015/evaluation/`：

- `inference_manifest.jsonl`：每行只有`image`、`path`，推理端只读此文件。
- `truth_manifest.jsonl`：评估器专用，含split、source_id、image_sha256及三个字段的required/optional。
- `truth_train.jsonl`、`truth_dev_eval.jsonl`：明确分离训练与评价真值。
- `manifest_audit.json`：分母、hash、跨split标签重复风险。已有输出仅允许内容相同的重建。

train可评164/192字段：product60、brand59、variety45。dev_eval可评76/96字段：28、28、20。空list及只有optional=true的字段均NOT_EVALUABLE，不能转换为不可回答真值或负类。

名称和原图SHA256跨split均无重复。eval中product、brand、variety完整规范化required标签曾在train出现的分别9/28、2/28、2/20；三字段合并签名相同1/32图。签名重复不是相同实物的确认，这个split也没有摄影者/实物独立性，仍是已消费Development。

## 固定评分

输入统一做Unicode NFKD/casefold、去组合音标、`&`转and、去英文/弯引号，标点作为token边界。只用对应字段回答。

1. **strict（主）**：每条`optional=false`必需短语必须以连续token序列出现于回答。条目之间无顺序要求。
2. **relaxed（敏感性）**：必需token全部覆盖，短语内部可乱序；同token在多个短语里重复时取单条短语所需的最大次数。没有同义词扩展、标签后改或看eval后定阈值。
3. 辅助输出必需span覆盖率、token precision/recall/F1及exact token bag。它们同样只是词面诊断。空预测/明确拒答若缺必需词，会记为读取任务未完成；不能把它解释为自信错误回答。

两主口径容许额外词，因此可能漏掉额外幻觉；正确同义词可能被误记为错误。`soda`与`soft drink`不自动等同。生成`scoring_disputes_review_queue.jsonl`：包括全部strict/relaxed分歧，以及按image+field hash选取的每model/split/field前三个strict错误，仅供争议诊断；审阅不回改本轮原标签或主评分。

[作者仓库](https://github.com/Accessibility-Research-Collective-UCI/image-quality-vlm-chi26)解释了optional语义。[原论文](https://dl.acm.org/doi/full/10.1145/3772318.3791309)由4名研究者人工评价7,436条caption，允许轻微拼写/术语变体，核查品牌/子品牌、品种及重大幻觉；本轮自动代理不是作者人工评分的复现，不能直接对比其caption准确率。

## 输入与运行

```json
{"image":"VizWiz_train_00015916.jpg","fields":{"product":{"text":"milk","risk_scores":{"global_quality":0.3,"field_evidence":0.1}},"brand":{"text":"brand"},"variety":{"text":"original"}}}
```

`fields`值也可为纯字符串。每份prediction必须包含truth manifest每张图的三个字段；缺失推理报错而不静默丢分母。推理失败可由上游显式记录，并在报告里与正常拒答分开，不伪装为成功输出。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python research/active/assistive-thesis-screen/product_fields/evaluation/build_manifest.py
python research/active/assistive-thesis-screen/product_fields/evaluation/verify_contract.py
python research/active/assistive-thesis-screen/product_fields/evaluation/score_fields.py --predictions model=path/to/predictions.jsonl --out artifacts.local/work/product-fields-dev-20261015/evaluation/scored
```

## 错误预测比较建议

正类固定为读取代理错误，分数越高越可能错。报告AUROC、error AP（常数基线即错误率）、AURC（低风险先接受，越低越好）、同覆盖25%/50%/75%/90%/100%的错误数与分母。AURC及同覆盖期望按同分组随机次序解析计算；另给固定hash tie-break下实际整数计数，不让常数分数依文件顺序获得虚假收益。没有正/负类时AUROC返回null，零错误时AP返回null。

仅76个eval字段且同图三个字段相关，应以**图像为单位**bootstrap差值，不能把76字段视为独立受试者；若报告区间，建议固定1000次seed，不用其结果调模型。主比较应保留同预测骨干、同64张训练图、同正则/超参、相同插补规则，只改变全图与字段证据。单模型低维线性预测器足够本轮探索，训练内交叉验证可按image分组，插补/标准化须在每折训练内完成。

固定覆盖曲线仅是排序描述，不是可部署阈值；部署切点必须只用训练/训练内折外数据选定，再原样应用dev_eval。主线只报告新模型生成一次后固定的全部候选，禁止看32图成绩再挑特征/阈值而仍称未见评价。
