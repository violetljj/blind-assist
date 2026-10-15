# 商品字段读取与错误判断

2026-10-15，EXPLORE，复用96张Development商品图片。固定前64图拟合、后32图
评价，不追加数据、不选题定案。[结果与判断](RESULTS_20261015.md) ·
[预先记录的范围](PLAN.md)。

## 实际流程

1. [OCR](ocr/README.md)：PP-OCRv5 server检测与识别；另存相同检测框上的英文
   识别器输出，只用于独立诊断。
2. `vlm/run_vlm.py`：固定Qwen3-VL-2B-Instruct，图片一次、图片加默认接受的
   OCR文字一次；保存生成token置信度。两个prompt在读取评价结局前确定。
3. [字段证据](evidence/README.md)：预测字段与观测OCR匹配，保存原始图片hash、
   EXIF朝向和匹配框。所有推理与特征代码不读评价真值。
4. [评价合同](evaluation/README.md)：必需短语词面覆盖为主代理，放宽词序为
   敏感性分析，空标注NOT_EVALUABLE。训练164个有效字段，评价76个有效字段。
5. `compare_evidence.py fit`：只读64图训练真值；4折按图像隔离、每折独立
   插补/标准化，按训练折外log loss从C=0.1/1/10选择逻辑回归。每种特征组
   使用相同模型容量规则。保存模型、全部风险预测、训练折外结果和hash回执。
6. 两个读取模型全部封存后，`compare_evidence.py evaluate`才打开32图真值。
   strict/relaxed复用同一风险分数；分字段报告，按32图聚类bootstrap 1000次。

## 固定对照

| 错误判断器 | 输入 |
| --- | --- |
| field_length | 字段类型、是否空输出、输出token数，共5维 |
| global_quality | 上述5维、原16维像素特征、5维全图OCR统计，共26维 |
| global_plus_vlm_conf | 再加字段/序列平均生成log probability，共28维 |
| field_text_only | 4维字段预测文字与OCR的支持统计 |
| global_plus_vlm_and_text | 28维控制加4维字段文字支持，共32维 |
| all_plus_field_evidence | 28维控制加10维字段文字及局部几何/清晰度，共38维 |
| all_plus_field_and_agreement | 再加两次读取同字段token F1，共39维 |

主增量比较为38维字段证据对28维全部控制。39维一致性需要第二次VLM读取，
单列成本。先拟合后评价，不按评价最好的一组宣称确认结果。阈值取训练折外
风险的80%经验分位，再原样应用评价；同分时实际接受率可超过80%。训练折外
结果参与超参选择，不能作为独立性能估计。

## 复现与产物

Python需要已有NumPy、Pillow、scikit-learn；推理依赖分别见OCR及VLM记录。
所有运行输出放入`artifacts.local/work/product-fields-dev-20261015/`。
输出目录拒绝覆盖原拟合/评价；复现时使用新的任务内目录。

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& $researchPython research/active/assistive-thesis-screen/product_fields/compare_evidence.py fit `
  --features <observed-features.jsonl> `
  --train-truth artifacts.local/work/product-fields-dev-20261015/evaluation/truth_train.jsonl `
  --out <new-fitted-directory>
& $researchPython research/active/assistive-thesis-screen/product_fields/compare_evidence.py evaluate `
  --fitted <sealed-fitted-directory> `
  --eval-truth artifacts.local/work/product-fields-dev-20261015/evaluation/truth_dev_eval.jsonl `
  --out <new-evaluation-directory>
```

模型pickle只供可信本地产物复用，主要审计依据是JSON/JSONL、源代码和hash回执。
指标正类是错误，AUROC/AP越高越好，AURC及同覆盖错误率越低越好。固定覆盖
排序曲线是描述，不能替代训练阈值外推。自动词面分数可能误判同义词、漏检
额外幻觉；匹配文字也不证明属于目标物体。所有输出仍是Development诊断，
没有真实补拍、用户收益或现代大模型最优性结论。

主比较结束后另执行了[统一180度旋转诊断](vlm/ROTATION_DIAGNOSTIC.md)，
用于核查六图目视中发现的实体倒置。它独立标为posthoc Development，使用
`rotated_180`文件前缀和独立评价目录，不替换主结果或声称未见确认。
