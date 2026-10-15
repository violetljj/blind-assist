# 商品图片 OCR 观测

固定复用 `accessible_qa/image_receipts.json` 中原顺序的 96 张 Development
图片。推理脚本只读取图片和下载回执，不读取商品字段、问题、质量标签或
`products.json`。这部分输出是 OCR 观测，不是字段正确率或商品读取结果。

## 本次执行结果（2026-10-15）

两套观测都已跑完原 96 张图，无运行失败，1/96 张没有文本；该图仍在输出
和分母中。主 detector 共保留 995 个非空文字框，英文补充全部重读这些框。

| 观测 | 图数 | 默认接受文本行 / 全部框 | 运行墙钟 | 单图中位数 / P95 |
| --- | --- | --- | --- | --- |
| v5 server det + rec | 96/96 | 858/995 | 413.710 s | 3.936 / 6.883 s |
| 英文补充 rec（复用框） | 96/96 | 866/995 | 16.885 s | 0.111 / 0.604 s |

主运行墙钟包含模型下载和初始化；英文补充时模型已在首次失败启动中下载，
所以其 16.885 s 不包含那次下载。**二者工作量不同，不能用此表声称完整
OCR 提速比例。** 新模型合计 180,568,141 字节（180.57 MB）；原方向模型
585,532 字节从现成环境复制，未联网下载。

主模型的95个有文本图并不等于95张商品读对；默认接受行数增加也不是准确率
提升。首图人工目视诊断显示，即使 `UNSALTED/BUTTER` 高置信读出，同图
`CLOVER` 品牌仍被两个识别器读错。字段是否可靠需要下一层与目标字段及
独立评价答案比较。

逐图输入 hash、输出 hash、原96图顺序、995个框的边界及分数范围检查通过。
5/96 图的 EXIF orientation 为6，91/96 为1；已明确统一 EXIF 后坐标合同。
英文初次启动因序列化 enum name/value 不兼容失败，修正后运行；0条预测
在失败启动中生成，失败记录保留于 `english_decoder/startup_failure.json`。

## 固定模型与实现

- 主观测：PP-OCRv5 **server detection + server recognition**，原有 PP-OCRv4
  0/180 度文字行方向分类器；RapidOCR 3.9.2、ONNX Runtime 1.27.0。
- 补充观测：PP-OCRv5 English mobile recognition，只重读主观测保留的相同框。
  不新增框、不调阈值，不进入预设主字段特征；保留为独立一致性诊断。
- 实际 backend 为 `CPUExecutionProvider`。主 GPU 留给并行视觉模型；每个
  ORT session 使用 4 个 intra-op 和 1 个 inter-op 线程，OpenCV 1 线程。
- 主 detector 使用 RapidOCR 的 `limit_side_len=736, limit_type=min`；全图
  max side 2000。显式设 v5 官方 `mean=[0.485,0.456,0.406]` 与
  `std=[0.229,0.224,0.225]`，因为当前安装包的默认配置已经针对 v6。
- 保存全部非空识别行，`text_score=0`；`accepted_at_default` 单独标记
  识别分数 >=0.5，`text` 仅连接这些默认接受行。没有依据字段结果调整阈值。
- `box` 为 **EXIF 校正后全分辨率图像**的像素坐标四边形，不是 JPEG 存储
  raster 坐标。RapidOCR 路径 loader 使用 `ImageOps.exif_transpose`，英文补充
  的 OpenCV 默认读取同样遵循 EXIF。下游裁切和质量特征必须使用同一朝向；
  `coordinate_contract.json` 逐图记录存储尺寸、EXIF 和观测尺寸。保留无文本
  和失败行，不移出 96 图分母。
- 英文补充使用原分辨率裁切，主 server 识别使用预处理分辨率的裁切，故两者
  一致性是 **识别器加裁切分辨率的复合信号**，不能当成纯识别器架构消融。

官方模型来源和 SHA256：

| 模型 | SHA256 |
| --- | --- |
| `ch_PP-OCRv5_det_server.onnx` | `0f8846b1d4bba223a2a2f9d9b44022fbc22cc019051a602b41a7fda9667e4cad` |
| `ch_PP-OCRv5_rec_server.onnx` | `e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a` |
| `en_PP-OCRv5_rec_mobile.onnx` | `c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8` |

实际下载使用 RapidAI ModelScope `resolve/v3.9.2/onnx/PP-OCRv5/` 固定路由，
运行前核对上述官方登记 hash。URL、字节、耗时、源代码及输入 manifest
hash 均在 payload 回执中。

[RapidOCR 官方模型调用](https://rapidai.github.io/RapidOCRDocs/main/en/install_usage/rapidocr/how_to_use_ppocrv5/)、
[PaddleOCR 官方 v5 detector 配置](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/det/PP-OCRv5/PP-OCRv5_server_det.yml)、
[PaddleOCR 官方英文识别模型](https://www.paddleocr.ai/latest/en/version3.x/module_usage/text_recognition.html)。
这些文档用于配置核实，官方 benchmark 数字没有当成本次商品图片表现。

## 输出与复现

Payload：`artifacts.local/work/product-fields-dev-20261015/ocr/`。
主输出 `predictions.jsonl`，英文补充 `english_decoder/predictions.jsonl`；
两者均有 `manifest.json`、`progress.json`、`terminal.json` 和
`validation.json`、`coordinate_contract.json`。主目录另有 `download_receipts.json`。

每图 schema：`index, image, source_sha256, status, width, height, lines,
text, wall_s, error`。`lines` 包含 `text, confidence, accepted_at_default,
box`；英文补充增加对应主框的 `line_index`。主引擎无文本时不提供尺寸，
如果出现这种情况，真实尺寸保存在验证回执而不修改原始推理记录。

从 checkout 根目录运行，`$researchPython` 指向已有 RapidOCR 环境；输出目录
必须尚无该次 `predictions.jsonl`，脚本拒绝覆盖旧观测。

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$source = 'artifacts.local/work/assistive-thesis-screen-20261015/accessible_qa'
$output = 'artifacts.local/work/product-fields-dev-20261015/ocr'
& $researchPython research/active/assistive-thesis-screen/product_fields/ocr/run_ocr.py --input $source --output $output --threads 4 --wall-budget-s 1100
& $researchPython research/active/assistive-thesis-screen/product_fields/ocr/verify_observations.py --input $source --output $output
& $researchPython research/active/assistive-thesis-screen/product_fields/ocr/run_english_decoder.py --input $source --server-output $output --output "$output/english_decoder" --wall-budget-s 300
& $researchPython research/active/assistive-thesis-screen/product_fields/ocr/verify_observations.py --input $source --output "$output/english_decoder"
```

本次总授权预算：新增下载 <=300 MB、CPU command-wall <=1200 s；补充只使用
主观测完成后的剩余预算。墙钟上限不是许可把两个参数上限相加后超过总预算。
关闭 Python 进程即可释放计算资源；没有后台服务或远端资源。
