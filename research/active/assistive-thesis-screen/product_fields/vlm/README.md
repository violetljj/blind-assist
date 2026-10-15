# 商品字段：本地 Qwen3-VL 基线

本分支只从已有 96 张商品图生成 `product / brand / variety`，不读
`products.json`、质量标签或评价结果。它属于 Development 筛查；模型生成
置信度不是正确率，也不证明真实补拍收益。字段评分由上层独立评价脚本完成。

## 固定模型与输入

- [官方模型卡](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)：
  Qwen3-VL-2B-Instruct，revision
  `89644892e4d85e24eaac8bacfd4f463576704203`，Apache-2.0。
- BF16 权重 SHA256：
  `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`。
  本机 HF resolve TLS 失败；改用官方卡链接的 ModelScope 仓库下载，
  对照上述固定 HF revision 的 LFS SHA256 与文本 Git blob SHA1 全部核验。
  失败日志、来源 API 元数据及最终逐文件收据保留在 payload。
- 既有运行环境：Torch 2.11.0+cu130、Transformers 5.14.1；
  RTX 5060 Laptop 8 GB。整个 BF16 模型在 `cuda:0`，SDPA、batch=1，
  没有 CPU offload、量化、训练、新依赖或模型服务 API。
- 输入以原 `image_receipts.json` 顺序及 SHA256 校验，显式执行
  `ImageOps.exif_transpose`；记录原尺寸、旋转后尺寸、EXIF orientation。
  图像处理范围 65,536–1,048,576 pixels，保留比例、使用模型处理器。
- 固定英语提示要求仅返回可见的具体包装信息，缺证据用空字符串。
  greedy、最多 160 个新 token、seed=0；完整提示和配置保存在每臂 `config.json`。

## 两条读出路径

1. `image_only`：只给整图，一次生成三个字段。首三图作为启动 canary，
   成功后同配置续写余 93 图；没有重跑首三图。
2. `ocr_conditioned`：同一模型、图像和基础提示，附 PP-OCRv5 的现成 `text`
   字段，即 OCR 默认置信度 >=0.5 的接受行全文。固定后缀提醒 OCR 可能出错，
   仍以图片核验；每图保存实际合成提示。与字段证据算法可能使用的全部 OCR
   行不同，这里没有自行更改 OCR 门槛或依据评价改写文字。

每图保存原始输出、token ID/文本/字符位置/logprob。字段置信度只汇总与
对应 JSON 字符串值中英数字符相交的生成 token，排除键名及纯标点 token；
同时保留整条生成的平均 logprob。它们均为未校准模型分数。空字段无字段
logprob，不能被当作可信正确答案。推理失败单独标为 `status=error` 并保留
traceback；最早三条 canary 用 `error=null` 表示成功，原记录保留。

## 运行和证据

源码入口：`download_model.py`、`run_vlm.py`、`summarize.py`。
payload：`artifacts.local/work/product-fields-dev-20261015/vlm/`。
模型文件、日志、两臂 JSONL、配置、进度、终态和 `summary.json` 均在其中。
`summarize.py` 检查每臂图像唯一性、三字符串字段、token 文本重建和字段
logprob 重算；不读取商品真值。

复现示例（从 checkout 根目录运行；GPU Python 使用本机现有入口）：

```powershell
& E:/codex-tools/bin/blindassist-research-gpu.cmd `
  research/active/assistive-thesis-screen/product_fields/vlm/run_vlm.py `
  --model artifacts.local/work/product-fields-dev-20261015/vlm/model `
  --images artifacts.local/work/assistive-thesis-screen-20261015/accessible_qa/images `
  --manifest artifacts.local/work/assistive-thesis-screen-20261015/accessible_qa/image_receipts.json `
  --output artifacts.local/work/product-fields-dev-20261015/vlm/image_only
```

第二臂加 `--ocr-jsonl artifacts.local/work/product-fields-dev-20261015/ocr/predictions.jsonl`
并指定新的输出目录。相同目录只在配置完全一致时跳过已完成图，不改写原预测。
本轮 GPU 进程墙钟上限 1,800 秒、模型/依赖下载上限 6 GB；实际完整数字以
`summary.json` 和 `model/download_receipt.json` 为准。任务不启动驻留服务。

## 本轮执行结果

两臂各 **96/96 张成功、0 推理或 JSON 解析失败**；每臂非空输出均为
product 96、brand 96、variety 84。这只是生成覆盖率，尚不是字段正确率。

| 路径 | 推理中位数 / P95（秒/图） | 推理总秒 | 进程墙钟秒 |
| --- | --- | --- | --- |
| image_only | 1.414 / 1.921 | 169.594 | 206.875 |
| OCR-conditioned | 1.422 / 1.985 | 151.834 | 171.078 |

进程墙钟共 **377.953/1,800 秒**，含首三图 canary、冷启动和第二次模型加载；
不能按第二臂总推理更短断言 OCR 加速，第一臂首图有 18.25 秒冷启动开销。
最高 Torch allocated 为 **4,841,172,992 bytes**，不是整卡总显存。
最终模型运行资产为 **4,266,647,442 bytes（11 文件）**；下载/核验脚本墙钟
146.54 秒，另有少量元数据、1024-byte 权重探针与失败传输日志，下载总量低于
6 GB。没有新增依赖。

192 条生成记录的唯一图像、三字段类型、token 重建和字段 logprob 重算检查
通过；三个任务 Python 进程均已结束，显存回到约 1.5 GiB 的桌面背景占用。
模型和原预测由本 Development 小试保留作复现及独立评价，不是驻留服务。
