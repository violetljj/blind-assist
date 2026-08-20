# BA-ADT small-target visual upper bound R5 Attempt 03 protocol

状态：`FROZEN_BEFORE_TEACHER_OUTCOME / ATTEMPT_02_PRE_OUTCOME_SUPERSEDED / CONSUMED_DEVELOPMENT / TERMINAL_REDETECTION_GATE`

## Amendment reason

Attempt 02 的 OWLv2 replay 仅运行到 preview frame 450；最早冻结失败窗口从 frame 2039 开始，因而没有
读取任何 R5 判定帧，也没有产生科学结果。用户随后提供 24 GB RTX 4090D 并明确冻结终止性顺序：
`DINOv-SwinL -> 必要时 SAM 3.1 image-only -> R5 收口`。本 amendment 在任何固定帧 teacher inference
之前，以更贴合跨图 visual in-context prompting 的 DINOv-SwinL 取代 OWLv2。OWLv2 checkpoint 只保留为
工程中断记录，不进入 R5 比较或结论。

## Frozen Teacher A

Teacher A 为官方 `UX-Decoder/DINOv` Swin-L，source revision
`53bf20d5cfdbb86fa35141a1cff432d4923599f2`。官方 release `model_swinL.pth` 长度为
902,781,487 bytes，SHA-256 为
`167FEC1F006AF8D2D53C662290DD2DFF8E667AA66C8C0836AF1181533D334A9A`。使用官方 visual
in-context demo 路径：每个历史 exemplar 的整图 RGB 与正例 bbox binary mask 生成 content embedding，
五个 embedding 做算术平均，再对另一张 target image 进行 open-set mask proposal；mask 的正 logit 区域
转为 bbox。

输入冻结为 1408×1408 RGB、batch 1、FP16 autocast。官方 demo 内部 score filter 保持其 `0.12` 起点，
无候选时每次降低 `0.04` 直至至少一项；不做阈值 sweep。输出 bbox 以 IoU 0.70 去重并保留每帧 top 100。
不启用 text prompt、GT prompt、video propagation、memory update、后验人工修正或多尺度/tile sweep。

Outcome-blind mechanics preflight 发现官方 2024 repo 的 Torch 1.13/CUDA 11.7 环境无法与远端唯一 CUDA
12.8 compiler 混编。正式 runtime 因此在任何判定帧前冻结为 Python 3.12、Torch 2.8.0+cu128、
torchvision 0.23.0+cu128、Detectron2 0.6（source revision
`42121d75e10d9f858f3a91b6a39f5722c02868f0`）和 CUDA 12.8。DINOv 自有 MSDA op 只做两处
Torch API 等价迁移：`value.type()` 改为 `value.scalar_type()`；patch SHA-256 为
`8ABF6E3514486544B67162A4CB4E74531A845B9CF4DE20AA121A27F0DDBAB282`，patched source SHA-256
为 `0B7C38657C05CA77E335CDA87A06499B62B51DB40393DA3B57551DC7010DFCEF`。Pillow 12 只恢复已删除的
`Image.LINEAR = Image.BILINEAR` 别名，shim SHA-256 为
`594A51E9460F8BA78B18F9FC66D880E58BE6C51B9B38E597CE6809A7F50AA625`。两项兼容改动都不改变
kernel 数学或插值算法，runner 会验证 runtime 与 patched source hash 后才推理。

非判定 frame 225 mechanics smoke 已验证五 exemplar + 1408 image-only inference 可完成：target forward
阶段 3.12 s，24 个候选，峰值 CUDA reserved 15,426,650,112 bytes；该输出标记
`formal_run=false`，不进入 evaluator。正式 runner 在加载完整 DINOv checkpoint 前关闭冗余 Swin-L
ImageNet preload；checkpoint 已含 357 个 backbone keys，因此不额外下载/短暂加载另一份 backbone。

历史 exemplar 只由 GT-blind R1 observation 决定：首个 `ACQUIRED -> LOST` segment 内，选择 detector
confidence 不低于 0.60 的 top 5。冻结结果为 frame 208/206/198/200/203，对应 confidence
0.761395/0.679363/0.663399/0.643646/0.615761。bbox 全部来自当时的 RGB detector/track state，
不读取 ADT GT。

## Frozen execution cohort

本次按用户明确要求只推理 R4 已消费的 W2/W3/W4 97 个 eligible RGB frames，而不是完整 3,824 帧：

- W2 preview frames 2039-2065，共 27 帧；
- W3 preview frames 3222-3271，共 50 帧；
- W4 preview frames 3292-3311，共 20 帧。

帧区间来自已冻结的 R4 eligibility denominator，仅定义 consumed benchmark cohort；远端 teacher 不接收
GT archive、目标位置、bbox、distractor 标注、per-frame visibility 数值或 scenario answer。输出仍包含 3,824
个 frame slots，只有上述 97 帧有 teacher candidates，从而由原隔离 evaluator 按原索引读取。每 5 个已处理
帧原子 checkpoint；resume 必须验证 attempt、source revision、checkpoint/input hashes 和已处理 frame ids。

## Frozen outcome gate

原 R5 的 3-window/97-frame denominator、正确 proposal `IoU >= 0.10`、指标与结论上限不变：

- `2/3` 或 `3/3`：Teacher A 明确成功，立即停止，不运行 Teacher B；只允许另立 teacher-to-edge protocol；
- `0/3`：只允许机制不同的 SAM 3.1 image-only Teacher B，使用相同 exemplar、97 帧与 evaluator；
- `1/3`：先解释成功窗是否仅为较大目标；最多允许同一个 SAM 3.1 Teacher B，不增加其他 teacher；
- Teacher B 后无论结果如何，R5 永久收口，不建立 R6/R7 rescue，不追加 T-Rex2、DINOv2 或模型动物园。

`0/3 + 0/3` 关闭 appearance-only tiny-target redetection immediate priority，转向 Last-10m destination
grounding 与 camera motion/VIO/SLAM/world memory。positive 只证明当前 consumed RGB 存在可利用能力上界，
不自动授权 Sky、蒸馏、手机部署或默认 App。胡萝卜 tiny-instance 结论不得外推到医院入口、电梯门、服务台、
商店入口或公交站牌等大尺度 destination grounding。

## Frozen input identity

- preview RGB SHA-256：`77F952E3AF6531A4A4D9DB5D714292545B5D4A33F5C820D22EEEEC6541B4CC32`；
- RGB-only R1 observations SHA-256：`2B1D9D6D9B3B7548FE98DADD597D5403C56266417C747484DAF66480973C249F`；
- R1 failure accounting SHA-256：`7D7B8672FD0E53C272DD0829A6B0F346C5A31070F8F2832443828B4D4C888590`；
- R4 evaluation SHA-256：`83F87B9CB8FA528AAAA8CCAD84380D2285755FBFAEE41383E8BDC169154ECD76`；
- evaluator-only GT SHA-256：`18297581A15FEEF097B57109BA67D52414E203F53AABA82D781E0B658CFE6A63`。

正式 compute 可在用户提供的 AutoDL RTX 4090D worker 执行；协议、GT evaluator、结果验收与 terminal
receipt 保持本地权威。Claim ceiling 始终为 consumed Development teacher capability，不产生 held-out、edge
efficiency、真实用户、交互式导航、安全或默认 App 权限。
