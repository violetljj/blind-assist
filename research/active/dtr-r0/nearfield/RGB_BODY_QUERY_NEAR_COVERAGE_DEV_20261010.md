# RGB 近带负参考：新 rescan session 与缺失深度诊断

2026-10-10，EXPLORE／已消费 Development。继承 `d36e9e4b` 的[新负参考冻结评分](RGB_BODY_QUERY_NEGATIVE_SCORE_DEV_20261010.md)。用户“继续”授权补跨环境近场负参考；本轮采用尚未消费的缓存 rescan session，原同 scan 均匀扩帧停止规则继续生效。

新 16 个 rescan、15 个原额外校准环境、128 帧检验完成。严格 FREE 增加 22 个，其中近带只有 4 个，仍全部来自原来同一个环境。与旧 960 帧参考合并后，近带 23→27、环境数仍为 1，没有改善跨环境近负覆盖。按本轮事先条件，不运行新增 128 帧的推理或评分；当前 affine 和残差训练优先级保持原结论。

近带不足的一个具体原因已查明：大量 query 有已知 FREE 射线，但原生深度洞使整个 query 不能成为严格 FREE。这解释本数据上的负参考覆盖瓶颈，不定位所有距离误差，也不说明 RGB 没有近场信息。下一步应更换深度参考来源，优先另取未消费的 ARKit 官方 Training visit 作为校准候选；现有三条评价 capture 继续保留为 eval，不能转成 cal。本轮没有下载该新来源，也未选其帧或阈值。

## 选择、范围与停止

先只读库存：3RScan 缓存为官方 train 的 35 个环境、72 个 scan。原模型 train/cal/eval 涉及 20 个环境，已添加 cal 涉及另外 15 个；缓存已没有未消费的额外环境。额外 15 个环境仍有未消费的 16 条 rescan，每 scan 138–767 个配对 RGB/depth 帧，原生 RGB 960×540、depth 224×172，public K、depth shift=1000、identity sensor calibration extrinsics 可用。ARKit 本子线三 capture 已消费；TUM 本子线残包不完整，旧 manifest 的九个源路径也不存在，不能作为可用输入。

全部 16 条未消费 rescan 各固定均匀 8 帧，共 128 个唯一 `(scan, frame)`，沿用全 27 个固定 query；全量身份与 plan 先写，之后才读 depth。没有按 FREE、正例或模型输出筛环境、scan、帧、query。仍将 rescan 归入原环境身份，排除原 20 个 train/cal/eval 环境的全部 scan；官方 train 是源数据 split，本模型没有用这批帧训练。新 session 不是新环境，也未用 pose 确认实际机位改变。

严格 FREE 合同不变：query 内至少 16 条 FREE 射线，且 POSITIVE/UNKNOWN 射线都为零；缺失深度、box 前回波、无可观测域仍是 UNKNOWN。没有填洞、丢弃 invalid 射线、改 mask 或调整 query 边界。

执行前的决定检查：已有近带 FREE 只覆盖一个环境，最小有用覆盖变化是再增加一个环境。若新 session 的严格近带 FREE 覆盖多个环境，才对全 128 帧运行冻结模型和预定的全五臂绝对／归一化评分；否则停止这条路径，不再追加同 session 帧或运行 GPU。本结果触发后者；这不是否定其他传感器来源的停止规则，也不是残差模型的负证据。

## 参考结果

新帧共 3456 query：POSITIVE 1733、UNKNOWN 1701、严格 FREE 22，FREE 覆盖 8 个环境。各距离带都有 1152 个 query；有非零射线域，且域内全部在 public RGB 的 observed FOV 内。

| 距离带 | 新128帧 POSITIVE | UNKNOWN | 严格 FREE／环境 | 旧960帧 FREE／环境 | 合并1088帧 FREE／环境 |
| --- | --- | --- | --- | --- | --- |
| 0.3–0.8m | 202 | 946 | 4 / 1 | 23 / 1 | 27 / 1 |
| 0.8–1.5m | 747 | 388 | 17 / 7 | 86 / 12 | 103 / 12 |
| 1.5–3m | 784 | 367 | 1 / 1 | 20 / 7 | 21 / 7 |
| 全部 | 1733 | 1701 | 22 / 8 | 129 / 13 | 151 / 13 |

近带四个 FREE 都是 `1dd7209f-2ba0-22d9-8b9e-b5e270b2580f` 环境、`1dd720a1-2ba0-22d9-8b6e-bb00c888a414` scan 的 `fragment_x1_y2_z0`，帧 0、21、41、103。新旧参考负 query 身份交集为 0，合并计数不会重复计算。

合并 1088 帧仅指额外 15 个环境的参考覆盖，**不是新校准评分池**。其中旧 720 扩帧和本轮 128 帧均没有冻结预测评分；旧主 cal 的 32 FREE、阈值和原三折结果保持原样。未给这 151 个相关 FREE 创作新的 FPR 或独立确认结论。

## UNKNOWN 的机制诊断

旧 960 帧近带 8640 个 query 中，6928 个是 UNKNOWN，1689 个 POSITIVE，23 个严格 FREE。8617 个 query 含至少一条 UNKNOWN 射线；其中 6511 个已经有 ≥16 FREE 且零正像素，只因仍含 UNKNOWN 而不能成为严格负参考。这些 query 不能通过降低标准加入 cal。

固定各环境原选帧中按时间排序的首帧，共 15 帧，独立分解近带射线域 1,011,624 个：原生缺失深度 303,059、box 前回波 1193、box 内正像素 5981、box 后 FREE 701,391、RGB FOV 外 0。这个分解只限这 15 帧，不能外推为所有帧的精确成因比例。

中央近格 `fragment_x1_y1_z0` 在上述每帧的 domain 为 38,528，等于完整 172×224 depth 图。因此任何原生深度洞都会阻止中央近格成为严格 FREE。15 首帧共同缺失 1757 个像素位置，右边最后 7 列都缺失；该观察支持缓存原生 depth 的持续缺失问题，不能据此声称所有相机或全部距离带失效。其他近格也保留原始 UNKNOWN，FREE 不足并非因为这些 domain 未被 RGB FOV 观测。

新 128 帧近带 1152 query 中，903 个已经有 ≥16 FREE、零正像素，但含 UNKNOWN。固定各新 scan 的首选帧，共 16 帧、仍属于 15 环境，独立分解如下；中央近格在这些新首帧中仍覆盖完整 native depth 图，缺深度主导解释仍成立。

| 固定首帧近带射线分解 | 旧15 scan／15环境 | 新16 scan／15环境 |
| --- | --- | --- |
| domain | 1,011,624 | 1,080,432 |
| 原生缺失深度 | 303,059 | 314,458 |
| box 前回波 | 1193 | 718 |
| box 内正像素 | 5981 | 8358 |
| box 后 FREE | 701,391 | 756,898 |
| RGB FOV 外 | 0 | 0 |

新 session 分解也仅限上述固定首帧，不能外推为全部128帧的精确比例。`mechanism-review/new-session-check.json` 保存全新帧状态计数、固定首帧分解与独立一帧27query核验。无模型推理，也没有把“含已知FREE但局部UNKNOWN”换名为严格负参考。

## 核验、预算与复现

新 3456 个 query 的独立 XYZ／射线盒区间标签及严格状态计数逐项一致，128 帧原生 RGB/depth/K 身份一致，22 行负索引和 128 行 observation 序列化核验通过。另独立一个新帧的27query XYZ／计数／状态／observed与已存labels完全一致，没有重复全量核验。模型输入 observation 只含原 PUBLIC 字段，没有 evaluator depth、label、negative index。全量选择先于 depth 标签；无失败、无已有收据覆盖。

预算 CPU wall 1800s：库存／机制 240、参考 600、条件评分 500、核验／整合 460；条件 GPU allocation wall 400s（Depth Pro320、旧头80），下载／训练 0。参考实测 25.370/600s；条件评分／GPU／推理 NOT_RUN。CPU 总量保守计为库存／机制整个240s＋参考向上取整26s＋核验／整合整个460s，共726/1800s，而非测得的纯CPU时间。task-owned 进程、worker、GPU、锁均无保留；耐久参考、选择计划、来源哈希和诊断保留。

脚本：[新 session 参考](rgb_body_query_near_view_reference.py)。使用 `config/local.toml` 的 `dtr_r0_python`，从仓库运行 `research/active/dtr-r0/nearfield/rgb_body_query_near_view_reference.py --repo . --output artifacts.local/work/rgb-body-query-near-coverage-dev-20261010/view-reference --budget-s 600`。已执行目录不允许覆盖；复现应使用新的 ignored 输出目录。执行后只澄清 docstring／limitations 的新 session 与未核实机位含义，数值计算没有修改，原执行源码快照和 SHA 保留。

本轮 payload `artifacts.local/work/rgb-body-query-near-coverage-dev-20261010/` 包含 parent plan、`view-reference/` 的 manifest／observations／coverage／逐 query 计数／负索引／terminal／执行源码，以及 `mechanism-review/` 的原数据与新 session 诊断。原 960 帧 reference 未覆盖或重生成。

所有数字限于已消费 Development、相关 native first-return rays/query/frame 和同数据家族；严格 sampled FREE 不证明整盒、身体空闲或硬件安全。此次检验只判定缓存 rescan 未增加跨环境近负覆盖，没有测得新工作点、模型收益或训练效果；A0 的既有收益／成本分支与“失败不能否定重新训练”的边界保持原报告。
