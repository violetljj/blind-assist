# CNH + RGB 局部视锥融合：Development 实现接口

状态：`DEVELOPMENT_TRAINED / NO_RGB_BENEFIT_ESTABLISHED`。本页记录首个候选及后续对照，不变更 [CNH 路线方案](CNH_ROUTE_COMPARISON_PLAN_20260924.md)第 8 节七组、两 seed、训练预算、误报和事件判定规则。当前 2880 帧线性模型只用 H3 合成 CNH/ambient；它是数据到训练的工程对照，不能充当 RGB 融合收益基线。

## 机制与先验

- [cnh_rgb_frustum.py](cnh_rgb_frustum.py)仅用公开的相机内参、ToF 相对位姿、45° 分区、0.1–8 m 范围与声明的 3° 软边界，把各 ToF 分区投到 RGB 特征网格。相机和 ToF 不共心时，用 24 个固定深度样本近似软权重，再用射线与扩展分区的解析区间交集避免采样漏掉非零支持；权重大小仍需检查采样密度敏感性。不读场景深度、实例 ID、标签或预测结果。支持权重是“可能对应”，不是 CNH 峰的像素归属。
- [cnh_rgb_fusion.py](cnh_rgb_fusion.py)从无重叠 4×4 RGB patch 取得局部特征，按视锥支持加权聚合成每分区 RGB token；保留有符号 CNH，经 `sign(x) log1p(abs(x))` 编码，另外加入同响应标量、有效性、ambient 与 age。分区位置用两坐标共享线性映射，使 4×4/8×8 不因独立位置表产生额外参数。相同宽度和查询头可运行 `cnh`、`scalar`、`rgb` 三种输入臂。无效标量不被补成有效值；模型概率也不重写传感 `UNKNOWN`。
- 六个可学习查询对应现有 `left/centre/right × HEAD/BODY` 顺序，输出占用 logit 与非负距离。当前 Development 物化只有六项占用标签；距离头必须等独立距离监督定义/验证后训练，否则其数值无效。本实现没有时序记忆、提醒滞回或默认 App 集成。

这只是受约束的局部池化候选，不声称视锥校准已经准确。邻近分区可以因 3° 边界同时接收一个 RGB patch；`rgb` 模式只是工程消融，G1 应使用原方案 H1 的 RGB+标量输入。对基线比较须用相同 RGB 编码器、隐藏宽度、查询头和训练预算，记录实际参数量/计算量。正式数据输入在训练前还需核验曝光、标定、帧同步、asset split、实例标签精度和能量代理边界。旧 Street 布局与新巷道 train/dev 可作已披露 Development 调试；受保护 test 不作选模。

## 一个能推翻它的对照

先用同一批 Development 输入和固定布局划分，比较视锥支持与全局 RGB 池化：保持模型/训练设置一致，只有局部关联范围变动。观测六查询占用、按事件族的检出/漏报、首报时机及 `UNKNOWN`/覆盖，并核对两臂在同一曝光与 CNH 响应下的输入身份。若局部支持没有改善归属或显著增加漏报/计算，则不为“视锥”保留额外复杂度。随后按原方案比较 G3 与同配置 G3s，不能把 G3 对 G1 的整个配置差解释为直方图贡献。所有正式判定仍用原方案的布局 bootstrap 和误报预算；本页不预先宣称通过。

## 现有工作与贡献边界

[DELTAR (ECCV 2022)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610612.pdf) 已把轻量 ToF 的区域深度分布和 RGB 做对应 patch 的跨模态注意力并处理标定。[LiteSense (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Li_LiteSense_Lifting_Lightweight_ToF_with_RGB_for_High-Resolution_Metric_Depth_CVPR_2026_paper.html) 已用 CNH 与 RGB 的 patchwise 空间注入做度量深度估计。因此“CNH+RGB 局部融合”本身不能宣称新方法。这里待检验的区别是直接优化盲杖互补的六个接触/占用查询、保留弱回波与 `UNKNOWN`，并明确局部支持在标定不确定下的任务收益和误报/时机代价。区别是否有价值，取决于同场任务对照，而不是架构图。

## 已完成的窄检查

2026-09-25：本机 NumPy 几何测试 4/4（含稀/密深度采样不改变非零支持）；副机 research 环境最新代码的几何与 Torch 前向/局部支持测试共 7/7，含 H2/H3 候选参数差距不超过原 10% 目标。检查覆盖分区方向、非零基线、非刚体拒绝、有符号 CNH 输入、无效标量、三个模式的有限输出，以及改动支持区外的 RGB patch 不影响该单区查询。两次副机测试临时目录在成功调用中已清理。这些检查只证明实现接口，不是已训练性能、真实标定或几何准入。

[单帧真实输入烟测入口](cnh_rgb_fusion_smoke.py)随后在本机已有 Torch/Pillow 环境运行：原始 Street Development 帧 `street-dev-00/centre/0` 的 RGB、camera 与 H3 观测身份/哈希一致，64 区均有视锥支持，21,954 参数候选产生六项有限输出；不读取目标，回执为 `artifacts.local/evidence/cnh-rgb-fusion-smoke-20260925-v1/result.json`，状态 `PASS_REAL_DEVELOPMENT_INPUT_INTERFACE_ONLY`。这仍不是已训练的任务收益、RGB 质量或真实标定证据。原先向副机传送该帧时 SSH/SCP 两次中断；副机部分上传的数据目录及本地传输包已核对删除，后续确认并删除了唯一残留的任务脚本。

## 2026-09-25 同划分训练对照

六张 train/dev 巷道作者图修复 RGB 后，使用 480/480 帧及三颗固定 seed 训练相同架构的 ToF-only（RGB 置零）与 CNH+RGB。V1 两臂在固定 logit 0 阈值下每 seed 都是 TP0/FP0/FN408/TN2472；保存预测的事后 AUROC 约 0.677、AP 约 0.269，两臂差异不到 0.001。回执为 `artifacts.local/evidence/cnh-rgb-alley-dev-comparison-20260925-v1/result.json` 和 `artifacts.local/evidence/cnh-rgb-alley-rank-diagnostic-20260925-v1/result.json`。

另立并在运行前冻结的[V2方位掩码与训练集类别平衡协议](CNH_RGB_ALLEY_DEV_V2_PROTOCOL_20260925.md)保留相同六图、三seed、训练预算和固定阈值：ToF-only dev TP148–160/408、FP486–512/2472，CNH+RGB TP148、154、159，FP504、491、503。六组保存预测的计数独立复核一致；预设判定 `NONCOLLAPSED_DEVELOPMENT` 与 `NO_CONSISTENT_DEV_PARETO_SIGNAL`，回执 `artifacts.local/evidence/cnh-rgb-alley-dev-v2-20260925-v1/result.json`。V2 同时更改查询约束与类别权重，不能拆分归因；高查询级误报、仅三处 train 和三处 dev 物理场地、未独立核验的标签精度以及缺失的 Street 全量 RGB 限制结论。两轮均未接触受保护 test，不构成正式算法或安全收益。
