# 视障辅助操作与环境音课题筛选

2026-10-15。优先保留**面向视障辅助操作的动作条件功能部位定位**作为下一候选。环境音小试跑通了数据与模型，但简单阈值已经很强，加入局部证据只有收益与代价的交换；目前没有理由把它优先扩成毕业课题。本轮没有确定论文题目。

## 功能部位定位为什么值得继续

用户已经找到杯子，仍可能需要知道杯把在哪里；找到工具，也不等于知道哪一部分对应所需动作。这能衔接 BlindAssist 的物体感知与语音提示，但研究对象转向**物体内部的功能区域及其指向位置**。第一阶段可完全使用公开图片和现成标签，不依赖眼镜、机械臂或人工录像。

已从 AGD20K 官方公开 ZIP 按字节范围取得 6 对真实图片与标签，下载约 8.64 MB，全部能够解码且图像与标签尺寸一致。杯子在 hold、drink_with、pour 三种动作下的原图字节完全相同；hold 与 pour 的标签也相同，集中在杯把，drink_with 的标签集中在杯口。还取得冰箱、微波炉开门区域和刀柄样例。这直接证明了“同一物体、不同操作目的”有可用评价标签，同时说明不同动作的正确区域并不一定互斥。

AGD 标签是连续功能热图，6 个标签均有 256 个灰度级，不能阈值化后冒充精确部件 mask 或真实接触真值。该数据适合用作者的 KLD、SIM、NSS 等热图指标。已索引 2,215 个测试图像与标签配对记录，这不是 2,215 个独立物体或训练样本的声明；另有 35 张 Seen 测试图片缺少匹配 GT，须计为不可评价。训练主要使用图像级弱监督，本轮检查的测试热图不能混入训练。[AGD20K 与 LOCATE 作者仓库](https://github.com/Reagan1311/LOCATE)

[IIT-AFF 官方数据页](https://sites.google.com/site/iitaffdataset/)列出 8,835 张图片、14,642 个物体框、24,677 个功能部件，包含杯、锤、刀、锅等 10 个物体类别和 9 种功能区域。实际取得杯、刀、锤 3 对图片和标签，原生标签是 1024×1024 的整数像素矩阵，与 RGB 尺寸一致。刀与锤具有握持区域和功能头的离散标签；杯样例是容纳与包握区域，尚不能据此宣称有独立杯把 mask。该数据更适合部件分割与指向点评价；开关、抽屉等覆盖不能由类别表直接推出。

IIT 原生划分为 train_and_val 6,184 张、val 1,767 张、test 2,651 张；val 是 train_and_val 子集，剔除后训练为 4,417 张，test 与前两者无 ID 交叉。图片数不等于独立物体数，后续仍需检查近重复及物体来源。AGD 与 IIT 两项核验累计下载主体 16,367,996 字节，另留 1 MB HTML 探测记账余量；没有下载完整 ZIP，也没有执行视觉模型。

## 创新要落到具体失败上

[SelectiveCL ICCV 2025](https://arxiv.org/html/2508.07877)已经用 DINO 和 CLIP 区分物体、功能部位与背景，并设计了选择性的原型和像素对比学习。当前作者仓库有训练和测试代码，公开 Seen 权重响应为 712,621,477 字节；本轮没有下载或执行该视觉权重。作者旧依赖和 GT 缓存路径仍需适配。

[POP CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Tsai_Pointing_at_Parts_Training-Free_Few-Shot_Grounding_in_Multimodal_LLMs_CVPR_2026_paper.pdf)已经研究少样本部件指向，将多模态模型的注意力与视觉对应结合。[RoboMAP](https://arxiv.org/html/2510.10912v1)已经研究适应不同空间任务的热图与不确定性，并使用置信阈值。因此，“加对比学习”“热图变指向点”“加拒识”都不能单独宣称为新方法。

可继续检验的两个研究内容如下，当前均为假设：

1. **动作条件下的细小功能区域精修。**先测强基线是否经常找对物体却错过细窄把手。用动作相关证据选择局部高分辨率区域，并约束精修前后仍对应同一操作目的。比较原模型、统一提高分辨率、固定切片及相同计算量的裁剪方法。只有相同预算下，小部位与跨物体类别都获得收益，才支持方法贡献；单纯多看几张裁剪图不算。
2. **从功能区域选择稳定的指向位置。**在已有离散功能区标签上，联合考虑点是否落在区域内、与区域边界的关系、轻微视角或图像变化后的位移，以及允许不输出时的覆盖率。比较热图最大点、区域中心、距离变换选点和简单置信阈值。目标是减少指错位置而不靠大量拒绝；“安全抓取成功”不属于这些标签能证明的结果。

两部分可以形成区域感知与输出选择两个方法章节，再加自动评测和软件回放系统。数据准备、推理、训练小模块、消融、分组统计与报告可以自动化；人工主要用于检查少量失败样例、解释标签和界定需求。公开标签支撑算法评价，不能替代视障用户研究。

半年可行性的最弱环节是**强基线是否真的存在可利用的错误，以及改法是否超过简单控制**。下一步应先跑约 200–300 张、按原图来源隔离的 Development 图片，检查功能区面积、细长程度、动作和物体类别上的错误，再决定训练。冻结 DINO/CLIP、缓存特征、训练小模块是可控的起步方案；该视觉路线的本机显存和训练时长尚未实测。

## 环境音小试结果

使用 ESC-50 的 120 段真实录音，共 600 秒，来自 120 个不同来源。40 段用于拟合、20 段用于阈值校准、60 段用于评价；评价含 30 段已知主类和 30 段词表外主类。各分区没有同源或相同音频哈希交叉，校准与评价的词表外类别不同。方法和阈值规则见 [PLAN](PLAN.md)。

固定 CLAP 主干；每段编码完整 5 秒和 3 个固定 2.5 秒窗口。所有拒识方法只看预设的 10 个已知类别。阈值在校准集选择，以保留至少 9/10 段已知主类。评价结果如下：

| 方法 | 正确并接收的已知主类 /30 | 接收的词表外主类 /30 |
| --- | ---: | ---: |
| 不拒识 | 30 | 30 |
| 最大相似度阈值 | 24 | 0 |
| 第一与第二名分数差阈值 | 27 | 1 |
| 三个局部窗口取最大值 | 25 | 5 |
| 三种文本提示集成 | 24 | 0 |
| 全局特征 logistic 小头 | 27 | 1 |
| 全局加局部特征 logistic 小头 | 29 | 3 |

局部小头相对全局小头，救回 2 段已知主类，没有损失已正确接收的样本，但额外接收 2 段词表外主类。它与简单分数差阈值的总体正确计数都是 56/60；没有显示全面优势。使用完整 50 类词表的闭集诊断为 57/60，与上述拒识任务分开保存。

这只是**主类别是否位于指定词表内的代理任务**。ESC-50 没有穷尽标注一段声音里的全部事件，因而表中的词表外接收不能称为真实环境误报，也不能证明某个目标声音不存在。真实多声音共存、连续事件定位及对视障用户的帮助均未被这次实验覆盖。现成 [FlexSED](https://arxiv.org/html/2509.18606v1)等工作已处理自由文本事件查询及缺失标签，后续研究不能仅以裸 CLAP 为唯一强基线。

本轮 CLAP 模型与配置下载 617,900,277 字节，音频下载 52,925,280 字节。120 段完整实验产生 480 次编码，进程运行 31.150 秒；加 canary 累计 GPU 进程运行 53.503 秒，峰值已分配显存约 0.74 GB。没有付费 API、新依赖或人工标注。FSD50K 可选路径只取得部分索引，音频为 0 条，状态 NOT_RUN；不据此宣称已验证自然多事件任务。

## 复现与证据

代码：[数据获取](acquire_audio_pilot.py)、[模型下载](download_model.py)、[固定提示](class_prompts.json)、[固定推理](run_clap_pilot.py)、[七组比较](analyze_audio_pilot.py)、[AGD 数据核验](audit_affordance_data.py)、[IIT 数据核验](audit_iit_affordance_data.py)。模型及 ESC 源版本已固定，下载脚本交付时修复了随远端 HEAD 漂移的问题，没有重跑实验或改变原始数据。

以仓库根目录为工作目录，在已有 PyTorch、Transformers、scipy、scikit-learn 环境中运行：

```powershell
$audioRoot = 'artifacts.local/work/assistive-audio-pilot-20261015'
$audioCode = 'research/active/assistive-object-finding/audio_screen'
python research/active/assistive-object-finding/audio_screen/acquire_audio_pilot.py --root $audioRoot
python research/active/assistive-object-finding/audio_screen/download_model.py --root $audioRoot
python research/active/assistive-object-finding/audio_screen/analyze_audio_pilot.py --self-check
```

新实验目录完成数据获取后，复制固定 `class_prompts.json` 到其根目录，再执行以下命令。已完成目录不得覆盖评分：

```powershell
python "$audioCode/run_clap_pilot.py" --manifest "$audioRoot/data/observations.jsonl" --prompts "$audioRoot/class_prompts.json" --model-dir "$audioRoot/models/clap-htsat-unfused" --output-dir "$audioRoot/inference" --device cuda --batch-size 4 --max-seconds 800
python "$audioCode/analyze_audio_pilot.py" --root $audioRoot
```

完整推理参数、提示、模型版本、逐样本预测、分区哈希、训练系数、阈值与 840 行逐样本 decision 均在 payload 下；复现实验应使用另一个输出目录，并保留同一数据和提示清单。数据核验脚本的 `--deadline-utc` 默认保留本次停止时刻，后续执行须明确指定新任务的截止时间。

Payload：`artifacts.local/work/assistive-audio-pilot-20261015/`。其中 `analysis/summary.json` 是完整音频结果，`analysis/sealed_models_thresholds.json` 保存冻结参数；`alternatives/` 保存功能部位数据、来源和可视化核验。模型、数据、标签与失败回执保留用于续做，GPU 进程及写锁已释放。
