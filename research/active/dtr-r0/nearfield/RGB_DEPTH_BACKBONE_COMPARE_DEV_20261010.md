# RGB 深度主干横向比较：冻结 query 评价

日期：2026-10-10。`RGB_DEPTH_BACKBONE_COMPARE_DEV_20261010`；EXPLORE，仅推理与train-only统计标定，全部为已消费Development。起点Git `d0c52ce2`，集成前共享master `bb0cd8fa`。四个官方主干各完整496帧，DepthPro使用原缓存，无神经训练/微调。

## 结果与决定

**本轮冻结query系统的职责定为：近带0.3–0.8m交给ToF，RGB负责≥0.8m；中远带实验首选Depth Anything V2 Metric Indoor Large的原始米制输出（不加全局affine）。** 三ARKit中远带见证440/554、FREE25/199，对DepthPro affine为354/554、27/199；救138、损52，FREE增16、减18。这是本次Development选型判断，原DepthPro affine固定对照和旧停止配方保留。

推荐边界：DAV2 raw远带292/323低于对照319/323（救2、损29，−8.46%），收益集中于中带148/231 vs35/231。40777065中远FREE21/103 vs10/103，逐capture无成本替换未获支持；三capture合计成本下降不能消去这条漂移。UniDepth affine见证最多468/554，但FREE32/199高于27/199；若愿意承担更多误支持，它是见证上限候选，本次不把它写成满足“不增加误支持”的推荐。Metric3D affine399/554、FREE23/199，MoGe affine408/554、FREE38/199，全部两臂保留。

**原强信号为NOT_EVALUABLE。** 40777065近带POS=0，因此“双追加capture近带见证均严格提高”无检验力，不能算已通过，也不能以此否定所有模型近带信息。唯一可评价的追加capture40777060：DAV2 raw6/11、FREE1/133 vs原0/11、0/133，其余七个新臂均0/11；没有在这个capture获得近带见证提高且近带FREE不增。arkit16的DAV2 raw为2/44、FREE0/96，UniDepth raw/affine为30/44与26/44、FREE8/96与4/96，局部正结果不删。

**近带米制信息确实恢复，失败不能写成主干测距都不行。** 40777060原生近带10243像素中，DepthPro affine meanAbsLog2.0503、medianAbsRel3.3301；DAV2 raw为0.1928/0.0953，Metric3D raw0.1643/0.0502，MoGe raw0.1791/0.1271。Metric3D raw在arkit16/60的cutoff分别0.32044/0.27114，MoGe raw三折均约0.349；近带绝对余量最多(0.8−0.3)/2=0.25，因此这些cutoff结构上排除近带见证，即使像素深度准确。职责分工针对当前冻结读出，不是单目近带原理或硬件物理极限结论；后续若继续，新的主干信息与跨带分数尺度才是待研究对象，本轮不挪阈值。

## 冻结输入、配方与校准

公开清单496帧=56train（7个原3RScan环境）＋304cal（原16＋追加3RScan240＋追加ARKit48）＋136eval（validation24＋新3RScan64＋三个ARKit各16）。RGB字节SHA、公有K、color/native shape和注册采样与DepthPro完全一致；ARKit主输入256×192，3RScan沿用原960×540 RGB及native224×172。没有新增VGA次要臂。模型只读RGB和public K；native LiDAR只供train统计拟合/评价，不输入神经推理。

DAV2用官方Indoor Hypersim Large、518 lower-bound等比多14 resize/Imagenet归一化、maxdepth20；UniDepth supplied K、官方infer默认分辨率；Metric3D官方616×1064等比resize/mean padding，去pad与native回采样后乘fx×resize_scale/1000；MoGe-2真正scale_head metric版、官方默认resolution9/fov_x=None/apply_mask/force_projection。模型FP32参数，CUDA官方/AMP FP16推理，输出浮点/native插值。Metric3D保留官方clamp[0,300]；MoGe官方无效mask保留并计数。

每模型raw与自身global log-affine两臂。Affine只用相同56train的有效observed配对像素：初始OLS＋8次Huber IRLS，残差尺度0.2log，斜率限制[.25,4]，截距加权均值，与DepthPro完全同配方；cal不转train。每模型先封存fit与304cal切点，再读eval参考。三折每折切点仅取其他两个capture的严格sampledFREE，held自身FREE不选阈值；pooled cal是独立304cal切点补充。

query分数和已知正射线见证分数均第16大绝对余量min(z−entry,exit−z)。public query Z带0.3–0.8/0.8–1.5/1.5–3m；POS要求≥16已知正射线，严格FREE要求正0、UNKNOWN0、free≥16，UNKNOWN不进入负例。冻结DepthPro public有效域不变；模型无效预测只能失去支持，不能删负query。切点whole ties、预算≤floor(.05N)，允许负cutoff，完整沿用原评价；同5%校准预算不是每capture等错误率。三折cal FREE分母467/352/325、预算23/17/16，pooled588、预算29。

## 主三折：各主干×距离带合计

单位：W/POS为正见证；S/POS为POS上的query支持；F/FREE为严格FREE支持；U/UNKNOWN为UNKNOWN支持。三ARK帧相关，不是独立重复。

| 主干/臂 | 距离带 | W/POS | S/POS | F/FREE | U/UNKNOWN | 见证救/损 | FREE增/减 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DAV2 Indoor Large/raw | 0.3-0.8m | 8/55 | 10/55 | 1/373 | 0/4 | 8/0 | 1/0 |
| DAV2 Indoor Large/raw | 0.8-1.5m | 148/231 | 154/231 | 16/177 | 21/24 | 136/23 | 16/5 |
| DAV2 Indoor Large/raw | 1.5-3m | 292/323 | 307/323 | 9/22 | 42/87 | 2/29 | 0/13 |
| DAV2 Indoor Large/affine | 0.3-0.8m | 0/55 | 0/55 | 0/373 | 0/4 | 0/0 | 0/0 |
| DAV2 Indoor Large/affine | 0.8-1.5m | 97/231 | 102/231 | 11/177 | 21/24 | 91/29 | 11/5 |
| DAV2 Indoor Large/affine | 1.5-3m | 278/323 | 290/323 | 19/22 | 6/87 | 0/41 | 0/3 |
| DepthPro affine | 0.3-0.8m | 0/55 | 0/55 | 0/373 | 0/4 | 0/0 | 0/0 |
| DepthPro affine | 0.8-1.5m | 35/231 | 35/231 | 5/177 | 0/24 | 0/0 | 0/0 |
| DepthPro affine | 1.5-3m | 319/323 | 323/323 | 22/22 | 86/87 | 0/0 | 0/0 |
| UniDepth V2 ViT-L/raw | 0.3-0.8m | 30/55 | 32/55 | 8/373 | 2/4 | 30/0 | 8/0 |
| UniDepth V2 ViT-L/raw | 0.8-1.5m | 152/231 | 155/231 | 26/177 | 2/24 | 123/6 | 26/5 |
| UniDepth V2 ViT-L/raw | 1.5-3m | 292/323 | 302/323 | 2/22 | 31/87 | 0/27 | 0/20 |
| UniDepth V2 ViT-L/affine | 0.3-0.8m | 26/55 | 30/55 | 4/373 | 1/4 | 26/0 | 4/0 |
| UniDepth V2 ViT-L/affine | 0.8-1.5m | 161/231 | 164/231 | 26/177 | 23/24 | 132/6 | 26/5 |
| UniDepth V2 ViT-L/affine | 1.5-3m | 307/323 | 315/323 | 6/22 | 42/87 | 2/14 | 0/16 |
| Metric3D v2 ViT-L/raw | 0.3-0.8m | 0/55 | 0/55 | 0/373 | 0/4 | 0/0 | 0/0 |
| Metric3D v2 ViT-L/raw | 0.8-1.5m | 93/231 | 106/231 | 23/177 | 1/24 | 88/30 | 23/5 |
| Metric3D v2 ViT-L/raw | 1.5-3m | 208/323 | 220/323 | 11/22 | 6/87 | 0/111 | 0/11 |
| Metric3D v2 ViT-L/affine | 0.3-0.8m | 0/55 | 0/55 | 6/373 | 0/4 | 0/0 | 6/0 |
| Metric3D v2 ViT-L/affine | 0.8-1.5m | 95/231 | 98/231 | 13/177 | 22/24 | 93/33 | 13/5 |
| Metric3D v2 ViT-L/affine | 1.5-3m | 304/323 | 322/323 | 10/22 | 35/87 | 0/15 | 0/12 |
| MoGe-2 metric ViT-L/raw | 0.3-0.8m | 0/55 | 0/55 | 0/373 | 0/4 | 0/0 | 0/0 |
| MoGe-2 metric ViT-L/raw | 0.8-1.5m | 18/231 | 28/231 | 14/177 | 1/24 | 17/34 | 14/5 |
| MoGe-2 metric ViT-L/raw | 1.5-3m | 89/323 | 111/323 | 18/22 | 4/87 | 0/230 | 0/4 |
| MoGe-2 metric ViT-L/affine | 0.3-0.8m | 0/55 | 0/55 | 0/373 | 0/4 | 0/0 | 0/0 |
| MoGe-2 metric ViT-L/affine | 0.8-1.5m | 125/231 | 131/231 | 23/177 | 2/24 | 119/29 | 23/5 |
| MoGe-2 metric ViT-L/affine | 1.5-3m | 283/323 | 301/323 | 15/22 | 15/87 | 0/36 | 0/7 |

## 主三折：逐capture×距离带与配对

每臂同一432query；各带144query。下面全体逐带，逐query原始决策、双方支持/见证、救损在每模型final/main-loco/query_pairs.csv。

| Capture | 主干/臂 | 带 | W/POS | S/POS | F/FREE | U/UNKNOWN | 救/损 | FREE增/减 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arkit16 | DAV2 Indoor Large/raw | 0.3-0.8m | 2/44 | 4/44 | 0/96 | 0/4 | 2/0 | 0/0 |
| arkit16 | DAV2 Indoor Large/raw | 0.8-1.5m | 71/114 | 73/114 | 0/9 | 21/21 | 59/23 | 0/5 |
| arkit16 | DAV2 Indoor Large/raw | 1.5-3m | 56/70 | 70/70 | 0/0 (N/E) | 37/74 | 0/12 | 0/0 |
| arkit16 | DAV2 Indoor Large/affine | 0.3-0.8m | 0/44 | 0/44 | 0/96 | 0/4 | 0/0 | 0/0 |
| arkit16 | DAV2 Indoor Large/affine | 0.8-1.5m | 36/114 | 40/114 | 0/9 | 21/21 | 30/29 | 0/5 |
| arkit16 | DAV2 Indoor Large/affine | 1.5-3m | 48/70 | 53/70 | 0/0 (N/E) | 2/74 | 0/20 | 0/0 |
| arkit16 | DepthPro affine | 0.3-0.8m | 0/44 | 0/44 | 0/96 | 0/4 | -/- | -/- |
| arkit16 | DepthPro affine | 0.8-1.5m | 35/114 | 35/114 | 5/9 | 0/21 | -/- | -/- |
| arkit16 | DepthPro affine | 1.5-3m | 68/70 | 70/70 | 0/0 (N/E) | 74/74 | -/- | -/- |
| arkit_40777060 | DAV2 Indoor Large/raw | 0.3-0.8m | 6/11 | 6/11 | 1/133 | 0/0 (N/E) | 6/0 | 1/0 |
| arkit_40777060 | DAV2 Indoor Large/raw | 0.8-1.5m | 44/69 | 47/69 | 3/75 | 0/0 (N/E) | 44/0 | 3/0 |
| arkit_40777060 | DAV2 Indoor Large/raw | 1.5-3m | 107/122 | 107/122 | 1/12 | 2/10 | 2/15 | 0/11 |
| arkit_40777060 | DAV2 Indoor Large/affine | 0.3-0.8m | 0/11 | 0/11 | 0/133 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777060 | DAV2 Indoor Large/affine | 0.8-1.5m | 32/69 | 32/69 | 0/75 | 0/0 (N/E) | 32/0 | 0/0 |
| arkit_40777060 | DAV2 Indoor Large/affine | 1.5-3m | 110/122 | 114/122 | 9/12 | 1/10 | 0/10 | 0/3 |
| arkit_40777060 | DepthPro affine | 0.3-0.8m | 0/11 | 0/11 | 0/133 | 0/0 (N/E) | -/- | -/- |
| arkit_40777060 | DepthPro affine | 0.8-1.5m | 0/69 | 0/69 | 0/75 | 0/0 (N/E) | -/- | -/- |
| arkit_40777060 | DepthPro affine | 1.5-3m | 120/122 | 122/122 | 12/12 | 9/10 | -/- | -/- |
| arkit_40777065 | DAV2 Indoor Large/raw | 0.3-0.8m | 0/0 (N/E) | 0/0 (N/E) | 0/144 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777065 | DAV2 Indoor Large/raw | 0.8-1.5m | 33/48 | 34/48 | 13/93 | 0/3 | 33/0 | 13/0 |
| arkit_40777065 | DAV2 Indoor Large/raw | 1.5-3m | 129/131 | 130/131 | 8/10 | 3/3 | 0/2 | 0/2 |
| arkit_40777065 | DAV2 Indoor Large/affine | 0.3-0.8m | 0/0 (N/E) | 0/0 (N/E) | 0/144 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777065 | DAV2 Indoor Large/affine | 0.8-1.5m | 29/48 | 30/48 | 11/93 | 0/3 | 29/0 | 11/0 |
| arkit_40777065 | DAV2 Indoor Large/affine | 1.5-3m | 120/131 | 123/131 | 10/10 | 3/3 | 0/11 | 0/0 |
| arkit_40777065 | DepthPro affine | 0.3-0.8m | 0/0 (N/E) | 0/0 (N/E) | 0/144 | 0/0 (N/E) | -/- | -/- |
| arkit_40777065 | DepthPro affine | 0.8-1.5m | 0/48 | 0/48 | 0/93 | 0/3 | -/- | -/- |
| arkit_40777065 | DepthPro affine | 1.5-3m | 131/131 | 131/131 | 10/10 | 3/3 | -/- | -/- |
| arkit16 | UniDepth V2 ViT-L/raw | 0.3-0.8m | 30/44 | 32/44 | 8/96 | 2/4 | 30/0 | 8/0 |
| arkit16 | UniDepth V2 ViT-L/raw | 0.8-1.5m | 83/114 | 83/114 | 1/9 | 0/21 | 54/6 | 1/5 |
| arkit16 | UniDepth V2 ViT-L/raw | 1.5-3m | 63/70 | 69/70 | 0/0 (N/E) | 24/74 | 0/5 | 0/0 |
| arkit16 | UniDepth V2 ViT-L/affine | 0.3-0.8m | 26/44 | 30/44 | 4/96 | 1/4 | 26/0 | 4/0 |
| arkit16 | UniDepth V2 ViT-L/affine | 0.8-1.5m | 91/114 | 92/114 | 1/9 | 21/21 | 62/6 | 1/5 |
| arkit16 | UniDepth V2 ViT-L/affine | 1.5-3m | 63/70 | 70/70 | 0/0 (N/E) | 34/74 | 0/5 | 0/0 |
| arkit_40777060 | UniDepth V2 ViT-L/raw | 0.3-0.8m | 0/11 | 0/11 | 0/133 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777060 | UniDepth V2 ViT-L/raw | 0.8-1.5m | 33/69 | 35/69 | 3/75 | 0/0 (N/E) | 33/0 | 3/0 |
| arkit_40777060 | UniDepth V2 ViT-L/raw | 1.5-3m | 102/122 | 106/122 | 0/12 | 5/10 | 0/18 | 0/12 |
| arkit_40777060 | UniDepth V2 ViT-L/affine | 0.3-0.8m | 0/11 | 0/11 | 0/133 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777060 | UniDepth V2 ViT-L/affine | 0.8-1.5m | 34/69 | 35/69 | 3/75 | 0/0 (N/E) | 34/0 | 3/0 |
| arkit_40777060 | UniDepth V2 ViT-L/affine | 1.5-3m | 113/122 | 114/122 | 3/12 | 5/10 | 2/9 | 0/9 |
| arkit_40777065 | UniDepth V2 ViT-L/raw | 0.3-0.8m | 0/0 (N/E) | 0/0 (N/E) | 0/144 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777065 | UniDepth V2 ViT-L/raw | 0.8-1.5m | 36/48 | 37/48 | 22/93 | 2/3 | 36/0 | 22/0 |
| arkit_40777065 | UniDepth V2 ViT-L/raw | 1.5-3m | 127/131 | 127/131 | 2/10 | 2/3 | 0/4 | 0/8 |
| arkit_40777065 | UniDepth V2 ViT-L/affine | 0.3-0.8m | 0/0 (N/E) | 0/0 (N/E) | 0/144 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777065 | UniDepth V2 ViT-L/affine | 0.8-1.5m | 36/48 | 37/48 | 22/93 | 2/3 | 36/0 | 22/0 |
| arkit_40777065 | UniDepth V2 ViT-L/affine | 1.5-3m | 131/131 | 131/131 | 3/10 | 3/3 | 0/0 | 0/7 |
| arkit16 | Metric3D v2 ViT-L/raw | 0.3-0.8m | 0/44 | 0/44 | 0/96 | 0/4 | 0/0 | 0/0 |
| arkit16 | Metric3D v2 ViT-L/raw | 0.8-1.5m | 32/114 | 34/114 | 0/9 | 0/21 | 27/30 | 0/5 |
| arkit16 | Metric3D v2 ViT-L/raw | 1.5-3m | 27/70 | 33/70 | 0/0 (N/E) | 2/74 | 0/41 | 0/0 |
| arkit16 | Metric3D v2 ViT-L/affine | 0.3-0.8m | 0/44 | 0/44 | 0/96 | 0/4 | 0/0 | 0/0 |
| arkit16 | Metric3D v2 ViT-L/affine | 0.8-1.5m | 34/114 | 35/114 | 0/9 | 21/21 | 32/33 | 0/5 |
| arkit16 | Metric3D v2 ViT-L/affine | 1.5-3m | 55/70 | 70/70 | 0/0 (N/E) | 30/74 | 0/13 | 0/0 |
| arkit_40777060 | Metric3D v2 ViT-L/raw | 0.3-0.8m | 0/11 | 0/11 | 0/133 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777060 | Metric3D v2 ViT-L/raw | 0.8-1.5m | 36/69 | 43/69 | 6/75 | 0/0 (N/E) | 36/0 | 6/0 |
| arkit_40777060 | Metric3D v2 ViT-L/raw | 1.5-3m | 81/122 | 83/122 | 6/12 | 1/10 | 0/39 | 0/6 |
| arkit_40777060 | Metric3D v2 ViT-L/affine | 0.3-0.8m | 0/11 | 0/11 | 0/133 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777060 | Metric3D v2 ViT-L/affine | 0.8-1.5m | 41/69 | 43/69 | 4/75 | 0/0 (N/E) | 41/0 | 4/0 |
| arkit_40777060 | Metric3D v2 ViT-L/affine | 1.5-3m | 119/122 | 122/122 | 6/12 | 2/10 | 0/1 | 0/6 |
| arkit_40777065 | Metric3D v2 ViT-L/raw | 0.3-0.8m | 0/0 (N/E) | 0/0 (N/E) | 0/144 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777065 | Metric3D v2 ViT-L/raw | 0.8-1.5m | 25/48 | 29/48 | 17/93 | 1/3 | 25/0 | 17/0 |
| arkit_40777065 | Metric3D v2 ViT-L/raw | 1.5-3m | 100/131 | 104/131 | 5/10 | 3/3 | 0/31 | 0/5 |
| arkit_40777065 | Metric3D v2 ViT-L/affine | 0.3-0.8m | 0/0 (N/E) | 0/0 (N/E) | 6/144 | 0/0 (N/E) | 0/0 | 6/0 |
| arkit_40777065 | Metric3D v2 ViT-L/affine | 0.8-1.5m | 20/48 | 20/48 | 9/93 | 1/3 | 20/0 | 9/0 |
| arkit_40777065 | Metric3D v2 ViT-L/affine | 1.5-3m | 130/131 | 130/131 | 4/10 | 3/3 | 0/1 | 0/6 |
| arkit16 | MoGe-2 metric ViT-L/raw | 0.3-0.8m | 0/44 | 0/44 | 0/96 | 0/4 | 0/0 | 0/0 |
| arkit16 | MoGe-2 metric ViT-L/raw | 0.8-1.5m | 5/114 | 5/114 | 0/9 | 0/21 | 4/34 | 0/5 |
| arkit16 | MoGe-2 metric ViT-L/raw | 1.5-3m | 11/70 | 11/70 | 0/0 (N/E) | 0/74 | 0/57 | 0/0 |
| arkit16 | MoGe-2 metric ViT-L/affine | 0.3-0.8m | 0/44 | 0/44 | 0/96 | 0/4 | 0/0 | 0/0 |
| arkit16 | MoGe-2 metric ViT-L/affine | 0.8-1.5m | 45/114 | 46/114 | 0/9 | 0/21 | 39/29 | 0/5 |
| arkit16 | MoGe-2 metric ViT-L/affine | 1.5-3m | 52/70 | 63/70 | 0/0 (N/E) | 9/74 | 0/16 | 0/0 |
| arkit_40777060 | MoGe-2 metric ViT-L/raw | 0.3-0.8m | 0/11 | 0/11 | 0/133 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777060 | MoGe-2 metric ViT-L/raw | 0.8-1.5m | 4/69 | 7/69 | 3/75 | 0/0 (N/E) | 4/0 | 3/0 |
| arkit_40777060 | MoGe-2 metric ViT-L/raw | 1.5-3m | 50/122 | 57/122 | 8/12 | 1/10 | 0/70 | 0/4 |
| arkit_40777060 | MoGe-2 metric ViT-L/affine | 0.3-0.8m | 0/11 | 0/11 | 0/133 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777060 | MoGe-2 metric ViT-L/affine | 0.8-1.5m | 42/69 | 46/69 | 4/75 | 0/0 (N/E) | 42/0 | 4/0 |
| arkit_40777060 | MoGe-2 metric ViT-L/affine | 1.5-3m | 109/122 | 111/122 | 7/12 | 3/10 | 0/11 | 0/5 |
| arkit_40777065 | MoGe-2 metric ViT-L/raw | 0.3-0.8m | 0/0 (N/E) | 0/0 (N/E) | 0/144 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777065 | MoGe-2 metric ViT-L/raw | 0.8-1.5m | 9/48 | 16/48 | 11/93 | 1/3 | 9/0 | 11/0 |
| arkit_40777065 | MoGe-2 metric ViT-L/raw | 1.5-3m | 28/131 | 43/131 | 10/10 | 3/3 | 0/103 | 0/0 |
| arkit_40777065 | MoGe-2 metric ViT-L/affine | 0.3-0.8m | 0/0 (N/E) | 0/0 (N/E) | 0/144 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777065 | MoGe-2 metric ViT-L/affine | 0.8-1.5m | 38/48 | 39/48 | 19/93 | 2/3 | 38/0 | 19/0 |
| arkit_40777065 | MoGe-2 metric ViT-L/affine | 1.5-3m | 122/131 | 127/131 | 8/10 | 3/3 | 0/9 | 0/2 |

## 原生像素近/中/远诊断（eval ARKit逐capture）

GT native optical-Z选像素带；排除LiDAR缺失及预测非有限/非正并单列缺失计数，不裁掉误差尾。cal拟合内、所有3RScan capture、missing逐帧分母、全q0/1/5/50/95/99/100在pixel_metrics.json、pixel_frame_metrics.csv及完整payload表。

| Capture | 主干/臂 | GT带 | 像素数 | meanAbsLog | medianAbsRel | AbsLog p99 | AbsRel p99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arkitscenes_47333462 | DAV2 Indoor Large/raw | 0.3-0.8m | 150717 | 0.604055 | 0.93253 | 0.703616 | 1.02105 |
| arkitscenes_47333462 | DAV2 Indoor Large/affine | 0.3-0.8m | 150717 | 0.66817 | 1.02551 | 0.783978 | 1.19017 |
| arkitscenes_47333462 | DepthPro affine | 0.3-0.8m | 150717 | 1.34625 | 2.73465 | 4.06055 | 57.0065 |
| arkitscenes_47333462 | DAV2 Indoor Large/raw | 0.8-1.5m | 465115 | 0.308825 | 0.350007 | 0.563605 | 0.756995 |
| arkitscenes_47333462 | DAV2 Indoor Large/affine | 0.8-1.5m | 465115 | 0.237085 | 0.260776 | 0.436537 | 0.547339 |
| arkitscenes_47333462 | DepthPro affine | 0.8-1.5m | 465115 | 0.509144 | 0.545223 | 1.60251 | 3.96547 |
| arkitscenes_47333462 | DAV2 Indoor Large/raw | 1.5-3m | 163946 | 0.271991 | 0.301566 | 0.51521 | 0.673991 |
| arkitscenes_47333462 | DAV2 Indoor Large/affine | 1.5-3m | 163946 | 0.0821677 | 0.0611499 | 0.456004 | 0.366433 |
| arkitscenes_47333462 | DepthPro affine | 1.5-3m | 163946 | 0.0825038 | 0.0683025 | 0.522386 | 0.406971 |
| arkitscenes_40777060 | DAV2 Indoor Large/raw | 0.3-0.8m | 10243 | 0.192784 | 0.0953273 | 0.661947 | 0.938562 |
| arkitscenes_40777060 | DAV2 Indoor Large/affine | 0.3-0.8m | 10243 | 0.288562 | 0.246559 | 0.613238 | 0.8464 |
| arkitscenes_40777060 | DepthPro affine | 0.3-0.8m | 10243 | 2.05029 | 3.33014 | 3.97537 | 52.2696 |
| arkitscenes_40777060 | DAV2 Indoor Large/raw | 0.8-1.5m | 149651 | 0.188674 | 0.122126 | 0.619633 | 0.858245 |
| arkitscenes_40777060 | DAV2 Indoor Large/affine | 0.8-1.5m | 149651 | 0.142883 | 0.124395 | 0.486712 | 0.626958 |
| arkitscenes_40777060 | DepthPro affine | 0.8-1.5m | 149651 | 1.01142 | 1.09737 | 3.87484 | 47.1748 |
| arkitscenes_40777060 | DAV2 Indoor Large/raw | 1.5-3m | 498375 | 0.17446 | 0.155273 | 0.452931 | 0.568922 |
| arkitscenes_40777060 | DAV2 Indoor Large/affine | 1.5-3m | 498375 | 0.130954 | 0.106028 | 0.409887 | 0.336852 |
| arkitscenes_40777060 | DepthPro affine | 1.5-3m | 498375 | 0.185418 | 0.170729 | 0.569216 | 0.766881 |
| arkitscenes_40777065 | DAV2 Indoor Large/raw | 0.3-0.8m | 0 | N/E | N/E | N/E | N/E |
| arkitscenes_40777065 | DAV2 Indoor Large/affine | 0.3-0.8m | 0 | N/E | N/E | N/E | N/E |
| arkitscenes_40777065 | DepthPro affine | 0.3-0.8m | 0 | N/E | N/E | N/E | N/E |
| arkitscenes_40777065 | DAV2 Indoor Large/raw | 0.8-1.5m | 82270 | 0.194979 | 0.11769 | 1.05745 | 1.87901 |
| arkitscenes_40777065 | DAV2 Indoor Large/affine | 0.8-1.5m | 82270 | 0.119857 | 0.0813707 | 0.615437 | 0.850466 |
| arkitscenes_40777065 | DepthPro affine | 0.8-1.5m | 82270 | 0.526129 | 0.676903 | 0.867368 | 1.38064 |
| arkitscenes_40777065 | DAV2 Indoor Large/raw | 1.5-3m | 568585 | 0.146075 | 0.123953 | 0.456417 | 0.532397 |
| arkitscenes_40777065 | DAV2 Indoor Large/affine | 1.5-3m | 568585 | 0.211053 | 0.168144 | 0.495577 | 0.391567 |
| arkitscenes_40777065 | DepthPro affine | 1.5-3m | 568585 | 0.218559 | 0.207731 | 0.521826 | 0.685102 |
| arkitscenes_47333462 | UniDepth V2 ViT-L/raw | 0.3-0.8m | 150717 | 0.158485 | 0.174025 | 0.314105 | 0.369033 |
| arkitscenes_47333462 | UniDepth V2 ViT-L/affine | 0.3-0.8m | 150717 | 0.314749 | 0.379179 | 0.454043 | 0.574666 |
| arkitscenes_47333462 | UniDepth V2 ViT-L/raw | 0.8-1.5m | 465115 | 0.206159 | 0.202808 | 0.483401 | 0.605296 |
| arkitscenes_47333462 | UniDepth V2 ViT-L/affine | 0.8-1.5m | 465115 | 0.248453 | 0.261078 | 0.536241 | 0.709568 |
| arkitscenes_47333462 | UniDepth V2 ViT-L/raw | 1.5-3m | 163946 | 0.147702 | 0.126953 | 0.717261 | 0.51569 |
| arkitscenes_47333462 | UniDepth V2 ViT-L/affine | 1.5-3m | 163946 | 0.15994 | 0.14121 | 0.632439 | 0.500345 |
| arkitscenes_40777060 | UniDepth V2 ViT-L/raw | 0.3-0.8m | 10243 | 0.498256 | 0.720276 | 0.865421 | 1.37601 |
| arkitscenes_40777060 | UniDepth V2 ViT-L/affine | 0.3-0.8m | 10243 | 0.583221 | 0.87001 | 0.899246 | 1.45775 |
| arkitscenes_40777060 | UniDepth V2 ViT-L/raw | 0.8-1.5m | 149651 | 0.217872 | 0.146055 | 0.784644 | 1.19163 |
| arkitscenes_40777060 | UniDepth V2 ViT-L/affine | 0.8-1.5m | 149651 | 0.258428 | 0.22458 | 0.813939 | 1.25678 |
| arkitscenes_40777060 | UniDepth V2 ViT-L/raw | 1.5-3m | 498375 | 0.156859 | 0.121377 | 0.515293 | 0.668545 |
| arkitscenes_40777060 | UniDepth V2 ViT-L/affine | 1.5-3m | 498375 | 0.144775 | 0.111741 | 0.482371 | 0.615628 |
| arkitscenes_40777065 | UniDepth V2 ViT-L/raw | 0.3-0.8m | 0 | N/E | N/E | N/E | N/E |
| arkitscenes_40777065 | UniDepth V2 ViT-L/affine | 0.3-0.8m | 0 | N/E | N/E | N/E | N/E |
| arkitscenes_40777065 | UniDepth V2 ViT-L/raw | 0.8-1.5m | 82270 | 0.141932 | 0.0835142 | 1.16021 | 2.19062 |
| arkitscenes_40777065 | UniDepth V2 ViT-L/affine | 0.8-1.5m | 82270 | 0.143814 | 0.0718047 | 1.07233 | 1.92219 |
| arkitscenes_40777065 | UniDepth V2 ViT-L/raw | 1.5-3m | 568585 | 0.128653 | 0.090957 | 0.455317 | 0.518838 |
| arkitscenes_40777065 | UniDepth V2 ViT-L/affine | 1.5-3m | 568585 | 0.113189 | 0.0801348 | 0.420206 | 0.491489 |
| arkitscenes_47333462 | Metric3D v2 ViT-L/raw | 0.3-0.8m | 150717 | 0.301353 | 0.406715 | 0.400798 | 0.49301 |
| arkitscenes_47333462 | Metric3D v2 ViT-L/affine | 0.3-0.8m | 150717 | 0.580585 | 0.84095 | 0.674963 | 0.963958 |
| arkitscenes_47333462 | Metric3D v2 ViT-L/raw | 0.8-1.5m | 465115 | 0.150863 | 0.147719 | 0.389371 | 0.475945 |
| arkitscenes_47333462 | Metric3D v2 ViT-L/affine | 0.8-1.5m | 465115 | 0.350117 | 0.402164 | 0.599209 | 0.820461 |
| arkitscenes_47333462 | Metric3D v2 ViT-L/raw | 1.5-3m | 163946 | 0.167474 | 0.0954204 | 1.56186 | 3.65058 |
| arkitscenes_47333462 | Metric3D v2 ViT-L/affine | 1.5-3m | 163946 | 0.262915 | 0.267995 | 1.4553 | 3.24263 |
| arkitscenes_40777060 | Metric3D v2 ViT-L/raw | 0.3-0.8m | 10243 | 0.164299 | 0.0501544 | 0.402617 | 0.48717 |
| arkitscenes_40777060 | Metric3D v2 ViT-L/affine | 0.3-0.8m | 10243 | 0.295327 | 0.320484 | 0.629637 | 0.87693 |
| arkitscenes_40777060 | Metric3D v2 ViT-L/raw | 0.8-1.5m | 149651 | 0.226957 | 0.184129 | 0.544303 | 0.559326 |
| arkitscenes_40777060 | Metric3D v2 ViT-L/affine | 0.8-1.5m | 149651 | 0.138259 | 0.0930317 | 0.612344 | 0.83634 |
| arkitscenes_40777060 | Metric3D v2 ViT-L/raw | 1.5-3m | 498375 | 0.18758 | 0.165125 | 0.514096 | 0.411635 |
| arkitscenes_40777060 | Metric3D v2 ViT-L/affine | 1.5-3m | 498375 | 0.0910808 | 0.070388 | 0.369746 | 0.387421 |
| arkitscenes_40777065 | Metric3D v2 ViT-L/raw | 0.3-0.8m | 0 | N/E | N/E | N/E | N/E |
| arkitscenes_40777065 | Metric3D v2 ViT-L/affine | 0.3-0.8m | 0 | N/E | N/E | N/E | N/E |
| arkitscenes_40777065 | Metric3D v2 ViT-L/raw | 0.8-1.5m | 82270 | 0.215352 | 0.187753 | 0.566398 | 0.527343 |
| arkitscenes_40777065 | Metric3D v2 ViT-L/affine | 0.8-1.5m | 82270 | 0.202153 | 0.159798 | 0.554278 | 0.73185 |
| arkitscenes_40777065 | Metric3D v2 ViT-L/raw | 1.5-3m | 568585 | 0.238932 | 0.201567 | 0.597863 | 0.453977 |
| arkitscenes_40777065 | Metric3D v2 ViT-L/affine | 1.5-3m | 568585 | 0.146417 | 0.123781 | 0.42003 | 0.419849 |
| arkitscenes_47333462 | MoGe-2 metric ViT-L/raw | 0.3-0.8m | 150717 | 0.0292248 | 0.0216784 | 0.152556 | 0.149122 |
| arkitscenes_47333462 | MoGe-2 metric ViT-L/affine | 0.3-0.8m | 150717 | 0.285421 | 0.327739 | 0.385261 | 0.469997 |
| arkitscenes_47333462 | MoGe-2 metric ViT-L/raw | 0.8-1.5m | 465115 | 0.117446 | 0.0574533 | 0.716668 | 0.514141 |
| arkitscenes_47333462 | MoGe-2 metric ViT-L/affine | 0.8-1.5m | 465115 | 0.2484 | 0.275444 | 0.440742 | 0.463872 |
| arkitscenes_47333462 | MoGe-2 metric ViT-L/raw | 1.5-3m | 163946 | 0.0876103 | 0.0379463 | 0.739329 | 0.522566 |
| arkitscenes_47333462 | MoGe-2 metric ViT-L/affine | 1.5-3m | 163946 | 0.186917 | 0.197899 | 0.511048 | 0.438729 |
| arkitscenes_40777060 | MoGe-2 metric ViT-L/raw | 0.3-0.8m | 10243 | 0.179065 | 0.127139 | 0.32406 | 0.382612 |
| arkitscenes_40777060 | MoGe-2 metric ViT-L/affine | 0.3-0.8m | 10243 | 0.215989 | 0.173737 | 0.56153 | 0.753353 |
| arkitscenes_40777060 | MoGe-2 metric ViT-L/raw | 0.8-1.5m | 149651 | 0.225463 | 0.19854 | 0.570863 | 0.435254 |
| arkitscenes_40777060 | MoGe-2 metric ViT-L/affine | 0.8-1.5m | 149651 | 0.123691 | 0.0994551 | 0.463223 | 0.589187 |
| arkitscenes_40777060 | MoGe-2 metric ViT-L/raw | 1.5-3m | 498375 | 0.292505 | 0.273571 | 0.636932 | 0.471522 |
| arkitscenes_40777060 | MoGe-2 metric ViT-L/affine | 1.5-3m | 498375 | 0.147595 | 0.137943 | 0.405536 | 0.335789 |
| arkitscenes_40777065 | MoGe-2 metric ViT-L/raw | 0.3-0.8m | 0 | N/E | N/E | N/E | N/E |
| arkitscenes_40777065 | MoGe-2 metric ViT-L/affine | 0.3-0.8m | 0 | N/E | N/E | N/E | N/E |
| arkitscenes_40777065 | MoGe-2 metric ViT-L/raw | 0.8-1.5m | 82270 | 0.260399 | 0.222269 | 0.563619 | 0.431927 |
| arkitscenes_40777065 | MoGe-2 metric ViT-L/affine | 0.8-1.5m | 82270 | 0.0855843 | 0.0656422 | 0.397296 | 0.487576 |
| arkitscenes_40777065 | MoGe-2 metric ViT-L/raw | 1.5-3m | 568585 | 0.379355 | 0.316882 | 0.649503 | 0.477695 |
| arkitscenes_40777065 | MoGe-2 metric ViT-L/affine | 1.5-3m | 568585 | 0.179985 | 0.159344 | 0.432281 | 0.351551 |

## Pooled cal补充：全部五cohort×距离带

所有cal切点先封存；原validation24 FREE只有2、新3RScan64只有1，零错误的负例检验力很弱。完整capture细分与cal拟合内结果都在report-summary/full_tables.md，不以pooled结果覆盖主三折。

| Cohort | 主干/臂 | 带 | W/POS | S/POS | F/FREE | U/UNKNOWN | 救/损 | FREE增/减 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_validation24 | DAV2 Indoor Large/raw | 0.3-0.8m | 1/24 | 1/24 | 0/0 (N/E) | 1/192 | 1/3 | 0/0 |
| original_validation24 | DAV2 Indoor Large/raw | 0.8-1.5m | 70/137 | 78/137 | 0/2 | 25/77 | 15/62 | 0/1 |
| original_validation24 | DAV2 Indoor Large/raw | 1.5-3m | 46/134 | 50/134 | 0/0 (N/E) | 12/82 | 3/56 | 0/0 |
| original_validation24 | DAV2 Indoor Large/affine | 0.3-0.8m | 0/24 | 0/24 | 0/0 (N/E) | 0/192 | 0/3 | 0/0 |
| original_validation24 | DAV2 Indoor Large/affine | 0.8-1.5m | 84/137 | 92/137 | 0/2 | 29/77 | 15/48 | 0/1 |
| original_validation24 | DAV2 Indoor Large/affine | 1.5-3m | 35/134 | 49/134 | 0/0 (N/E) | 9/82 | 0/64 | 0/0 |
| original_validation24 | DepthPro affine | 0.3-0.8m | 3/24 | 3/24 | 0/0 (N/E) | 8/192 | -/- | -/- |
| original_validation24 | DepthPro affine | 0.8-1.5m | 117/137 | 122/137 | 1/2 | 53/77 | -/- | -/- |
| original_validation24 | DepthPro affine | 1.5-3m | 99/134 | 109/134 | 0/0 (N/E) | 25/82 | -/- | -/- |
| new_3rscan64 | DAV2 Indoor Large/raw | 0.3-0.8m | 4/168 | 4/168 | 0/0 (N/E) | 0/408 | 4/3 | 0/0 |
| new_3rscan64 | DAV2 Indoor Large/raw | 0.8-1.5m | 178/428 | 222/428 | 0/0 (N/E) | 37/148 | 18/203 | 0/0 |
| new_3rscan64 | DAV2 Indoor Large/raw | 1.5-3m | 154/270 | 169/270 | 1/1 | 58/305 | 18/68 | 0/0 |
| new_3rscan64 | DAV2 Indoor Large/affine | 0.3-0.8m | 1/168 | 1/168 | 0/0 (N/E) | 0/408 | 1/3 | 0/0 |
| new_3rscan64 | DAV2 Indoor Large/affine | 0.8-1.5m | 210/428 | 252/428 | 0/0 (N/E) | 50/148 | 26/179 | 0/0 |
| new_3rscan64 | DAV2 Indoor Large/affine | 1.5-3m | 121/270 | 138/270 | 1/1 | 39/305 | 6/89 | 0/0 |
| new_3rscan64 | DepthPro affine | 0.3-0.8m | 3/168 | 5/168 | 0/0 (N/E) | 9/408 | -/- | -/- |
| new_3rscan64 | DepthPro affine | 0.8-1.5m | 363/428 | 378/428 | 0/0 (N/E) | 92/148 | -/- | -/- |
| new_3rscan64 | DepthPro affine | 1.5-3m | 204/270 | 213/270 | 1/1 | 121/305 | -/- | -/- |
| arkit16 | DAV2 Indoor Large/raw | 0.3-0.8m | 0/44 | 0/44 | 0/96 | 0/4 | 0/0 | 0/0 |
| arkit16 | DAV2 Indoor Large/raw | 0.8-1.5m | 32/114 | 36/114 | 0/9 | 17/21 | 27/24 | 0/5 |
| arkit16 | DAV2 Indoor Large/raw | 1.5-3m | 52/70 | 64/70 | 0/0 (N/E) | 11/74 | 0/14 | 0/0 |
| arkit16 | DAV2 Indoor Large/affine | 0.3-0.8m | 0/44 | 0/44 | 0/96 | 0/4 | 0/0 | 0/0 |
| arkit16 | DAV2 Indoor Large/affine | 0.8-1.5m | 31/114 | 36/114 | 0/9 | 20/21 | 26/24 | 0/5 |
| arkit16 | DAV2 Indoor Large/affine | 1.5-3m | 47/70 | 50/70 | 0/0 (N/E) | 2/74 | 0/19 | 0/0 |
| arkit16 | DepthPro affine | 0.3-0.8m | 0/44 | 0/44 | 0/96 | 0/4 | -/- | -/- |
| arkit16 | DepthPro affine | 0.8-1.5m | 29/114 | 30/114 | 5/9 | 0/21 | -/- | -/- |
| arkit16 | DepthPro affine | 1.5-3m | 66/70 | 70/70 | 0/0 (N/E) | 74/74 | -/- | -/- |
| arkit_40777060 | DAV2 Indoor Large/raw | 0.3-0.8m | 0/11 | 0/11 | 0/133 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777060 | DAV2 Indoor Large/raw | 0.8-1.5m | 21/69 | 22/69 | 0/75 | 0/0 (N/E) | 21/0 | 0/0 |
| arkit_40777060 | DAV2 Indoor Large/raw | 1.5-3m | 92/122 | 99/122 | 0/12 | 1/10 | 0/28 | 0/12 |
| arkit_40777060 | DAV2 Indoor Large/affine | 0.3-0.8m | 0/11 | 0/11 | 0/133 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777060 | DAV2 Indoor Large/affine | 0.8-1.5m | 23/69 | 24/69 | 0/75 | 0/0 (N/E) | 23/0 | 0/0 |
| arkit_40777060 | DAV2 Indoor Large/affine | 1.5-3m | 105/122 | 109/122 | 8/12 | 1/10 | 0/15 | 0/4 |
| arkit_40777060 | DepthPro affine | 0.3-0.8m | 0/11 | 0/11 | 0/133 | 0/0 (N/E) | -/- | -/- |
| arkit_40777060 | DepthPro affine | 0.8-1.5m | 0/69 | 0/69 | 0/75 | 0/0 (N/E) | -/- | -/- |
| arkit_40777060 | DepthPro affine | 1.5-3m | 120/122 | 122/122 | 12/12 | 9/10 | -/- | -/- |
| arkit_40777065 | DAV2 Indoor Large/raw | 0.3-0.8m | 0/0 (N/E) | 0/0 (N/E) | 0/144 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777065 | DAV2 Indoor Large/raw | 0.8-1.5m | 14/48 | 15/48 | 3/93 | 0/3 | 14/0 | 3/0 |
| arkit_40777065 | DAV2 Indoor Large/raw | 1.5-3m | 110/131 | 115/131 | 3/10 | 2/3 | 0/21 | 0/7 |
| arkit_40777065 | DAV2 Indoor Large/affine | 0.3-0.8m | 0/0 (N/E) | 0/0 (N/E) | 0/144 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777065 | DAV2 Indoor Large/affine | 0.8-1.5m | 14/48 | 15/48 | 3/93 | 0/3 | 14/0 | 3/0 |
| arkit_40777065 | DAV2 Indoor Large/affine | 1.5-3m | 105/131 | 112/131 | 10/10 | 3/3 | 0/26 | 0/0 |
| arkit_40777065 | DepthPro affine | 0.3-0.8m | 0/0 (N/E) | 0/0 (N/E) | 0/144 | 0/0 (N/E) | -/- | -/- |
| arkit_40777065 | DepthPro affine | 0.8-1.5m | 0/48 | 0/48 | 0/93 | 0/3 | -/- | -/- |
| arkit_40777065 | DepthPro affine | 1.5-3m | 131/131 | 131/131 | 10/10 | 3/3 | -/- | -/- |
| original_validation24 | UniDepth V2 ViT-L/raw | 0.3-0.8m | 5/24 | 7/24 | 0/0 (N/E) | 17/192 | 4/2 | 0/0 |
| original_validation24 | UniDepth V2 ViT-L/raw | 0.8-1.5m | 105/137 | 111/137 | 0/2 | 31/77 | 17/29 | 0/1 |
| original_validation24 | UniDepth V2 ViT-L/raw | 1.5-3m | 70/134 | 77/134 | 0/0 (N/E) | 7/82 | 2/31 | 0/0 |
| original_validation24 | UniDepth V2 ViT-L/affine | 0.3-0.8m | 8/24 | 9/24 | 0/0 (N/E) | 13/192 | 6/1 | 0/0 |
| original_validation24 | UniDepth V2 ViT-L/affine | 0.8-1.5m | 120/137 | 124/137 | 0/2 | 37/77 | 17/14 | 0/1 |
| original_validation24 | UniDepth V2 ViT-L/affine | 1.5-3m | 89/134 | 101/134 | 0/0 (N/E) | 22/82 | 7/17 | 0/0 |
| new_3rscan64 | UniDepth V2 ViT-L/raw | 0.3-0.8m | 44/168 | 67/168 | 0/0 (N/E) | 13/408 | 43/2 | 0/0 |
| new_3rscan64 | UniDepth V2 ViT-L/raw | 0.8-1.5m | 317/428 | 352/428 | 0/0 (N/E) | 35/148 | 43/89 | 0/0 |
| new_3rscan64 | UniDepth V2 ViT-L/raw | 1.5-3m | 184/270 | 200/270 | 1/1 | 35/305 | 23/43 | 0/0 |
| new_3rscan64 | UniDepth V2 ViT-L/affine | 0.3-0.8m | 76/168 | 90/168 | 0/0 (N/E) | 30/408 | 74/1 | 0/0 |
| new_3rscan64 | UniDepth V2 ViT-L/affine | 0.8-1.5m | 366/428 | 390/428 | 0/0 (N/E) | 63/148 | 46/43 | 0/0 |
| new_3rscan64 | UniDepth V2 ViT-L/affine | 1.5-3m | 217/270 | 231/270 | 1/1 | 59/305 | 37/24 | 0/0 |
| arkit16 | UniDepth V2 ViT-L/raw | 0.3-0.8m | 19/44 | 20/44 | 0/96 | 0/4 | 19/0 | 0/0 |
| arkit16 | UniDepth V2 ViT-L/raw | 0.8-1.5m | 69/114 | 70/114 | 0/9 | 0/21 | 50/10 | 0/5 |
| arkit16 | UniDepth V2 ViT-L/raw | 1.5-3m | 56/70 | 65/70 | 0/0 (N/E) | 13/74 | 0/10 | 0/0 |
| arkit16 | UniDepth V2 ViT-L/affine | 0.3-0.8m | 17/44 | 20/44 | 0/96 | 1/4 | 17/0 | 0/0 |
| arkit16 | UniDepth V2 ViT-L/affine | 0.8-1.5m | 77/114 | 80/114 | 0/9 | 9/21 | 59/11 | 0/5 |
| arkit16 | UniDepth V2 ViT-L/affine | 1.5-3m | 62/70 | 70/70 | 0/0 (N/E) | 30/74 | 1/5 | 0/0 |
| arkit_40777060 | UniDepth V2 ViT-L/raw | 0.3-0.8m | 0/11 | 0/11 | 0/133 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777060 | UniDepth V2 ViT-L/raw | 0.8-1.5m | 30/69 | 31/69 | 3/75 | 0/0 (N/E) | 30/0 | 3/0 |
| arkit_40777060 | UniDepth V2 ViT-L/raw | 1.5-3m | 100/122 | 103/122 | 0/12 | 5/10 | 0/20 | 0/12 |
| arkit_40777060 | UniDepth V2 ViT-L/affine | 0.3-0.8m | 0/11 | 0/11 | 0/133 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777060 | UniDepth V2 ViT-L/affine | 0.8-1.5m | 30/69 | 30/69 | 1/75 | 0/0 (N/E) | 30/0 | 1/0 |
| arkit_40777060 | UniDepth V2 ViT-L/affine | 1.5-3m | 111/122 | 112/122 | 2/12 | 5/10 | 2/11 | 0/10 |
| arkit_40777065 | UniDepth V2 ViT-L/raw | 0.3-0.8m | 0/0 (N/E) | 0/0 (N/E) | 0/144 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777065 | UniDepth V2 ViT-L/raw | 0.8-1.5m | 27/48 | 27/48 | 7/93 | 0/3 | 27/0 | 7/0 |
| arkit_40777065 | UniDepth V2 ViT-L/raw | 1.5-3m | 120/131 | 121/131 | 0/10 | 0/3 | 0/11 | 0/10 |
| arkit_40777065 | UniDepth V2 ViT-L/affine | 0.3-0.8m | 0/0 (N/E) | 0/0 (N/E) | 0/144 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777065 | UniDepth V2 ViT-L/affine | 0.8-1.5m | 27/48 | 27/48 | 7/93 | 0/3 | 27/0 | 7/0 |
| arkit_40777065 | UniDepth V2 ViT-L/affine | 1.5-3m | 131/131 | 131/131 | 3/10 | 2/3 | 0/0 | 0/7 |
| original_validation24 | Metric3D v2 ViT-L/raw | 0.3-0.8m | 0/24 | 0/24 | 0/0 (N/E) | 0/192 | 0/3 | 0/0 |
| original_validation24 | Metric3D v2 ViT-L/raw | 0.8-1.5m | 22/137 | 35/137 | 0/2 | 5/77 | 3/98 | 0/1 |
| original_validation24 | Metric3D v2 ViT-L/raw | 1.5-3m | 19/134 | 27/134 | 0/0 (N/E) | 8/82 | 0/80 | 0/0 |
| original_validation24 | Metric3D v2 ViT-L/affine | 0.3-0.8m | 1/24 | 1/24 | 0/0 (N/E) | 5/192 | 1/3 | 0/0 |
| original_validation24 | Metric3D v2 ViT-L/affine | 0.8-1.5m | 93/137 | 104/137 | 0/2 | 27/77 | 11/35 | 0/1 |
| original_validation24 | Metric3D v2 ViT-L/affine | 1.5-3m | 88/134 | 96/134 | 0/0 (N/E) | 15/82 | 13/24 | 0/0 |
| new_3rscan64 | Metric3D v2 ViT-L/raw | 0.3-0.8m | 0/168 | 0/168 | 0/0 (N/E) | 0/408 | 0/3 | 0/0 |
| new_3rscan64 | Metric3D v2 ViT-L/raw | 0.8-1.5m | 60/428 | 109/428 | 0/0 (N/E) | 11/148 | 11/314 | 0/0 |
| new_3rscan64 | Metric3D v2 ViT-L/raw | 1.5-3m | 69/270 | 80/270 | 1/1 | 10/305 | 0/135 | 0/0 |
| new_3rscan64 | Metric3D v2 ViT-L/affine | 0.3-0.8m | 13/168 | 22/168 | 0/0 (N/E) | 4/408 | 11/1 | 0/0 |
| new_3rscan64 | Metric3D v2 ViT-L/affine | 0.8-1.5m | 305/428 | 332/428 | 0/0 (N/E) | 42/148 | 26/84 | 0/0 |
| new_3rscan64 | Metric3D v2 ViT-L/affine | 1.5-3m | 186/270 | 201/270 | 1/1 | 50/305 | 27/45 | 0/0 |
| arkit16 | Metric3D v2 ViT-L/raw | 0.3-0.8m | 0/44 | 0/44 | 0/96 | 0/4 | 0/0 | 0/0 |
| arkit16 | Metric3D v2 ViT-L/raw | 0.8-1.5m | 23/114 | 25/114 | 0/9 | 0/21 | 21/27 | 0/5 |
| arkit16 | Metric3D v2 ViT-L/raw | 1.5-3m | 25/70 | 29/70 | 0/0 (N/E) | 2/74 | 0/41 | 0/0 |
| arkit16 | Metric3D v2 ViT-L/affine | 0.3-0.8m | 0/44 | 0/44 | 0/96 | 0/4 | 0/0 | 0/0 |
| arkit16 | Metric3D v2 ViT-L/affine | 0.8-1.5m | 31/114 | 34/114 | 0/9 | 20/21 | 29/27 | 0/5 |
| arkit16 | Metric3D v2 ViT-L/affine | 1.5-3m | 54/70 | 69/70 | 0/0 (N/E) | 29/74 | 0/12 | 0/0 |
| arkit_40777060 | Metric3D v2 ViT-L/raw | 0.3-0.8m | 0/11 | 0/11 | 0/133 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777060 | Metric3D v2 ViT-L/raw | 0.8-1.5m | 18/69 | 23/69 | 3/75 | 0/0 (N/E) | 18/0 | 3/0 |
| arkit_40777060 | Metric3D v2 ViT-L/raw | 1.5-3m | 67/122 | 73/122 | 6/12 | 1/10 | 0/53 | 0/6 |
| arkit_40777060 | Metric3D v2 ViT-L/affine | 0.3-0.8m | 0/11 | 0/11 | 0/133 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777060 | Metric3D v2 ViT-L/affine | 0.8-1.5m | 32/69 | 35/69 | 0/75 | 0/0 (N/E) | 32/0 | 0/0 |
| arkit_40777060 | Metric3D v2 ViT-L/affine | 1.5-3m | 118/122 | 121/122 | 6/12 | 2/10 | 0/2 | 0/6 |
| arkit_40777065 | Metric3D v2 ViT-L/raw | 0.3-0.8m | 0/0 (N/E) | 0/0 (N/E) | 0/144 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777065 | Metric3D v2 ViT-L/raw | 0.8-1.5m | 6/48 | 14/48 | 7/93 | 0/3 | 6/0 | 7/0 |
| arkit_40777065 | Metric3D v2 ViT-L/raw | 1.5-3m | 80/131 | 88/131 | 4/10 | 3/3 | 0/51 | 0/6 |
| arkit_40777065 | Metric3D v2 ViT-L/affine | 0.3-0.8m | 0/0 (N/E) | 0/0 (N/E) | 0/144 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777065 | Metric3D v2 ViT-L/affine | 0.8-1.5m | 14/48 | 14/48 | 5/93 | 1/3 | 14/0 | 5/0 |
| arkit_40777065 | Metric3D v2 ViT-L/affine | 1.5-3m | 125/131 | 125/131 | 4/10 | 3/3 | 0/6 | 0/6 |
| original_validation24 | MoGe-2 metric ViT-L/raw | 0.3-0.8m | 0/24 | 0/24 | 0/0 (N/E) | 0/192 | 0/3 | 0/0 |
| original_validation24 | MoGe-2 metric ViT-L/raw | 0.8-1.5m | 15/137 | 31/137 | 0/2 | 5/77 | 1/103 | 0/1 |
| original_validation24 | MoGe-2 metric ViT-L/raw | 1.5-3m | 12/134 | 16/134 | 0/0 (N/E) | 2/82 | 0/87 | 0/0 |
| original_validation24 | MoGe-2 metric ViT-L/affine | 0.3-0.8m | 2/24 | 3/24 | 0/0 (N/E) | 2/192 | 1/2 | 0/0 |
| original_validation24 | MoGe-2 metric ViT-L/affine | 0.8-1.5m | 101/137 | 113/137 | 0/2 | 37/77 | 13/29 | 0/1 |
| original_validation24 | MoGe-2 metric ViT-L/affine | 1.5-3m | 65/134 | 74/134 | 0/0 (N/E) | 10/82 | 3/37 | 0/0 |
| new_3rscan64 | MoGe-2 metric ViT-L/raw | 0.3-0.8m | 0/168 | 0/168 | 0/0 (N/E) | 0/408 | 0/3 | 0/0 |
| new_3rscan64 | MoGe-2 metric ViT-L/raw | 0.8-1.5m | 59/428 | 100/428 | 0/0 (N/E) | 12/148 | 9/313 | 0/0 |
| new_3rscan64 | MoGe-2 metric ViT-L/raw | 1.5-3m | 75/270 | 84/270 | 1/1 | 11/305 | 0/129 | 0/0 |
| new_3rscan64 | MoGe-2 metric ViT-L/affine | 0.3-0.8m | 10/168 | 18/168 | 0/0 (N/E) | 1/408 | 9/2 | 0/0 |
| new_3rscan64 | MoGe-2 metric ViT-L/affine | 0.8-1.5m | 279/428 | 316/428 | 0/0 (N/E) | 47/148 | 32/116 | 0/0 |
| new_3rscan64 | MoGe-2 metric ViT-L/affine | 1.5-3m | 182/270 | 193/270 | 1/1 | 63/305 | 29/51 | 0/0 |
| arkit16 | MoGe-2 metric ViT-L/raw | 0.3-0.8m | 0/44 | 0/44 | 0/96 | 0/4 | 0/0 | 0/0 |
| arkit16 | MoGe-2 metric ViT-L/raw | 0.8-1.5m | 28/114 | 28/114 | 0/9 | 0/21 | 18/19 | 0/5 |
| arkit16 | MoGe-2 metric ViT-L/raw | 1.5-3m | 11/70 | 11/70 | 0/0 (N/E) | 0/74 | 0/55 | 0/0 |
| arkit16 | MoGe-2 metric ViT-L/affine | 0.3-0.8m | 0/44 | 1/44 | 0/96 | 1/4 | 0/0 | 0/0 |
| arkit16 | MoGe-2 metric ViT-L/affine | 0.8-1.5m | 50/114 | 50/114 | 0/9 | 0/21 | 43/22 | 0/5 |
| arkit16 | MoGe-2 metric ViT-L/affine | 1.5-3m | 56/70 | 66/70 | 0/0 (N/E) | 17/74 | 0/10 | 0/0 |
| arkit_40777060 | MoGe-2 metric ViT-L/raw | 0.3-0.8m | 0/11 | 0/11 | 0/133 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777060 | MoGe-2 metric ViT-L/raw | 0.8-1.5m | 13/69 | 24/69 | 12/75 | 0/0 (N/E) | 13/0 | 12/0 |
| arkit_40777060 | MoGe-2 metric ViT-L/raw | 1.5-3m | 50/122 | 57/122 | 8/12 | 1/10 | 0/70 | 0/4 |
| arkit_40777060 | MoGe-2 metric ViT-L/affine | 0.3-0.8m | 0/11 | 0/11 | 0/133 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777060 | MoGe-2 metric ViT-L/affine | 0.8-1.5m | 47/69 | 49/69 | 6/75 | 0/0 (N/E) | 47/0 | 6/0 |
| arkit_40777060 | MoGe-2 metric ViT-L/affine | 1.5-3m | 109/122 | 111/122 | 7/12 | 3/10 | 0/11 | 0/5 |
| arkit_40777065 | MoGe-2 metric ViT-L/raw | 0.3-0.8m | 0/0 (N/E) | 0/0 (N/E) | 0/144 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777065 | MoGe-2 metric ViT-L/raw | 0.8-1.5m | 13/48 | 26/48 | 15/93 | 2/3 | 13/0 | 15/0 |
| arkit_40777065 | MoGe-2 metric ViT-L/raw | 1.5-3m | 28/131 | 43/131 | 10/10 | 3/3 | 0/103 | 0/0 |
| arkit_40777065 | MoGe-2 metric ViT-L/affine | 0.3-0.8m | 0/0 (N/E) | 0/0 (N/E) | 0/144 | 0/0 (N/E) | 0/0 | 0/0 |
| arkit_40777065 | MoGe-2 metric ViT-L/affine | 0.8-1.5m | 35/48 | 36/48 | 15/93 | 2/3 | 35/0 | 15/0 |
| arkit_40777065 | MoGe-2 metric ViT-L/affine | 1.5-3m | 118/131 | 124/131 | 8/10 | 3/3 | 0/13 | 0/2 |

## 拟合与实际切点收据

| 主干 | a | b | train有效配对像素 | train帧 |
| --- | --- | --- | --- | --- |
| DAV2 Indoor Large | 0.5966844404605843 | 0.11245020924493154 | 1591791 | 56 |
| UniDepth V2 ViT-L | 0.8643321279538296 | 0.11211243305679827 | 1591791 | 56 |
| Metric3D v2 ViT-L | 0.8391956063599566 | 0.24989576197827823 | 1591791 | 56 |
| MoGe-2 metric ViT-L | 0.9087444144936451 | 0.24316348622064007 | 1591791 | 56 |

| 主干 | 校准角色 | 臂 | cutoff(m) | cal FREE支持 | cal FREE数 | 最大预算 |
| --- | --- | --- | --- | --- | --- | --- |
| DAV2 Indoor Large | pooled304 | raw | 0.244038 | 29 | 588 | 29 |
| DAV2 Indoor Large | pooled304 | affine | 0.221021 | 29 | 588 | 29 |
| DepthPro | pooled304 | depthpro_affine | -0.0810714 | 29 | 588 | 29 |
| DAV2 Indoor Large | arkit16 | raw | -0.0103754 | 23 | 467 | 23 |
| DAV2 Indoor Large | arkit16 | affine | 0.202061 | 23 | 467 | 23 |
| DepthPro | arkit16 | depthpro_affine | -0.123238 | 23 | 467 | 23 |
| DAV2 Indoor Large | arkit_40777060 | raw | -0.00864736 | 17 | 352 | 17 |
| DAV2 Indoor Large | arkit_40777060 | affine | 0.154852 | 17 | 352 | 17 |
| DepthPro | arkit_40777060 | depthpro_affine | -0.128097 | 17 | 352 | 17 |
| DAV2 Indoor Large | arkit_40777065 | raw | -0.135322 | 16 | 325 | 16 |
| DAV2 Indoor Large | arkit_40777065 | affine | 0.0662778 | 16 | 325 | 16 |
| DepthPro | arkit_40777065 | depthpro_affine | -0.0622055 | 16 | 325 | 16 |
| UniDepth V2 ViT-L | pooled304 | raw | 0.096161 | 29 | 588 | 29 |
| UniDepth V2 ViT-L | pooled304 | affine | 0.00539215 | 29 | 588 | 29 |
| UniDepth V2 ViT-L | arkit16 | raw | 0.00395322 | 23 | 467 | 23 |
| UniDepth V2 ViT-L | arkit16 | affine | -0.061674 | 23 | 467 | 23 |
| UniDepth V2 ViT-L | arkit_40777060 | raw | 0.0527946 | 17 | 352 | 17 |
| UniDepth V2 ViT-L | arkit_40777060 | affine | -0.0580255 | 17 | 352 | 17 |
| UniDepth V2 ViT-L | arkit_40777065 | raw | -0.0410674 | 16 | 325 | 16 |
| UniDepth V2 ViT-L | arkit_40777065 | affine | -0.123584 | 16 | 325 | 16 |
| Metric3D v2 ViT-L | pooled304 | raw | 0.342445 | 29 | 588 | 29 |
| Metric3D v2 ViT-L | pooled304 | affine | 0.122191 | 29 | 588 | 29 |
| Metric3D v2 ViT-L | arkit16 | raw | 0.320444 | 23 | 467 | 23 |
| Metric3D v2 ViT-L | arkit16 | affine | 0.0815194 | 23 | 467 | 23 |
| Metric3D v2 ViT-L | arkit_40777060 | raw | 0.271139 | 17 | 352 | 17 |
| Metric3D v2 ViT-L | arkit_40777060 | affine | 0.0740393 | 17 | 352 | 17 |
| Metric3D v2 ViT-L | arkit_40777065 | raw | 0.214989 | 16 | 325 | 16 |
| Metric3D v2 ViT-L | arkit_40777065 | affine | 0.0334085 | 16 | 325 | 16 |
| MoGe-2 metric ViT-L | pooled304 | raw | 0.34612 | 29 | 588 | 29 |
| MoGe-2 metric ViT-L | pooled304 | affine | 0.142833 | 29 | 588 | 29 |
| MoGe-2 metric ViT-L | arkit16 | raw | 0.34928 | 23 | 467 | 23 |
| MoGe-2 metric ViT-L | arkit16 | affine | 0.199378 | 23 | 467 | 23 |
| MoGe-2 metric ViT-L | arkit_40777060 | raw | 0.349034 | 17 | 352 | 17 |
| MoGe-2 metric ViT-L | arkit_40777060 | affine | 0.183021 | 17 | 352 | 17 |
| MoGe-2 metric ViT-L | arkit_40777065 | raw | 0.347927 | 16 | 325 | 16 |
| MoGe-2 metric ViT-L | arkit_40777065 | affine | 0.114293 | 16 | 325 | 16 |

## 官方版本、权重与许可证

权重只用于本地非商业研究，留在artifacts.local/models，不再分发。

### DAV2 Indoor Large

官方源：[仓库](https://github.com/DepthAnything/Depth-Anything-V2)；source SHA `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`。

权重：[固定版本下载](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/79720800638389a78b2defc92caa885104f69974/depth_anything_v2_metric_hypersim_vitl.pth)；revision `79720800638389a78b2defc92caa885104f69974`；SHA256 `6f82ff2bc543ac02ddff4aa31fa363676a8305dd3ccf04e80e2af115a044cb6d`。

许可证：CC-BY-NC-4.0 model Large; Apache-2.0 repository code。

### UniDepth V2 ViT-L

官方源：[仓库](https://github.com/lpiccinelli-eth/UniDepth)；source SHA `8d8cfe4c7ee15297099983607febf0d4f32eb3d6`。

权重：[固定版本下载](https://huggingface.co/lpiccinelli/unidepth-v2-vitl14/resolve/52b349b514bd8b47642f67ac78cb7b5dc5c51dd9/model.safetensors)；revision `52b349b514bd8b47642f67ac78cb7b5dc5c51dd9`；SHA256 `ba73d3de735302ccc64a50f1e557122050c4b1893e6060b28dba05d6af3e67c6`。

许可证：CC-BY-NC-4.0。

### Metric3D v2 ViT-L

官方源：[仓库](https://github.com/YvanYin/Metric3D)；source SHA `eb5b6fac0dc155e4e52f576e304fbf11655ff339`。

权重：[固定版本下载](https://huggingface.co/JUGGHM/Metric3D/resolve/80d2d1410afb4b23cd9d18c6be9144483d4b70b6/metric_depth_vit_large_800k.pth)；revision `80d2d1410afb4b23cd9d18c6be9144483d4b70b6`；SHA256 `15328ffc42b528b95f188687418f6f03b3f123eb34ccdbd686c112abbea6d972`。

许可证：BSD-2-Clause (code; official weight endpoint has no separate model card)。

### MoGe-2 metric ViT-L

官方源：[仓库](https://github.com/microsoft/MoGe)；source SHA `74fbce054ebed49800de42d0ad0e83495065719a`。

权重：[固定版本下载](https://huggingface.co/Ruicheng/moge-2-vitl-normal/resolve/cb0e8bbd6b1e243589717c78e750b1ba4c093acf/model.pt)；revision `cb0e8bbd6b1e243589717c78e750b1ba4c093acf`；SHA256 `280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01`。

许可证：MIT (HF model card); MIT code with Apache-2.0 DINOv2 code。

Metric3D官方HF没有独立model card/weight license声明；README明确code BSD-2-Clause并要求商业用途联系作者。本次记录这个证据缺口，按官方公开推理下载、本地非商业使用，不推导权重再分发权。原文LICENSE与README footer收据完整保存。MoGe pinned utils3d_moge只安装到本run/deps，共享venv不改；原生公式及依赖锁见model_adapter_provenance.json。DepthPro权重沿用原SHA256 `3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce`，其历史code/输入数值收据沿用INPUT_NUMERICS，不重新神经推理。

## GPU单帧延迟与参数量

RTX5060 Laptop、Torch2.11.0+cu130。计时同步GPU前后，包含官方预处理/推理/回CPU，native采样与缓存写盘在其外；全部495后续帧是两源不同尺寸的混合集，中位/P95仅部署参考。首帧含CUDA首次开销，单列。DepthPro本轮只CPU构建架构计参数，不加载权重、不推理；旧2.950s是单样本旧工作负载，不是本轮同496帧速度对照。

| 主干 | 参数 | 首帧s | 后续median s | P95 s | GPU进程墙钟s |
| --- | --- | --- | --- | --- | --- |
| DAV2 Indoor Large | 335315649 | 16.7658 | 0.349101 | 0.391575 | 201.904 |
| UniDepth V2 ViT-L | 353831043 | 17.1153 | 0.197053 | 0.229195 | 130.292 |
| Metric3D v2 ViT-L | 411941915 | 14.7938 | 0.680853 | 0.71412 | 378.402 |
| MoGe-2 metric ViT-L | 330901544 | 10.3707 | 0.300761 | 0.340539 | 183.97 |
| DepthPro（原缓存） | 951991330 | N/R | 旧样本2.950，不同workload | N/R | 0 |

## 独立复算、预算与结束状态

独立脚本不调用生产评分/汇总函数，四模型PASS：224个train帧重新IRLS、47,520条原DepthPro评分不变、108阈值、59,616决策、3,360分组、39,744逐query配对/CSV行、3,888原生16th分量（各ARKit first/last）、192帧身份、1,728像素分母/缺失、108全ARKit像素带组/25,066,824像素对及完整尾分位。fitSHA、cal封存时间、train/cal/eval身份隔离PASS；独立复算验证已保存预测，没有独立重跑神经推理。

CPU按command-wall保守计费 **1008/1800s**，含权重下载命令墙钟295.705s、评价含启动137.427s、主定向查读/集成/交付上界300s、适配器185s、协议/报告54s、独立20.4s、DepthPro架构15s。GPU全部拥有进程墙钟 **905.489/2400s**，包含首次Metric3D加载失败10.921s；四成功各496帧，没有partial子集。权重实际 **5,728,574,053B**，源/metadata/依赖按500,000,000B保守扣账（保留源码树301,136,261B），合计≤6,228,574,053B/8,000,000,000B。

失败完整保留：Metric3D初load缺depth_model.encoder.mask_token；官方推理masks=None不会读这个仅训练用参数，修复只放行此精确missing key，其他missing仍报错。旧失败terminal在failed-attempts/metric3d-load-attempt1，非静默覆盖。两weight TLS零字节失败、MoGe metadata TLS失败、adapter初meta构建/依赖失败与恢复分别有收据；MoGe下载改Windows curl TLS backend和缓存官方metadata后SHA通过，所有预算累计不重置。

四任务GPU拥有进程均已退出并核验PID不再存活，未留服务/worker。模型、第三方源、run-local依赖与全部预测/真值引用/失败收据由本run保留为复算证据，非驻留计算资源；共享进程保留。

交付检查：四个脚本AST、任务范围git diff --check及新增报告链接通过。全局docs index仅旧语义字面量检查失败：HEAD的CURRENT_DECISION已用“盲杖互补前视感知主线；原M3…保留”替代脚本要求的旧句，所有本轮报告链接可解析；这条既有文档/检查器漂移不由RGB任务改写。

证据均为已消费Development、native first-return、固定sampled query-ray；FREE不等于整身体清空，UNKNOWN不当负例，相关帧不证明独立迁移、硬件或导航安全。40777065无近POS是分母缺口，不是近带能力否定。旧A0、有界残差、归一化/带读出、INPUT_NUMERICS/INPUT_BASELINE及其停止约束不修改，不重开本run阈值调参。

## 可复算交付位置

canonical payload：`artifacts.local/work/rgb-depth-backbone-compare-dev-20261010/`，junction实际F盘。每模型有predictions.json与496份native npz；evaluation/calibration有fit、thresholds、inputs、seal；evaluation/final/main-loco和pooled-supplemental有results、summary.csv、query_pairs.csv，各cohort scores下有pixel_metrics.json/pixel_frame_metrics.csv。report-summary/full_tables.md与summary.json覆盖全部capture/cal/eval/距离带/UNKNOWN/尾部，independent_audit.py/json、evaluator-selfcheck、官方license与版本、resource_final.json完整保留。

源码：rgb_depth_backbone_acquire.py、rgb_depth_backbone_adapters.py、rgb_depth_backbone_infer.py、rgb_depth_backbone_evaluate.py。顺序prepare公开496roster→官方fixed revision下载→逐模型GPU run→calibrate封存→final→一次独立复算；重新执行须用新拥有输出并计入当时授权预算，不覆盖本run或隐式重新下载。
