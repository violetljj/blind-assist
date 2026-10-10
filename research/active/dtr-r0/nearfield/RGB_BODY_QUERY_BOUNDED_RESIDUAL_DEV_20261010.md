# RGB：实际训练有界 log-Z 残差与局部深度上下文

日期：2026-10-10。`RGB_BODY_QUERY_BOUNDED_RESIDUAL_DEV_20261010`，EXPLORE，已消费 Development。起点提交 `b97299b4`，用户授权“那就做”。前轮[残差目标诊断](RGB_BODY_QUERY_RESIDUAL_TARGET_DEV_20261010.md)的邻域代理不等于本轮训练模型。

## 结果与决定

两条实际训练残差均未形成跨capture的有利取舍，本轮固定配方不升级，**全局 affine 继续作 RGB 代表**。主三折在两个追加capture，affine为见证/FREE **120/12、131/10**；depth/ray残差为 **103/10、104/26**，局部上下文残差为 **94/22、102/9**。上下文对60号相对affine救3损29且FREE净增10；65号救2损31、FREE净减1。两个追加capture的主见证和事后同成本都低于affine，属于这次配方的负信号，不能归结为“只差一次阈值漂移”。

局部收益完整保留：arkit16事后固定原affine工作点 **5/105 FREE**，上下文见证 **138/228**，对冻结主affine的103救54损19（净+35）；若affine同样填满FREE阈值平台，则affine能达113，候选对其救50损25（净+25）。近带2/44见证、FREE1/96，对冻结affine近带0/44、FREE0/96。总FREE相同不意味着误支持位置相同，不能记作近带零成本。两个追加capture没有对应收益，不逐capture换臂或挑阈值。

上下文相对depth/ray也有取舍：主65号省17个FREE、净少2个见证；pooled补充65号救16损10、净省6FREE，但60号救4损19、净增8FREE。训练最后一batch SmoothL1从同初始.230879变为depth/ray .178320、context .118843，训练损失改善未变成稳定查询优势。

主三折与pooled补充两trained臂的全部阈值均 **>.25m**，结构性排除0.3–0.8m带（区间余量最大.25m）；其近带零见证不证明模型没有近信息。304cal的两trained补充阈值分别.265199845/.348014469m，均实际29/588 FREE；first-return训练损失、区间排名和全局校准的作用仍可能共同影响结果，不能定位单一原因。

本run已完成并停止，不追加此固定配方epoch/δ/阈值调参；保留两step600模型、完整失败与配对供复核。若继续残差路线，优先改变可观测信息或训练域覆盖（例如冻结RGB骨干上下文/新增合规train来源），而非把cal改作train或继续只放大界限。不同机制保持开放。

## 训练及可比性

两臂均冻结原 log-affine：`log z_affine = 0.3713312368564329 log z_DP + 0.2635894826641125`，训练输出 `log z = log z_affine + B*tanh(h/B)`，B固定 **3.1847211408780134**，来自前轮预声明 train/cal q99 envelope，没有用 eval 修改。输出层零初始化，初始输出等于 affine。

两臂都是 Linear11→64/ReLU→64/ReLU→1，共4993参数；对照 `trained_depth_ray` 只用 train标准化 logDP/ray_x/ray_y，后8列置零。候选 `trained_context` 增加原生 logDP中央x/y差分，以及3/9/25方窗均值减中心和标准差；边界replicate、无效DP邻域按public validity忽略。特征没有 GT距离带、query标签或来源ID；每列标准化仅原 train有效监督点。它增加深度图的空间上下文，没有新增 RGB骨干特征。

只监督原 **56帧 / 7环境 / 1,591,791点**，原train每环境8帧。每步每帧均匀有放回取512有效监督点，batch28672，使帧/环境等权；两臂使用同初始化seed20261010、同样本顺序、同600step。完整无界目标 `log native first-return reference−log z_affine` 保留，超bound标签不删除。SmoothL1(beta=.2)、AdamW(lr=.001,weight_decay=.0001)，只使用最后step600，不用cal挑epoch/seed或调参。单seed不证明训练稳定性。

**304 cal**（原16＋追加3RScan240＋ARKit48）保持校准角色；**136 eval**为既有Development的24原validation、64新3RScan与三条各16帧ARKit capture。冻结训练配置并完成pooledcal阈值后，才打开eval结果；本轮没有新DepthPro推理、下载或数据扩张。cal重复消费并非独立确认。

## 查询协议与计数

每帧固定27query；query分数为public interval domain内第16大 `min(z−entry,exit−z)`，单位m；已知正见证分数在evaluator-known正射线上同样取第16大。query POS需至少16已知正射线；严格采样FREE需正0、UNKNOWN0、free≥16，UNKNOWN不计负。见证改善/损失与query支撑分开；这些不是整身体/体积占据证据。

主结果沿原三capture留一：阈值仅另两capture严格FREE选≤5%经验工作点，每臂各自校准，保留ties。补充工作点使用304cal的588严格FREE、实际≤floor(.05×588)=29；先保存阈值再读eval。held capture自身FREE对齐原affine成本和完整曲线仅事后诊断，不能反过来选模型。全两臂/三capture保留，没有逐capture挑最佳。


## 主结果：三capture留一

| Cohort / 臂 | 正见证 / POS | FREE支持 / FREE | query正支持 | UNKNOWN支持 / UNKNOWN | cutoff(m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| arkit16 / affine | 103/228 | 5/105 | 105 | 74/99 | -0.123237819 |
| arkit16 / trained_depth_ray | 47/228 | 0/105 | 65 | 59/99 | 0.289702120 |
| arkit16 / trained_context | 52/228 | 0/105 | 73 | 67/99 | 0.348505574 |
| arkit_40777060 / affine | 120/202 | 12/220 | 122 | 9/10 | -0.128096933 |
| arkit_40777060 / trained_depth_ray | 103/202 | 10/220 | 106 | 5/10 | 0.273557611 |
| arkit_40777060 / trained_context | 94/202 | 22/220 | 118 | 7/10 | 0.347828753 |
| arkit_40777065 / affine | 131/179 | 10/247 | 131 | 3/6 | -0.062205461 |
| arkit_40777065 / trained_depth_ray | 104/179 | 26/247 | 111 | 1/6 | 0.252451704 |
| arkit_40777065 / trained_context | 102/179 | 9/247 | 113 | 1/6 | 0.348327227 |

| Paired对照 | 正见证救 / 损 | FREE增 / 减 |
| --- | ---: | ---: |
| arkit16/trained_depth_ray/vs_affine | 8/64 | 0/5 |
| arkit16/trained_context/vs_affine | 4/55 | 0/5 |
| arkit16/trained_context/vs_depth_ray | 18/13 | 0/0 |
| arkit_40777060/trained_depth_ray/vs_affine | 9/26 | 2/4 |
| arkit_40777060/trained_context/vs_affine | 3/29 | 12/2 |
| arkit_40777060/trained_context/vs_depth_ray | 9/18 | 14/2 |
| arkit_40777065/trained_depth_ray/vs_affine | 3/30 | 18/2 |
| arkit_40777065/trained_context/vs_affine | 2/31 | 4/5 |
| arkit_40777065/trained_context/vs_depth_ray | 14/16 | 4/21 |

### 0.3–0.8m 单列

| Cohort / 臂 | 近带正见证 / POS | 近带FREE支持 / FREE |
| --- | ---: | ---: |
| arkit16 / affine | 0/44 | 0/96 |
| arkit16 / trained_depth_ray | 0/44 | 0/96 |
| arkit16 / trained_context | 0/44 | 0/96 |
| arkit_40777060 / affine | 0/11 | 0/133 |
| arkit_40777060 / trained_depth_ray | 0/11 | 0/133 |
| arkit_40777060 / trained_context | 0/11 | 0/133 |
| arkit_40777065 / affine | N/E (POS0) | 0/144 |
| arkit_40777065 / trained_depth_ray | N/E (POS0) | 0/144 |
| arkit_40777065 / trained_context | N/E (POS0) | 0/144 |

| 近带Paired对照 | 正见证救 / 损 | FREE增 / 减 |
| --- | ---: | ---: |
| arkit16/trained_depth_ray/vs_affine | 0/0 | 0/0 |
| arkit16/trained_context/vs_affine | 0/0 | 0/0 |
| arkit16/trained_context/vs_depth_ray | 0/0 | 0/0 |
| arkit_40777060/trained_depth_ray/vs_affine | 0/0 | 0/0 |
| arkit_40777060/trained_context/vs_affine | 0/0 | 0/0 |
| arkit_40777060/trained_context/vs_depth_ray | 0/0 | 0/0 |
| arkit_40777065/trained_depth_ray/vs_affine | 0/0 | 0/0 |
| arkit_40777065/trained_context/vs_affine | 0/0 | 0/0 |
| arkit_40777065/trained_context/vs_depth_ray | 0/0 | 0/0 |

## 补充：304cal冻结工作点

| Cohort / 臂 | 正见证 / POS | FREE支持 / FREE | query正支持 | UNKNOWN支持 / UNKNOWN | cutoff(m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| original_validation24 / affine | 219/295 | 1/2 | 234 | 86/351 | -0.081071358 |
| original_validation24 / trained_depth_ray | 104/295 | 0/2 | 128 | 24/351 | 0.265199845 |
| original_validation24 / trained_context | 88/295 | 0/2 | 119 | 25/351 | 0.348014469 |
| new_3rscan64 / affine | 570/866 | 1/1 | 596 | 222/861 | -0.081071358 |
| new_3rscan64 / trained_depth_ray | 256/866 | 1/1 | 332 | 103/861 | 0.265199845 |
| new_3rscan64 / trained_context | 211/866 | 1/1 | 264 | 155/861 | 0.348014469 |
| arkit16 / affine | 95/228 | 5/105 | 100 | 74/99 | -0.081071358 |
| arkit16 / trained_depth_ray | 50/228 | 0/105 | 68 | 60/99 | 0.265199845 |
| arkit16 / trained_context | 53/228 | 0/105 | 74 | 67/99 | 0.348014469 |
| arkit_40777060 / affine | 120/202 | 12/220 | 122 | 9/10 | -0.081071358 |
| arkit_40777060 / trained_depth_ray | 106/202 | 11/220 | 109 | 5/10 | 0.265199845 |
| arkit_40777060 / trained_context | 91/202 | 19/220 | 115 | 7/10 | 0.348014469 |
| arkit_40777065 / affine | 131/179 | 10/247 | 131 | 3/6 | -0.081071358 |
| arkit_40777065 / trained_depth_ray | 96/179 | 19/247 | 100 | 1/6 | 0.265199845 |
| arkit_40777065 / trained_context | 102/179 | 13/247 | 115 | 1/6 | 0.348014469 |

| Paired对照 | 正见证救 / 损 | FREE增 / 减 |
| --- | ---: | ---: |
| original_validation24/trained_depth_ray/vs_affine | 2/117 | 0/1 |
| original_validation24/trained_context/vs_affine | 11/142 | 0/1 |
| original_validation24/trained_context/vs_depth_ray | 30/46 | 0/0 |
| new_3rscan64/trained_depth_ray/vs_affine | 2/316 | 0/0 |
| new_3rscan64/trained_context/vs_affine | 19/378 | 0/0 |
| new_3rscan64/trained_context/vs_depth_ray | 61/106 | 0/0 |
| arkit16/trained_depth_ray/vs_affine | 8/53 | 0/5 |
| arkit16/trained_context/vs_affine | 4/46 | 0/5 |
| arkit16/trained_context/vs_depth_ray | 16/13 | 0/0 |
| arkit_40777060/trained_depth_ray/vs_affine | 10/24 | 3/4 |
| arkit_40777060/trained_context/vs_affine | 0/29 | 9/2 |
| arkit_40777060/trained_context/vs_depth_ray | 4/19 | 11/3 |
| arkit_40777065/trained_depth_ray/vs_affine | 0/35 | 11/2 |
| arkit_40777065/trained_context/vs_affine | 2/31 | 8/5 |
| arkit_40777065/trained_context/vs_depth_ray | 16/10 | 8/14 |

### 0.3–0.8m 单列

| Cohort / 臂 | 近带正见证 / POS | 近带FREE支持 / FREE |
| --- | ---: | ---: |
| original_validation24 / affine | 3/24 | N/E (FREE0) |
| original_validation24 / trained_depth_ray | 0/24 | N/E (FREE0) |
| original_validation24 / trained_context | 0/24 | N/E (FREE0) |
| new_3rscan64 / affine | 3/168 | N/E (FREE0) |
| new_3rscan64 / trained_depth_ray | 0/168 | N/E (FREE0) |
| new_3rscan64 / trained_context | 0/168 | N/E (FREE0) |
| arkit16 / affine | 0/44 | 0/96 |
| arkit16 / trained_depth_ray | 0/44 | 0/96 |
| arkit16 / trained_context | 0/44 | 0/96 |
| arkit_40777060 / affine | 0/11 | 0/133 |
| arkit_40777060 / trained_depth_ray | 0/11 | 0/133 |
| arkit_40777060 / trained_context | 0/11 | 0/133 |
| arkit_40777065 / affine | N/E (POS0) | 0/144 |
| arkit_40777065 / trained_depth_ray | N/E (POS0) | 0/144 |
| arkit_40777065 / trained_context | N/E (POS0) | 0/144 |

| 近带Paired对照 | 正见证救 / 损 | FREE增 / 减 |
| --- | ---: | ---: |
| original_validation24/trained_depth_ray/vs_affine | 0/3 | 0/0 |
| original_validation24/trained_context/vs_affine | 0/3 | 0/0 |
| original_validation24/trained_context/vs_depth_ray | 0/0 | 0/0 |
| new_3rscan64/trained_depth_ray/vs_affine | 0/3 | 0/0 |
| new_3rscan64/trained_context/vs_affine | 0/3 | 0/0 |
| new_3rscan64/trained_context/vs_depth_ray | 0/0 | 0/0 |
| arkit16/trained_depth_ray/vs_affine | 0/0 | 0/0 |
| arkit16/trained_context/vs_affine | 0/0 | 0/0 |
| arkit16/trained_context/vs_depth_ray | 0/0 | 0/0 |
| arkit_40777060/trained_depth_ray/vs_affine | 0/0 | 0/0 |
| arkit_40777060/trained_context/vs_affine | 0/0 | 0/0 |
| arkit_40777060/trained_context/vs_depth_ray | 0/0 | 0/0 |
| arkit_40777065/trained_depth_ray/vs_affine | 0/0 | 0/0 |
| arkit_40777065/trained_context/vs_affine | 0/0 | 0/0 |
| arkit_40777065/trained_context/vs_depth_ray | 0/0 | 0/0 |

## 事后辅助：原affine held-FREE成本

| Cohort / 臂 | 正见证 / POS | FREE支持 / FREE | query正支持 | UNKNOWN支持 / UNKNOWN | cutoff(m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| arkit16 / affine | 103/228 | 5/105 | 105 | 74/99 | 辅助见ledger |
| arkit16 / trained_depth_ray | 103/228 | 5/105 | 121 | 85/99 | 辅助见ledger |
| arkit16 / trained_context | 138/228 | 5/105 | 155 | 90/99 | 辅助见ledger |
| arkit_40777060 / affine | 120/202 | 12/220 | 122 | 9/10 | 辅助见ledger |
| arkit_40777060 / trained_depth_ray | 110/202 | 12/220 | 118 | 5/10 | 辅助见ledger |
| arkit_40777060 / trained_context | 91/202 | 12/220 | 106 | 7/10 | 辅助见ledger |
| arkit_40777065 / affine | 131/179 | 10/247 | 131 | 3/6 | 辅助见ledger |
| arkit_40777065 / trained_depth_ray | 80/179 | 10/247 | 83 | 0/6 | 辅助见ledger |
| arkit_40777065 / trained_context | 102/179 | 10/247 | 114 | 1/6 | 辅助见ledger |

| Paired对照 | 正见证救 / 损 | FREE增 / 减 |
| --- | ---: | ---: |
| arkit16/affine/vs_fixed_affine | 0/0 | 0/0 |
| arkit16/trained_depth_ray/vs_fixed_affine | 32/32 | 2/2 |
| arkit16/trained_context/vs_fixed_affine | 54/19 | 3/3 |
| arkit_40777060/affine/vs_fixed_affine | 0/0 | 0/0 |
| arkit_40777060/trained_depth_ray/vs_fixed_affine | 12/22 | 4/4 |
| arkit_40777060/trained_context/vs_fixed_affine | 0/29 | 2/2 |
| arkit_40777065/affine/vs_fixed_affine | 0/0 | 0/0 |
| arkit_40777065/trained_depth_ray/vs_fixed_affine | 0/51 | 5/5 |
| arkit_40777065/trained_context/vs_fixed_affine | 2/31 | 5/5 |

### 0.3–0.8m 单列

| Cohort / 臂 | 近带正见证 / POS | 近带FREE支持 / FREE |
| --- | ---: | ---: |
| arkit16 / affine | 0/44 | 0/96 |
| arkit16 / trained_depth_ray | 0/44 | 0/96 |
| arkit16 / trained_context | 2/44 | 1/96 |
| arkit_40777060 / affine | 0/11 | 0/133 |
| arkit_40777060 / trained_depth_ray | 0/11 | 0/133 |
| arkit_40777060 / trained_context | 0/11 | 0/133 |
| arkit_40777065 / affine | N/E (POS0) | 0/144 |
| arkit_40777065 / trained_depth_ray | N/E (POS0) | 0/144 |
| arkit_40777065 / trained_context | N/E (POS0) | 0/144 |

| 近带Paired对照 | 正见证救 / 损 | FREE增 / 减 |
| --- | ---: | ---: |
| arkit16/affine/vs_fixed_affine | 0/0 | 0/0 |
| arkit16/trained_depth_ray/vs_fixed_affine | 0/0 | 0/0 |
| arkit16/trained_context/vs_fixed_affine | 2/0 | 1/0 |
| arkit_40777060/affine/vs_fixed_affine | 0/0 | 0/0 |
| arkit_40777060/trained_depth_ray/vs_fixed_affine | 0/0 | 0/0 |
| arkit_40777060/trained_context/vs_fixed_affine | 0/0 | 0/0 |
| arkit_40777065/affine/vs_fixed_affine | 0/0 | 0/0 |
| arkit_40777065/trained_depth_ray/vs_fixed_affine | 0/0 | 0/0 |
| arkit_40777065/trained_context/vs_fixed_affine | 0/0 | 0/0 |

完整各带/环境、逐query评分与配对、全部事后成本曲线分别在 `evaluation/final/{main-loco,pooled-supplemental,posthoc-fixed-affine-cost}/`、`*_scores.json`、`posthoc_matched_points.csv`、`posthoc_curves.csv`。辅助比较报告actual FREE，保留score ties；不把要求预算当实际成本。

### 同FREE预算下，affine也填满阈值平台的辅助比较

固定主affine点仍保留上表；完整曲线用FREE-only选最低允许阈值，会在相同成本平台容纳更多正例。以下同样只是事后held-FREE诊断，不用于选模型。

| Capture / 臂 | affine曲线见证 | 候选见证 | 双方实际FREE | 救 / 损 |
| --- | ---: | ---: | ---: | ---: |
| arkit16/trained_depth_ray | 113 | 103 | 5 / 105 | 28/38 |
| arkit16/trained_context | 113 | 138 | 5 / 105 | 50/25 |
| arkit_40777060/trained_depth_ray | 122 | 110 | 12 / 220 | 12/24 |
| arkit_40777060/trained_context | 122 | 91 | 12 / 220 | 0/31 |
| arkit_40777065/trained_depth_ray | 132 | 80 | 10 / 247 | 0/52 |
| arkit_40777065/trained_context | 132 | 102 | 10 / 247 | 2/32 |

此表逐query配对另存 `evaluation/final/posthoc_curve_point_pairs.json`，它直接来自已核完整曲线与固定评分，没有改阈值选择规则。60/65在这些成本点已低于affine，故两条追加capture均不满足曲线“不劣”的强信号。

## 核验、成本与可复现证据

输入核验独立重算1原生train frame全部11特征，最大差1.96e−12，33监督点/全部训练目标和600×28672批次一致；两臂4993参数、初始化SHA、零输出和final600选择通过。训练/cal/eval角色及public特征源按执行source/hash/receipt核验；没有额外文件访问tracer，原生核验帧DP全positive，缺邻域逻辑仅核源码合同。input审计一次因发布源码默认路径变动而误断言，failed source/terminal保留，修正为对执行/交付版本分开核对。

focused核验通过2final600 checkpoint、608cal＋272eval实际有界数组、3pooled阈值＋9LOCO阈值、18,792query决策＋18,792完整配对（含直接context-vs-depth/ray）、1,725保ties曲线和9固定成本点，442,611离散/130,248数值项；独立Ark16原生一帧81臂query排序最大误差0。独立CPU重跑1公共cal帧两checkpoint，residual最大差1.56e−7/2.38e−7（CPU/CUDA float32差），小于声明2e−5；无重复训练。原affine cal/eval全部23,760 query_score/known_positive_score精确复现。

执行Torch2.11.0+cu130、CUDA13.0、float32、matmul TF32=false、cudnn TF32=true、precision=highest；MLP无cudnn卷积，不称全算子严格IEEE。代表相同batch forward/backward三次中位CUDA.002053s、CPU.004718s，CUDA首次kernel13.322s已计费；benchmark6次forward/backward无optimizer且不消耗批次RNG，选择CUDA不改科学配置。两臂共1,200优化步，公开native residual预测2×(304＋136)=880帧臂，另focused2次CPU复核；新DepthPro前向0、下载0。

GPU过程训练＋cal **33.992s**、eval公开特征/预测 **15.136s**，累计 **49.128/900s**（含benchmark）。CPU预处理保守32.997s（17.861prepare＋eval整个15.136GPU过程重复计CPU上界）、查询cal/final27.159s、input审计8.5s保守（含失败/修复）、focused5.961s、发布修复保守1s、整合全额200s；合计保守上界275.617s，向上十秒取整 **280/1200s**，不是实测CPU算时。主科学执行与focused均无失败；input自身失败留证。

CUDA峰值264,953,856B（252.68MiB）；cleanup后train/eval仍报告18,087,936/9,568,256B，原数保留，不称allocation为0。进程PID33096/65032均退出、writer不存在，GPU回共享约520MiB基线；无常驻task资源。两个step600 checkpoint和原始输入/特征/配对批次/预测作为本run拥有的持久证据保留，不删除或替换。

计划SHA `6c65094a19158d9a3a5ed486803c61dd1f5e166b1d824a66b10345b09e0878a6`。trainer执行 `8af8d92ce09309f2a0107ffdfb33dbc07d6851ea46dd24c9b2980fdbe8eaa55d`，交付 `1d654d3888e28c0b50340f79378bee924c8801def9c0f67ab081b283d514d6a1`：只把默认repo由机器路径改为source parents[4]；本次命令省略--repo，两默认都定位同checkout，`publication_note.json`明示，不重跑数值。scorer校准执行 `ca184d3ea26059841f6812f537757a8a77af2206bd45d239e89c94dbe8c5a37e`、final/交付 `56459109c32fc07c89aa8aff85e239023522ae1762ee4bf1493615dda647e12e`：只澄清无效DP的finite验证限定public valid mask，评分公式与已冻cut不变；两个执行snapshot均保留。

两final600 SHA：depth/ray `a3866aac9738d7c3594bd36212a21ccef93c2116e43035f0d54f977b1c569562`；context `d629a677641c27de4a7070a4b5a0b035a1ec1e9fe4243f3be62ca30068e99205`。初始化SHA `1450bbdd1f3d4032ad863f0fa21fce63d1820cd062c43d2713543ebab366673e`，batchSHA `dce0bbbbcf22e80a3a0014d049d257a9f26e684a378063772f58e6c10d411466`；calcut冻文件SHA `1377cc1ff20f1f4df38a3aee4f14158553255cb39f1e9a93fb55a2fdb93e762d`随eval receipt记录。

复现源码：[训练与公开预测](rgb_body_query_bounded_residual.py)、[查询评分与校准](rgb_body_query_bounded_evaluate.py)。阶段依次prepare→train→calibrate→predict-eval（显式--calibration-freeze）→final；需要项目Torch/CUDA、NumPy/OpenCV/SciPy runtime，根先提供冻结plan。现有完成stage禁止隐式覆盖/重启，不以重新启动重置累计预算。

完整证据根 `artifacts.local/work/rgb-body-query-bounded-residual-dev-20261010/`：`plan.json`、`training_inputs.json`、`normalization.npz`、`train_features.npy`/`train_targets.npy`/`train_identity.npz`、`batch_indices.npy`、两step600.pt及progress、各executed.py、CPU/GPU ledger和terminal、`benchmark.json`、`predictions/`、`evaluation/`、`input-audit/`、`focused-audit.json`、`publication_note.json`、`producer_postcheck.json`。

主干/模型/阈值均为已消费Development探索，query ray与相关capture不能提升成独立确认、整身体/体积、实机、安全或泛化证明。first-return参考/注册与原推理SHA限制沿前报告保留。局部上下文这一配方的结果不否定RGB深层特征、更广训练来源或其他残差形式。原A0的全部δ、arkit16退化、LOCO/事后成本边界及各run停止规则继续保留；未满足双判据不自动为负，收益和成本同时增加完整展示。
