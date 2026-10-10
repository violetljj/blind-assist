# RGB body-query：输入尺度链与精度诊断（Development，2026-10-10）

96帧ARKit输入合约核验通过，未发现时间戳配对、单位、内参或焦距缩放的实现错误。事先固定16帧的FP16/float32对照没有消除10000m尾，也没有恢复原生近距离覆盖；arkit16不含饱和的近正点仍有约3.79倍affine度量偏差。归一化分带阈值下保留局部救3、损1、FREE增减0的混合结果，其中一项来自cal近带。当前全局log-affine继续作为RGB代表；本轮不切换精度或提高残差训练优先级，真正重新训练的有界残差仍未被否定。

## 固定范围与输入合约

父计划先于执行保存在`artifacts.local/work/rgb-body-query-input-numerics-dev-20261010/plan.json`。CPU预算1200s，输入审计240s、evaluator及审计240s、集成720s；GPU进程全过程300s。沿用已有模型、缓存、参考与阈值，无训练、下载或新数据。

六个ARKit capture为47333462、40777060、40777065、40777073、40809740、40958733，各16帧，共96帧。核对原始ZIP成员RGB/depth/pincam同名时间戳与SHA、缓存字节/像素、256×192原生尺寸、K缩放1、color_K=depth_K、uint16毫米除以1000（零保持缺失）、identity采样映射、observed掩码，以及ToTensor→Normalize(.5,.5)→float16的CPU逐值一致性。既有head使用的sampled DP与原生缓存逐值一致。

public fx为209.798–218.957px；源码在1536内部resize之前记录原宽256，再计算`inverse_depth = canonical_inverse_depth * (256/f_px)`，回到原尺寸后取`1/clamp(inverse_depth,1e-4,1e4)`。W/fx为1.169179–1.220221，没有错用1536宽度缩放焦距的迹象。提供fx时不使用预测FOV；fy、主点和畸变未进入Depth Pro的米制尺度计算。网络head末端ReLU使零inverse也会进入下限端点，不能仅凭10000m断言underflow。

新cal48帧的执行snapshot、12个upstream Python文件及权重SHA吻合。旧48帧推理收据缺历史runner/upstream SHA，保留来源缺口；当前16帧FP16输出精确复现缓存，不补造旧历史执行SHA。输入合约通过也不能证明真实RGB/LiDAR物理对齐。

## 精度对照与饱和

在读取GT或分数前，六个ARKit capture按公开RGB时间戳排序各取first/last；原val24、新3RScan64两个cohort按(frame,scan)排序各取first/last，共16帧。六个ARKit均为frame0/15；两个3RScan为frame0/494、frame0/895，完整scan身份与输入SHA在`numerics/roster.json`。ARKit RGB/depth为192×256；3RScan RGB540×960，native depth172×224。不得把这个端点样本称作全部capture的精度复测。

两臂使用同权重、public color fx、预处理和upstream infer，仅改变model/input dtype。观察器只保存实际pre-clamp inverse tensor，并返回原torch.clamp结果，finally恢复函数。float32沿当前默认flags：matmul TF32=false、cuDNN TF32=true、matmul precision=highest，不是全算子严格IEEE FP32对照。FP16的16帧与既有缓存全部精确相等，max abs=0。

| 原色网格统计（每臂2663424像素） | FP16 | float32 |
| --- | ---: | ---: |
| inverse恰为0 | 14830 | 14757 |
| inverse严格小于实际dtype的clamp下限 | 14830 | 14857 |
| 输出10000m | 14830 | 14857 |
| inverse负值/非有限值 | 0/0 | 0/0 |

float32中100个非零inverse仍低于下限；不能把“零减少73”写成“尾已消除”。两臂共32个dtype/倒数合约检查使用实际dtype下限与reciprocal ULP，不用float64端点假设代替FP16算术。上述端点来自全部16帧原色网格，不能与下面的native近正union点分母混用。

## 0.3–0.8m带：度量与同成员对照

使用原参考中严格POSITIVE>=16查询的label=1，按每帧native像素取union；点与帧相关，GT是现有first-return参考。两精度共同可用掩码配对，不填UNKNOWN。affine固定原train参数a=.3713312368564329、b=.2635894826641125，在native sampled DP上计算`log z_affine=a log z_DP+b`。

| cohort（所选帧） | 近正native点 | raw预测/GT中位 FP16→float32 | affine预测/GT中位 FP16→float32 | affine轴向near带inside FP16→float32 |
| --- | ---: | ---: | ---: | ---: |
| arkit16 | 49152 | 6.9309→6.9409 | 3.7923→3.7941 | 0→0 |
| 40777060 | 0 | NOT_EVALUABLE | NOT_EVALUABLE | NOT_EVALUABLE |
| 40777065 | 0 | NOT_EVALUABLE | NOT_EVALUABLE | NOT_EVALUABLE |
| 新ARKit cal（6帧合并） | 29112 | 22.3215→22.1882 | 4.9388→4.9263 | 0→0 |
| 原val24端点 | 12538 | 1.0232→1.0246 | 1.7379→1.7389 | 0→0 |
| 新3RScan64端点 | 4302 | 2.3920→2.3929 | 2.1810→2.1813 | 0→0 |

40777060的完整旧16帧有近POS，但本轮固定两个端点没有近POS；因此其float32近带为NOT_EVALUABLE，不能把缺样本写成负结果。40777065同样保留缺正例。

另外对两臂都未触发高/低clamp、成员完全相同的点做辅助诊断。arkit16的49152点全部未饱和，affine误差中位仍+1.5992→+1.6002m；cal共同未饱和20178点的raw ratio为2.6873→2.6846，affine为2.6512→2.6505，affine轴向inside仍0。这个同成员比较支持“这些样本的偏差不只发生于饱和点”，不能外推为所有输入或定位唯一原因。原val raw轴向inside两臂均12538、新3RScan均1278，与affine表分开；这些点级差异不是raw模型的校准查询优势。

native inside仅指轴向0.3–0.8m带，不冒充XYZ盒包含、身体clearance或known-positive查询见证。全三距离带、逐帧、all/low/high/unclamped分组及配对差异完整保存在`evaluation/{metric_results.json,metric_frame_rows.csv,native_paired.json}`和`common-unclamped-diagnostic.json`。

## 冻结查询读出：救／损及实际成本

只对两精度下的固定train-affine读出比较，不给raw DP套用affine阈值。沿用全局588 FREE校准与已消费分带5%诊断的absolute/normalized阈值，不重新校准，包含并列分数和UNKNOWN分母。16帧432查询，每个readout/policy432个配对，共1728配对；完整工作点、环境/距离分母与逐查询表在`evaluation/query_results.json`、`query_pairs.csv`及各evaluation JSON。

| 冻结读出/阈值来源 | float32救 | float32损 | FREE新增 | FREE移除 |
| --- | ---: | ---: | ---: | ---: |
| absolute / global588 | 0 | 0 | 0 | 0 |
| absolute / band5 | 0 | 0 | 0 | 0 |
| normalized / global588 | 0 | 0 | 0 | 0 |
| normalized / band5（辅助） | 3 | 1 | 0 | 0 |

normalized/band5的四个变化全部列出：

| 来源 | frame | query | 距离带 | 变化 |
| --- | ---: | --- | --- | --- |
| eval 40777060 | 15 | fragment_x2_y2_z2 | 1.5–3m | 救1 |
| cal 40777073 | 15 | fragment_x0_y1_z2 | 1.5–3m | 损1 |
| cal 40777073 | 15 | fragment_x0_y2_z2 | 1.5–3m | 救1 |
| cal 40958733 | 0 | fragment_x1_y2_z0 | 0.3–0.8m | 救1 |

对应40777060远带FREE支持1/2→1/2；cal近带0/40→0/40；cal远带FREE分母0，成本NOT_EVALUABLE，不报告为已验证零误支持。这里的cal近带见证救1与原生inside0并不矛盾：既有normalized near cutoff为−3.9667565，允许负margin。没有新增eval近带见证，不能把cal事后局部获救写成可迁移收益；也不删除这个混合信号。

## 决策、成本与核验

本轮未找到可直接修复的输入尺度链错误，默认float32对照未解决饱和尾和所选近正点的主要偏差。保留全局log-affine作为当前RGB代表，旧A0三δ三capture、原三折、补参考与条件校准结果全部保留；不从本轮少量局部变化推出替代配方或训练强信号。FP32推理成本明显上升，当前没有支持全面重跑的收益。

下一步转向真正残差的可学习目标与界限：复用已有train/cal分析需要修正的log-Z范围及输入可辨识性，界限和训练选择只由train/cal确定；不能用eval中位误差倒推bound，也不默认启用q01/q99回退。A0裁剪绝对距离旧模型、实际残差训练和本轮精度对照是不同机制，本轮不能否定重新训练。双判据仍是双方认可的强信号，不改写成唯一门槛；收益与成本同时增加仍属于待权衡结果。

GPU进程全过程262.191/300s，共32次模型推理；FP16/float32 arm wall为24.837/226.581s，纯infer为13.079/213.181s，CUDA peak为3977119744/8532470272 bytes。没有OOM或推理失败。进程退出前cleanup后allocated仍为9568256 bytes，如实留在收据；PID已退出、writer已释放，GPU回到共享桌面占用，未保留task-owned进程/模型。

输入审计CPU10.151/240s；evaluator3.729s、独立核验1.772s，另同成员辅助汇总两次.354/.342s、补核验1.457s及一次inspection引号SyntaxError .417s，累计8.072/240s。辅助汇总首次FREE枚举匹配错误导致成本字段为空，原版本与修正仅重汇总的版本均保留，metric和主核验不变。public名单/32产物/SHA/dtype/fx及预算postcheck .396s。集成按720s上限保守计费，CPU总保守741/1200s；无训练、下载。

核验独立复算32个dtype/clamp倒数、48份native band值、336指标组、3456冻结decisions及1728配对；arkit16/frame0另从native参考重算108个排序score/witness，max abs=0。补核验42个同成员组/1944分位标量、36组实际FREE成本和原冻结描述符。源snapshot与本轮三helper一致，AST、scoped whitespace和hot-doc index核验见`integration_checks.json`。没有重跑全96输入或全部模型来完成核验。

本轮所有结果仅限已消费Development、现有参考、固定16帧端点选择及默认数值flags；cal救损/同成员曲线是事后诊断，不是独立确认，也不证明整盒、身体、实机、安全或泛化。数据合约和精度对照均未定位模型转移偏差的唯一成因。

三个helper：`rgb_body_query_input_contract.py`、`rgb_body_query_precision_probe.py`、`rgb_body_query_precision_eval.py`。全payload与执行命令/收据位于canonical `artifacts.local/work/rgb-body-query-input-numerics-dev-20261010/`（junction到F盘），不覆盖旧run。精度stage顺序为`--stage prepare`后`--stage run --budget-s 300`，重新执行须使用新输出目录并先记录父预算，不隐式重置已执行预算。

前文：[近带排序与几何](RGB_BODY_QUERY_NEAR_RANK_GEOMETRY_DEV_20261010.md)、[分带校准](RGB_BODY_QUERY_DISTANCE_CAL_DEV_20261010.md)、[补ARKit参考](RGB_BODY_QUERY_ARKIT_CAL_DEV_20261010.md)、[A0原三折](RGB_BODY_QUERY_A0_DEV_20261010.md)。
