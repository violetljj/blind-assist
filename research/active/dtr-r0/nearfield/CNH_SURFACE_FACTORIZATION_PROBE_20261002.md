ORACLE_GEOMETRY_DIAGNOSTIC

# 可见表面角度—距离对应的误差拆分

**确认当前边缘分布读出会丢失几何对应，但没有建立新的报警收益。** 96个既有训练单位、2112场景、54912个查询帧；边缘化误差均值0.00027098，径向量化误差0.00003314，前者约为后者8.18倍。二者差0.00023784，整单位配对95%区间[0.00022350,0.00025497]。单位是256微格等权平均质量，不是米、准确率或物理面积。

0–2cm浅擦碰2314查询帧（96单位）中，边缘化相对保留对应的总L1误差净增0.00066471 [0.00061503,0.00072400]。它既丢失局部真实质量，也向错误方向加入质量；全查询平均质量反而净增0.00037132，不能简单说成“信号变小导致漏停”。

2243条13帧全程清晰查询序列，EXACT/JOINT_BIN/MARGINAL出现任何非零质量的序列数为0/1/842。清晰38504查询帧上的边缘化额外质量0.00000990 [0.00000676,0.00001374]；质量很小且未设报警阈值，842不是误停次数。另591条擦身序列边缘化有573条非零；擦身报警本来可接受，不计清晰误报。

## 下一决定

若继续几何读出，优先检验**逐方向首可见距离及可见性**，保留角—距对应后再与身体查询相交；不再把一个微格的距离分布独立摊给该格所有方向。下一问题是这种对应能否由实际native CNH及因果多帧姿态学出，且改善报警任务。本轮未训练这种模型、未读取校准/评估，也未启动后续方法。

M3仍保留；SURF的NOT_ESTABLISHED_SINGLE_SEED_DEV及此前失败不改判。本轮与SURF评价不是相同样本/任务，不能把几何误差直接归因为其25/31及时停或224次清晰误停。

## 三个量与验证

每微格原64条射线、归一固体角权重w、首可见真实径向距离d、箱号b，公开查询为I：

- EXACT E = sum_r w_r I(r,d_r)：保留射线与真距离对应。
- JOINT_BIN J = sum_r w_r I(r,c_b)：同射线，距离换成3.75348cm箱中心。
- MARGINAL F = sum_b P(b) sum_r w_r I(r,c_b)：真距离箱分布与公共查询角权重相乘。

逐微格计算|J−E|、|F−J|、|F−E|后再平均，避免正负抵消；另报告总误差净增|F−E|−|J−E|。用float64直方图与公开角权重，排除旧标签float16误差；新公共权重与冻结float32版本偏差<5e−8。所有帧断言无径向域外点进入query，故J−E不混入4.804m截断损失。

解析小例验证：同一微格两个角度交换距离时P(b)不变，但查询质量从0.5变为1；边缘表示不能区分。相同正权射线下J>0必有F>0，因此边缘化可改幅度、加支持，不能独自删除零阈值支持。全队列逐帧验证EXACT支持必属全物体contact。

用既有训练场景所有box物理表面，按当前travel完整刚体变换后裁剪，分为0–2/2–5/>5cm擦碰、0–10cm擦身和清晰。帧按当时类别；序列按13帧出现的最深类别，清晰必须全部13帧清晰。本轮不是0.9m停步真值。统计先在单位内平均后单位等权；类别条件项按共同抽中单位的类别分子/分母重新汇总。1000整单位共同bootstrap，seed2026100233。

仅几何射线重算，无新sensor响应合成、模型训练或校准/评估访问。两块并行，96个单位记录耗时合计315.85秒；这不是并行墙钟时间。保存单位数组独立统计复算见payload内verification.json。

## 完整统计

| 误差 | 单位均匀均值 | 单位bootstrap 95%区间 |
|---|---:|---|
| radial_quantization | 0.00003314 | [0.00002908, 0.00003759] |
| angle_range_marginalization | 0.00027098 | [0.00025332, 0.00029133] |
| total | 0.00028277 | [0.00026484, 0.00030371] |
| angle_minus_quantization | 0.00023784 | [0.00022350, 0.00025497] |
| marginal_total_minus_joint_total | 0.00024963 | [0.00023399, 0.00026736] |

按当前全物体类别条件汇总误差变化（帧均值，全部96单位共同重采样；正负质量由逐微格L1与有符号质量差拆分）：

| 类别 | query-frame /单位 | 指标 | 均值 | 配对单位95%区间 |
|---|---:|---|---:|---|
| contact0-2cm | 2314 / 96 | marginal_total_minus_joint_total | 0.00066471 | [0.00061503, 0.00072400] |
| contact0-2cm | 2314 / 96 | JOINT_BIN_signed_mass_minus_exact | -0.00001605 | [-0.00003394, 0.00000009] |
| contact0-2cm | 2314 / 96 | JOINT_BIN_added_mass_vs_exact | 0.00008871 | [0.00007512, 0.00010314] |
| contact0-2cm | 2314 / 96 | JOINT_BIN_removed_mass_vs_exact | 0.00010477 | [0.00008696, 0.00012697] |
| contact0-2cm | 2314 / 96 | MARGINAL_signed_mass_minus_exact | 0.00037132 | [0.00030789, 0.00043471] |
| contact0-2cm | 2314 / 96 | MARGINAL_added_mass_vs_exact | 0.00061475 | [0.00057294, 0.00066131] |
| contact0-2cm | 2314 / 96 | MARGINAL_removed_mass_vs_exact | 0.00024344 | [0.00019160, 0.00029848] |
| contact2-5cm | 2692 / 96 | marginal_total_minus_joint_total | 0.00044834 | [0.00038458, 0.00053483] |
| contact2-5cm | 2692 / 96 | JOINT_BIN_signed_mass_minus_exact | -0.00000623 | [-0.00001756, 0.00000459] |
| contact2-5cm | 2692 / 96 | JOINT_BIN_added_mass_vs_exact | 0.00009482 | [0.00008088, 0.00011070] |
| contact2-5cm | 2692 / 96 | JOINT_BIN_removed_mass_vs_exact | 0.00010104 | [0.00008745, 0.00011517] |
| contact2-5cm | 2692 / 96 | MARGINAL_signed_mass_minus_exact | 0.00002607 | [-0.00002812, 0.00009012] |
| contact2-5cm | 2692 / 96 | MARGINAL_added_mass_vs_exact | 0.00033513 | [0.00027828, 0.00041420] |
| contact2-5cm | 2692 / 96 | MARGINAL_removed_mass_vs_exact | 0.00030906 | [0.00027610, 0.00034286] |
| contact>5cm | 6006 / 96 | marginal_total_minus_joint_total | 0.00079665 | [0.00072702, 0.00087348] |
| contact>5cm | 6006 / 96 | JOINT_BIN_signed_mass_minus_exact | -0.00002354 | [-0.00003292, -0.00001581] |
| contact>5cm | 6006 / 96 | JOINT_BIN_added_mass_vs_exact | 0.00005336 | [0.00004017, 0.00006638] |
| contact>5cm | 6006 / 96 | JOINT_BIN_removed_mass_vs_exact | 0.00007690 | [0.00006429, 0.00008967] |
| contact>5cm | 6006 / 96 | MARGINAL_signed_mass_minus_exact | -0.00058364 | [-0.00065060, -0.00052726] |
| contact>5cm | 6006 / 96 | MARGINAL_added_mass_vs_exact | 0.00017164 | [0.00012005, 0.00022396] |
| contact>5cm | 6006 / 96 | MARGINAL_removed_mass_vs_exact | 0.00075528 | [0.00070190, 0.00081777] |
| pass0-10cm | 5396 / 96 | marginal_total_minus_joint_total | 0.00107428 | [0.00101101, 0.00114038] |
| pass0-10cm | 5396 / 96 | JOINT_BIN_signed_mass_minus_exact | 0.00000308 | [0.00000081, 0.00000707] |
| pass0-10cm | 5396 / 96 | JOINT_BIN_added_mass_vs_exact | 0.00000308 | [0.00000081, 0.00000707] |
| pass0-10cm | 5396 / 96 | JOINT_BIN_removed_mass_vs_exact | 0.00000000 | [0.00000000, 0.00000000] |
| pass0-10cm | 5396 / 96 | MARGINAL_signed_mass_minus_exact | 0.00107736 | [0.00101386, 0.00114287] |
| pass0-10cm | 5396 / 96 | MARGINAL_added_mass_vs_exact | 0.00107736 | [0.00101386, 0.00114287] |
| pass0-10cm | 5396 / 96 | MARGINAL_removed_mass_vs_exact | 0.00000000 | [0.00000000, 0.00000000] |
| clear | 38504 / 96 | marginal_total_minus_joint_total | 0.00000990 | [0.00000676, 0.00001374] |
| clear | 38504 / 96 | JOINT_BIN_signed_mass_minus_exact | 0.00000119 | [0.00000079, 0.00000163] |
| clear | 38504 / 96 | JOINT_BIN_added_mass_vs_exact | 0.00000119 | [0.00000079, 0.00000163] |
| clear | 38504 / 96 | JOINT_BIN_removed_mass_vs_exact | 0.00000000 | [0.00000000, 0.00000000] |
| clear | 38504 / 96 | MARGINAL_signed_mass_minus_exact | 0.00001108 | [0.00000795, 0.00001498] |
| clear | 38504 / 96 | MARGINAL_added_mass_vs_exact | 0.00001108 | [0.00000795, 0.00001498] |
| clear | 38504 / 96 | MARGINAL_removed_mass_vs_exact | 0.00000000 | [0.00000000, 0.00000000] |

frame分类：episode按13帧最深类别归组；清晰须13帧全清晰，支持取任一帧。

| 类别 | 分母 | 臂 | 支持n/分母 | 平均质量 | EXACT不支持微格内质量 |
|---|---:|---|---:|---:|---:|
| contact0-2cm | 2314 | EXACT | 1904/2314 | 0.00356209 | 0.00000000 |
| contact0-2cm | 2314 | JOINT_BIN | 1903/2314 | 0.00354603 | 0.00000769 |
| contact0-2cm | 2314 | MARGINAL | 1918/2314 | 0.00393340 | 0.00005744 |
| contact2-5cm | 2692 | EXACT | 2337/2692 | 0.00691751 | 0.00000000 |
| contact2-5cm | 2692 | JOINT_BIN | 2337/2692 | 0.00691129 | 0.00000744 |
| contact2-5cm | 2692 | MARGINAL | 2337/2692 | 0.00694358 | 0.00007513 |
| contact>5cm | 6006 | EXACT | 5764/6006 | 0.01545372 | 0.00000000 |
| contact>5cm | 6006 | JOINT_BIN | 5764/6006 | 0.01543018 | 0.00000498 |
| contact>5cm | 6006 | MARGINAL | 5768/6006 | 0.01487008 | 0.00008355 |
| pass0-10cm | 5396 | EXACT | 0/5396 | 0.00000000 | 0.00000000 |
| pass0-10cm | 5396 | JOINT_BIN | 19/5396 | 0.00000308 | 0.00000308 |
| pass0-10cm | 5396 | MARGINAL | 3683/5396 | 0.00107736 | 0.00107736 |
| clear | 38504 | EXACT | 0/38504 | 0.00000000 | 0.00000000 |
| clear | 38504 | JOINT_BIN | 95/38504 | 0.00000119 | 0.00000119 |
| clear | 38504 | MARGINAL | 2429/38504 | 0.00001108 | 0.00001108 |

episode分类：episode按13帧最深类别归组；清晰须13帧全清晰，支持取任一帧。

| 类别 | 分母 | 臂 | 支持n/分母 | 平均质量 | EXACT不支持微格内质量 |
|---|---:|---|---:|---:|---:|
| contact0-2cm | 289 | EXACT | 285/289 | 0.00183489 | 0.00000000 |
| contact0-2cm | 289 | JOINT_BIN | 287/289 | 0.00183429 | 0.00000775 |
| contact0-2cm | 289 | MARGINAL | 288/289 | 0.00206849 | 0.00005122 |
| contact2-5cm | 298 | EXACT | 296/298 | 0.00386841 | 0.00000000 |
| contact2-5cm | 298 | JOINT_BIN | 297/298 | 0.00386729 | 0.00000739 |
| contact2-5cm | 298 | MARGINAL | 297/298 | 0.00387218 | 0.00004008 |
| contact>5cm | 803 | EXACT | 803/803 | 0.00936869 | 0.00000000 |
| contact>5cm | 803 | JOINT_BIN | 803/803 | 0.00935625 | 0.00000659 |
| contact>5cm | 803 | MARGINAL | 803/803 | 0.00915932 | 0.00016973 |
| pass0-10cm | 591 | EXACT | 0/591 | 0.00000000 | 0.00000000 |
| pass0-10cm | 591 | JOINT_BIN | 0/591 | 0.00000000 | 0.00000000 |
| pass0-10cm | 591 | MARGINAL | 573/591 | 0.00060704 | 0.00060704 |
| clear | 2243 | EXACT | 0/2243 | 0.00000000 | 0.00000000 |
| clear | 2243 | JOINT_BIN | 1/2243 | 0.00000012 | 0.00000012 |
| clear | 2243 | MARGINAL | 842/2243 | 0.00001008 | 0.00001008 |

frame质量为当前帧；episode质量为13帧均值而非最大值。EXACT不支持微格内质量仅描述采样几何差异；擦身/清晰类别的全部query质量均为相对全物体contact的虚假质量。

相同正权射线集合下，JOINT_BIN正支持必然保留为MARGINAL正支持。边缘化可稀释幅度、增加虚假支持，不能单独删除零阈值下的支持；不能据此把全部误差解释为漏检。L1增量允许为负，反映误差抵消。

全物体surface类别包含遮挡表面；可见采样支持与它分开统计。零可见质量不证明清晰，第129类仅几何无命中/域外，不是设备低SNR或产品UNKNOWN。未改动任何既有实验结论。

## 边界与研究参考

E仍是既定有限射线上的首可见几何，不能覆盖遮挡表面、所有连续细结构或探测噪声。即使真分布的读出存在误差，也不证明传感器能够恢复对应。新表示与可学习性、报警收益、跨源泛化均待验证；已消费Development，不能作为实机或安全证据。

相关的[ICCV 2025参数化场景恢复研究](https://openaccess.thecvf.com/content/ICCV2025/papers/Sifferman_Recovering_Parametric_Scenes_from_Very_Few_Time-of-Flight_Pixels_ICCV_2025_paper.pdf)用完整瞬态、已知传感器姿态和强几何先验进行预测及渲染优化。它提示应保留几何对应，但其已知物体模型/分布式观测假设不等于本项目的8×8移动前视任务；没有借用其成绩作为本轮依据。

源码：[cnh_surface_factorization_probe.py](cnh_surface_factorization_probe.py)。payload：`artifacts.local/work/cnh-surface-factorization-probe-20261002/`，保留96单位NPZ/receipt、完整result/REPORT及verification。RUNS一跑一行；无新的报警成功判据。

- source SHA256: `56e3ed062a204ef325aa96d4ab849ae4c2d50f897f9bfb4fb051f21cf6853d75`
- result.json SHA256: `a4257bd9dca21b8c0527870e02004cdf0a6d47c9c33a9091be6798aa170911db`
