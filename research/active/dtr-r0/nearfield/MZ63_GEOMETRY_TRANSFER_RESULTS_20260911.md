# MZ63：固定权重对新几何的迁移结果

EXPLORE、consumed controlled Development。预注册 **DROP / HELDOUT_GEOMETRY1024 两臂描述旗标均PASS**：相对OLD_NEG，CONTROL新增8个native BODY_NEAR，COVERAGE新增1个，四query均无新增FP。旗标不选择赢家，也不掩盖其他profile的误报成本。

全4096帧有16,384个已知query bits，其中4096正、12,288负；每query1024正。HELD1024帧有1024正bits（每query256）。TRAIN2048/CAL1024/HELD1024全是此次描述性分组，未参与任何新fit、normalization、cutoff或模型选择。

表中为TP/FP，C、V均指与OLD_NEG的最终OR；OLD_NEG与保留MZ57在本源决策相同。

| Profile / 帧数 | OLD_NEG＝MZ57 | CONTROL OR | COVERAGE OR |
| --- | ---: | ---: | ---: |
| IDEAL / 4096 | 3514/274 | 3535/313 | 3529/300 |
| IDEAL / HELD1024 | 865/70 | 871/80 | 868/77 |
| MERGE_CLOSE / 4096 | 3513/118 | 3521/126 | 3513/121 |
| MERGE_CLOSE / HELD1024 | 860/31 | 863/32 | 860/31 |
| DROP_CLOSE / 4096 | 2985/154 | 3034/155 | 3005/154 |
| DROP_CLOSE / HELD1024 | 740/38 | 756/38 | 743/38 |
| ALL_INVALID / 4096 | 17/16 | 71/16 | 20/16 |
| ALL_INVALID / HELD1024 | 5/4 | 10/4 | 5/4 |

四query顺序为BODY_NEAR/BODY_FAR/HEAD_NEAR/HEAD_FAR。全源DROP，CONTROL新增TP=[35,0,12,2]、FP=[0,0,1,0]；COVERAGE新增TP=[11,0,9,0]、FP全0。HELD分别新增TP=[12,0,3,1]和[2,0,1,0]，FP全0；相对OLD_NEG/MZ57均无TP损失或FP移除。原基线全源FP=[0,31,24,99]、HELD=[0,10,5,23]，绝非零风险基线。唯一新增DROP FP是CALIBRATION的variable_rod BODY_ONLY near g02 unsupported，被已知错位置判为HEAD_NEAR。IDEAL全源C/V另加39/26 FP，MERGE加8/3 FP，均保留。

**新增native贡献有明确范围。** 全源DROP的BODY_NEAR新增：C为27native/2已知错位置/6局部UNKNOWN，V为9/1/1；HELD为8/1/3和1/1/0。C的held8native来自sign_panel7、square_grille1，V的1来自sign_panel；**held awning没有native BODY_NEAR新增**，C的3个awning新增赢家均局部UNKNOWN。C还新增3个held HEAD_NEAR（1native、2错位置）和1个HEAD_FAR（UNKNOWN）；V多1个错位置HEAD_NEAR。计分只给“候选阳性、比较器阴性、真实已知事件、native赢家”的新分支，不把继承OR命中算新贡献。

held BODY_NEAR全部256正例中，C/V赢家native数量162/164，低于原cutoff的native数量97/141；后者包含旧基线已命中行，不能直接当最终FN。完整native/已知错位置/UNKNOWN、support与margin分组见保存结果。DROP仍有477/475个C/V support pairs变决策（旧union488/2048），而原生事件标签保持一致。

**ALL_INVALID并未解决无有效回波。** ranges/valid全零，三head均0个可用anchor、4096个全零向量。C全源仅71/4096正bits被检出，V20/4096；C新增54个BODY_NEAR（51native/2错位置/1UNKNOWN），V新增3个native。HELD新增5（4native/1错位置）与0。完全缺失条件下仍有少量检出，但远未建立可用的备用能力；未推定真实传感器失效率或硬件性能。

MZ61的64个held BODY_NEAR正配置在count及presence上均未与本源TRAIN精确重复。这限定了未来使用MZ61训练时的精确标签重复风险；本次模型未用这些TRAIN行训练，不能据此排除与旧训练源相似。每配置四个site/support副本并不独立。整张presence仍有21/192个held正配置重复，HEAD_FAR为29/64。标签非重复不证明统计独立或泛化，也未比较所有旧训练源。MZ62两臂是等预算对照；与600步旧模型比较同时改变预算和采样，不能单独归因于profile安排。

冻结8decisions全部保留：MZ37、OLD_NEG/UNION、MZ56 GLOBAL候选及MZ57 union、MZ62 CONTROL/COVERAGE候选及各自union。未加入新训练、阈值搜索或后继实验；已有MZ60负对照不因本结果被抹去。没有自然场景、Android部署或安全有效性声明。

完整证据：[result.json](../../../../artifacts.local/work/mz63-geometry-transfer-20260911/score-v1/result.json)、[逐事件与support pairs](../../../../artifacts.local/work/mz63-geometry-transfer-20260911/score-v1/paired-events.json)、[独立audit](../../../../artifacts.local/work/mz63-geometry-transfer-20260911/score-v1/audit.json)、[解释明细](../../../../artifacts.local/work/mz63-geometry-transfer-20260911/score-interpretation-v1/summary.json)、[源新颖性审计解释](../../../../artifacts.local/work/mz61-geometry-source-20260911/source-novelty-interpretation-v1.md)。结果SHA `ac3d9a6cbbc6da861d9bf80a2c2972f21bc0765119ea86ab7e55dbfc342edc81`；score receipt SHA `1135e68db083f89688fb5d93d3f021cf09b0b63f9fc71501abb23fee5fa55305`。
