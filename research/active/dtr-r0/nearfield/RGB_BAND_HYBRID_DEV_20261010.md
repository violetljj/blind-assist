# RGB 近带 Uni / 中远 DAV hybrid（Development，2026-10-10）

**主门 FAIL**：HYBRID_ARK 近带见证 48/159，DAV-only 1/159；6/6 capture 严格改善，近带绝对 FREE 29/694（4.18%），超过≤2%的要求，允许整数 13（exact 50F≤N）。 本次保留 DAV2 raw R0 的 RGB≥0.8m 职责，近带由 ToF 承担；近带收益与成本完整保留。

中/远带逐 query 直接复用 DAV 决策，包括分数、cutoff、支持与见证，每字段严格一致；这是结构性保留，不是独立测得的中远改进。既有 matchedFREE run 的全模型排序交叉及阶段2 stop保持原结论，本次为另行授权的近带职责混合。

## 预封存协议与唯一校准

PLAN 于2026-10-10 14:37:01 UTC封存，SHA256 `c630ee8caae3ceb229d50d3e5615da813211a83a0cad93f0356dfbcb80c5a56f`。仅三臂：DAV_ONLY 使用原 DAV raw R0 pooled304 cut 0.24403834342956543m；HYBRID_ARK 近0.3–0.8m使用Uni raw R0唯一ARK-global-cal cut、中0.8–1.5m/远1.5–3m原样使用DAV；HYBRID_POOLED只把近带Uni cut换回原pooled304 0.09616100788116455m作为漂移控制。无训练、affine、读出改动或旧推理重跑。

唯一 Uni ARKcal 切点 **0.038585889339447066m**，低于原pooled304 cut，属于阈值放宽；不将它描述为成本收紧。校准仅使用已消费old6+confirm6+original48ARKcal，按(scan,frame,query)真实去重：240帧、2019 strict sampledFREE query，重复去除0；old6贡献841、confirm6贡献622、原48贡献556，排除3RScan FREE32。使用全部三带FREE，未按近带/新eval/POS/UNKNOWN选阈值。支持100/2019≤floor5%100，保留完整ties；原cal48为40777073/40809740/40958733，与old12不同，但仍运行去重检查。

切点于 `2026-10-10T14:40:54.304481+00:00` 封存，root随后启动新模型推理；threshold_seal.json保留切点文件/FREE分数/PLAN SHA。校准无新eval使用、POS/UNKNOWN选择、GPU或训练。所获结论只覆盖新capture Development，不能把消费过的cal重新称为确认。

Fresh按官方CSV Validation物理顺序选首六个未消费visit，reference-only近POS≥16 gate，每capture16个时间排序均匀对齐帧包含端点，共96帧；不足才允许Training fallback。实际选择/skip/fallback见source_summary.json与官方source收据。新公开域为注册256×192 grid有限几何域，不取两模型valid交集；invalid depth丢支持但不删query。native positive witness按冻结参考正射线与原16th统计量。

主门只要求完整六capture、至少4个近W严格优于DAV，且HYBRID_ARK近带**绝对**误支持率≤2%（exact50F≤N）；不是FREE净增门，也不是≤2%加额外逐capture门。分母0或六capture不完整为NOT_EVALUABLE，不升级。冻结cal与control都不会由描述曲线选择工作点。

## 公开来源与固定模型身份

官方选择使用 [ARKitScenes raw split CSV](https://github.com/apple-aiml-research/ARKitScenes/blob/main/raw/raw_train_val_splits.csv) 与官方Range资产，24个已消费visit/capture排除清单及逐请求HTTP206、Content-Range、CRC/SHA保存在source_summary.json、consumed_inventory.json、official_receipts.json与acquisition_terminal.json。六个新候选均通过覆盖门，无本次跳过，无Training fallback。CSV SHA256 `5a817c0ee8150cea1de4fa9d6399880eaa646c2b62326c3314e674d201536c57`。

| capture | visit | CSV物理行序（0基） | 近POS |
| --- | --- | --- | --- |
| 41159538 | 381868 | 427 | 20 |
| 41159553 | 381888 | 430 | 29 |
| 41159566 | 381872 | 434 | 21 |
| 41254246 | 382857 | 514 | 23 |
| 41254382 | 382858 | 517 | 44 |
| 41254400 | 382854 | 520 | 22 |

两模型均复用官方固定revision缓存，保持FP32权重/FP16 autocast原推理方式；Uni使用公开K。完整code/weight许可证和先前版本收据由本次license_receipts.json逐项路径与SHA绑定。

| 模型 | 官方code SHA | 官方HF revision | weight SHA256 | 许可证 |
| --- | --- | --- | --- | --- |
| [DAV2 Metric Indoor Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/tree/79720800638389a78b2defc92caa885104f69974) | a561b849ebae10a6f5ef49e26c83cbbcd36c71bf | 79720800638389a78b2defc92caa885104f69974 | 6f82ff2bc543ac02ddff4aa31fa363676a8305dd3ccf04e80e2af115a044cb6d | metric Large权重CC-BY-NC-4.0；代码Apache-2.0 |
| [UniDepth V2 ViT-L](https://huggingface.co/lpiccinelli/unidepth-v2-vitl14/tree/52b349b514bd8b47642f67ac78cb7b5dc5c51dd9) | 8d8cfe4c7ee15297099983607febf0d4f32eb3d6 | 52b349b514bd8b47642f67ac78cb7b5dc5c51dd9 | ba73d3de735302ccc64a50f1e557122050c4b1893e6060b28dba05d6af3e67c6 | CC-BY-NC-4.0 |

[Apple官方研究许可证](https://github.com/apple-aiml-research/ARKitScenes/blob/main/LICENSE)本次重新下载逐字节一致，SHA256 `1b6a8700127de50c9d56f8f33eb202a64f6f212fd4b133435f7c8b6bccd3db59`。数据与权重仅本地非商业研究，不再分发；receipt保留原文而不推导额外权限。

## 新六capture近带与逐query配对

| capture | arm | W/POS | POS支持/POS | FREE/FREE | UNKNOWN/UNKNOWN | 相对DAV救/损 | FREE增/减 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bandhybrid_arkit_41159538 | DAV_ONLY | 0/20 | 0/20 | 0/122 | 0/2 | - | - |
| bandhybrid_arkit_41159538 | HYBRID_ARK | 2/20 | 2/20 | 0/122 | 0/2 | 2/0 | 0/0 |
| bandhybrid_arkit_41159538 | HYBRID_POOLED | 1/20 | 1/20 | 0/122 | 0/2 | 1/0 | 0/0 |
| bandhybrid_arkit_41159553 | DAV_ONLY | 0/29 | 0/29 | 0/112 | 0/3 | - | - |
| bandhybrid_arkit_41159553 | HYBRID_ARK | 9/29 | 9/29 | 1/112 | 0/3 | 9/0 | 1/0 |
| bandhybrid_arkit_41159553 | HYBRID_POOLED | 6/29 | 6/29 | 0/112 | 0/3 | 6/0 | 0/0 |
| bandhybrid_arkit_41159566 | DAV_ONLY | 0/21 | 0/21 | 0/121 | 0/2 | - | - |
| bandhybrid_arkit_41159566 | HYBRID_ARK | 6/21 | 9/21 | 3/121 | 0/2 | 6/0 | 3/0 |
| bandhybrid_arkit_41159566 | HYBRID_POOLED | 3/21 | 4/21 | 1/121 | 0/2 | 3/0 | 1/0 |
| bandhybrid_arkit_41254246 | DAV_ONLY | 1/23 | 1/23 | 1/121 | 0/0 | - | - |
| bandhybrid_arkit_41254246 | HYBRID_ARK | 13/23 | 13/23 | 13/121 | 0/0 | 12/0 | 12/0 |
| bandhybrid_arkit_41254246 | HYBRID_POOLED | 8/23 | 8/23 | 4/121 | 0/0 | 7/0 | 3/0 |
| bandhybrid_arkit_41254382 | DAV_ONLY | 0/44 | 0/44 | 0/98 | 0/2 | - | - |
| bandhybrid_arkit_41254382 | HYBRID_ARK | 14/44 | 14/44 | 11/98 | 0/2 | 14/0 | 11/0 |
| bandhybrid_arkit_41254382 | HYBRID_POOLED | 8/44 | 8/44 | 6/98 | 0/2 | 8/0 | 6/0 |
| bandhybrid_arkit_41254400 | DAV_ONLY | 0/22 | 0/22 | 0/120 | 0/2 | - | - |
| bandhybrid_arkit_41254400 | HYBRID_ARK | 4/22 | 4/22 | 1/120 | 1/2 | 4/0 | 1/0 |
| bandhybrid_arkit_41254400 | HYBRID_POOLED | 1/22 | 1/22 | 0/120 | 0/2 | 1/0 | 0/0 |

## 三带合计与漂移控制

| 带 | arm | W/POS | POS支持/POS | FREE/FREE | UNKNOWN/UNKNOWN | 相对DAV救/损 | FREE增/减 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3-0.8m | DAV_ONLY | 1/159 | 1/159 | 1/694 | 0/11 | - | - |
| 0.3-0.8m | HYBRID_ARK | 48/159 | 51/159 | 29/694 | 1/11 | 47/0 | 28/0 |
| 0.3-0.8m | HYBRID_POOLED | 27/159 | 28/159 | 11/694 | 0/11 | 26/0 | 10/0 |
| 0.8-1.5m | DAV_ONLY | 208/634 | 225/634 | 0/208 | 12/22 | - | - |
| 0.8-1.5m | HYBRID_ARK | 208/634 | 225/634 | 0/208 | 12/22 | 0/0 | 0/0 |
| 0.8-1.5m | HYBRID_POOLED | 208/634 | 225/634 | 0/208 | 12/22 | 0/0 | 0/0 |
| 1.5-3m | DAV_ONLY | 401/606 | 445/606 | 0/36 | 76/222 | - | - |
| 1.5-3m | HYBRID_ARK | 401/606 | 445/606 | 0/36 | 76/222 | 0/0 | 0/0 |
| 1.5-3m | HYBRID_POOLED | 401/606 | 445/606 | 0/36 | 76/222 | 0/0 | 0/0 |

HYBRID_ARK与HYBRID_POOLED的近带差异体现同一主干对ARK全局cal与旧pooledcal的工作点迁移。它不是多次搜索后挑出的阈值；两个切点均先于新模型输出固定。ARK相对pooled控制新增21个见证、18个FREE支持；全带5%校准本身不保证新近带≤2%的采用目标。

固定pooled漂移对照在新六capture同样全部近带W严格增加，FREE11/694（1.585%）低于2%，但预声明主门只对HYBRID_ARK生效；不在ARK主臂失败后将control换成主判，当前代表不据此升级。这个控制结果完整保留，表明本次校准放宽增加了成本，而不是否定Uni近带信息。

| 主门 | 测量 | 条件 |
| --- | --- | --- |
| 六capture完整 | True | True |
| near严格改善 | 6 | ≥4/6 |
| near绝对FREE | 29/694 | 50×29≤694 |
| 结论 | FAIL | 要求全部主条件通过；本次FREE条件失败 |

## 每capture中远保留检查

| capture | 带 | DAV W/POS | DAV FREE/FREE | DAV UNKNOWN/UNKNOWN | hybridARK/pooled相对DAV |
| --- | --- | --- | --- | --- | --- |
| bandhybrid_arkit_41159538 | 0.8-1.5m | 26/79 | 0/64 | 0/1 | 逐query全字段一致；救/损与FREE增/减0/0 |
| bandhybrid_arkit_41159538 | 1.5-3m | 72/116 | 0/16 | 1/12 | 逐query全字段一致；救/损与FREE增/减0/0 |
| bandhybrid_arkit_41159553 | 0.8-1.5m | 33/99 | 0/42 | 1/3 | 逐query全字段一致；救/损与FREE增/减0/0 |
| bandhybrid_arkit_41159553 | 1.5-3m | 84/100 | 0/8 | 10/36 | 逐query全字段一致；救/损与FREE增/减0/0 |
| bandhybrid_arkit_41159566 | 0.8-1.5m | 44/113 | 0/31 | 0/0 | 逐query全字段一致；救/损与FREE增/减0/0 |
| bandhybrid_arkit_41159566 | 1.5-3m | 73/98 | 0/0 | 18/46 | 逐query全字段一致；救/损与FREE增/减0/0 |
| bandhybrid_arkit_41254246 | 0.8-1.5m | 44/112 | 0/29 | 0/3 | 逐query全字段一致；救/损与FREE增/减0/0 |
| bandhybrid_arkit_41254246 | 1.5-3m | 69/107 | 0/4 | 6/33 | 逐query全字段一致；救/损与FREE增/减0/0 |
| bandhybrid_arkit_41254382 | 0.8-1.5m | 42/121 | 0/11 | 10/12 | 逐query全字段一致；救/损与FREE增/减0/0 |
| bandhybrid_arkit_41254382 | 1.5-3m | 45/73 | 0/5 | 24/66 | 逐query全字段一致；救/损与FREE增/减0/0 |
| bandhybrid_arkit_41254400 | 0.8-1.5m | 19/110 | 0/31 | 1/3 | 逐query全字段一致；救/损与FREE增/减0/0 |
| bandhybrid_arkit_41254400 | 1.5-3m | 58/112 | 0/3 | 17/29 | 逐query全字段一致；救/损与FREE增/减0/0 |

程序核对两hybrid共3456个中远query全字段相同；保持原DAV的收益、误支持和未知，不能扩展为模型安全或普适改进。

## Near曲线与真实冻结工作点

每capture及aggregate扫描原始有限R0 query/positive-witness完整ties，以实际FREE计数给描述W上包络；invalid −inf永不支持，不拆ties、不插值。图中方块/星/X分别是三个**实际冻结工作点**（直接计score≥已封存cut），与上包络分开，不能把同FREE的描述最大W误认实际部署W。

![六capture及合计near曲线与冻结标记](../../../../artifacts.local/work/rgb-band-hybrid-dev-20261010/final/near_capture_and_aggregate.png)

PNG及独立PDF、near_all_ties.csv、near_envelope.csv、near_operational_points.csv、near_curves.json可独立复建；未因曲线改cutoff。

| capture/aggregate | 共同精确F点 | Uni−DAV最小W差 | Uni较低实际F连续区间 | 低于DAV点数 |
| --- | --- | --- | --- | --- |
| bandhybrid_arkit_41159538 | 123 | -3 | [[14, 35]] | 22 |
| bandhybrid_arkit_41159553 | 111 | -3 | [[0, 1], [3, 3], [15, 30]] | 19 |
| bandhybrid_arkit_41159566 | 120 | -4 | [[0, 2], [34, 35]] | 5 |
| bandhybrid_arkit_41254246 | 117 | -7 | [[0, 12], [23, 59], [61, 68]] | 58 |
| bandhybrid_arkit_41254382 | 94 | -23 | [[0, 22], [24, 44], [47, 50], [52, 63], [65, 81]] | 77 |
| bandhybrid_arkit_41254400 | 116 | -1 | [[84, 96]] | 13 |
| aggregate_fresh | 615 | -31 | [[0, 11], [13, 49], [106, 121], [123, 141], [143, 145], [147, 148], [151, 152], [154, 157], [159, 162], [164, 168], [170, 176], [179, 183], [185, 196], [200, 203], [205, 212], [215, 230], [232, 234], [237, 244], [247, 249], [253, 258], [260, 262], [265, 266], [270, 279], [281, 290], [292, 299], [301, 305], [307, 312], [315, 324], [327, 344], [346, 349], [351, 352], [355, 362], [364, 374], [376, 379], [383, 386], [388, 399], [543, 557], [559, 567]] | 317 |

新near曲线的这些共同精确F交叉完整保留（没有插值或拆ties），不能由旧数据near支配推断新capture全域支配。它们仅是事后描述，不加入主门，也不挑新cutoff。完整点表为report-summary/near_common_actual_FREE.csv。

## 部署、许可证、资源与独立复算

两模型成功同时驻留GPU，同帧按DAV→Uni顺序执行，核函数不重叠；实际参数 DAV335315649、Uni353831043，合计689146692。95个warm帧combined median0.2712968s/P95 0.3350201s；DAV median0.2005376/P95 0.2472577s，Uni median0.0718231/P95 0.0859019s。首cold帧单列：combined17.7493374s（DAV11.5944323s、Uni6.1548663s）。同步测量含预处理、network、CPU拷贝，不含native mapping/文件写入。peak allocated3339468800B、reserved3487563776B；GPUowned wall65.4916927s，只计一次双模型进程，CPU orchestration command67.0518553s归root。deployment_summary.json保留完整数据，PID73916已检查不存在，无task-owned常驻GPU。实际两模型推理仅针对本次新96帧。模型及ARKit许可证/版本/URL/SHA沿用已固定官方收据，仅本地非商业研究，不再分发权重。

实际下载 **77,287,325/1,000,000,000 B**（含官方metadata/许可证151,157 B），仅Range子集、无权重新下载；GPU拥有进程墙钟 **65.492/600 s**。CPU按command-wall保守计账 **355/1200 s**：采集/适配80、评价/报告45、独立核验50、根代理推理编排/查读/目视/整合/交付180（其中实测GPU编排shell67.052 s，含在180中，不重复计）。账本包含失败，非CPU内核使用时间；所有任务workers已结束。产物保留在canonical artifacts.local/work/rgb-band-hybrid-dev-20261010。

一次独立核验从新native预测与冻结参考重建R0分数，验证校准来源与整组ties、官方选源/CRC与标签、逐query决策/救损、合计/主门与描述曲线：**PASS**，详见independent_audit_final.json及源码/终端收据。独立cal首检查误把pooled_FREE_scores.json归入old-final分数文件列表，失败independent_cal_audit.json保留，按父目录修复后attempt2通过。Fresh首检查的独立summary带额外capture别名，与生产environment分组schema不一致，首FAIL independent_audit.json保留；只适配分组schema并续未完成节点，已通过来源/首capture数学未重复。两个消费者修复均未改变科学配方、阈值或原数值。核验复算已有预测，未独立重跑神经模型。

主要产物：calibration/thresholds.json、threshold_seal.json、deduplicated_FREE_scores.json与duplicates/sourceSHA收据；新native predictions及public/reference manifests；final/<cohort>_scores.json、<cohort>_evaluation.json、results.json、summary.csv、query_pairs.csv、summary.json、near曲线CSV/PNG/PDF；run中的独立审计、GPU部署/释放和最终预算收据。源码为 [rgb_band_hybrid_evaluate.py](rgb_band_hybrid_evaluate.py)，cal/final执行快照与输入SHA均保留。

本run固定配方已结束，不在这六条新eval上调阈值、改cal子集或改主门。当前DAV2代表与ToF近带主力职责保留；pooled控制的收益/成本是后续独立决策可参考的Development证据。
