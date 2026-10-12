# bin级查询读出原型：有限形状互补有效，signed-face增量未获支持

2026-10-09，EXPLORE、已消费模拟Development。原M3与5格融合均保留。

**本轮最值得继续的结果是bin级token与既有局部读出互补：保留原5格融合报警，加入简化消融头后，HEAD543→562/688、BODY461→481/688，分别救回19/20、损失0/0；ideal联合clear仍46/4576。** 但pass clips由74→101/352，+3°固定阈值clear从旧融合175升到191/4576（M3为176），消融头held-unit macroAUC .839265低于M3 .868327。因此保留条件候选、继续冻结M3和旧融合，不能称普遍提升。

**机制解释已经更正：移除六个signed query-face通道的匹配消融反而更强（578/510），超过完整中心头552/496。当前不支持signed-face坐标或extent角域扩展是收益来源；更可信的探索方向是简化的bin级token/readout结构和已有局部证据互补。**

## 机制与公平对照

第一种表示为16个距离bin分别与身体查询软覆盖交互，再汇总历史均值／当前帧并做zone空间编码；center重复中心射线，extent保留zone的3×3 slope内点及覆盖mean/min/max。四臂共用相同39936训练行、FP16缓存、随机初态、seed、row顺序、优化器、epochs和校准规则。pair只在同一观测的HEAD/BODY均有效且标签不同时加入query contrast。它不是背景替换或障碍移除反事实；完整原表示／新表示×原监督／真正反事实监督四格尚未执行。

12轮四臂均退步，在线BCE仍下降，于是保留v1并做一次共同拟合检查：从各自12轮checkpoint继续52轮，四臂同样重置AdamW状态，最终64轮checkpoint；未用评价选择epoch。训练拟合改善但aligned检出继续退步，本轮结束该覆盖头拟合检查。

第二种机制直接保留每个radial bin的signed query-face margins及观测证据，让共享token编码器先学二者交互，随后才压缩bin与空间邻域。两个新头都是普通BCE、相同初态/seed/训练行/32轮/row顺序/预算；center与extent之间为匹配机制对照。signed-boundary与第一种覆盖头网络及特征不同，其差异支持探索方向，不能独立归因于某一特征。M3是已有预训练五seed均值，5格融合是冻结实用参考，均非与新头匹配的随机训练机制对照。

随后做一个匹配训练消融：仅置零通道6:12六个signed query-face margin，保留membership、outside distance、radius、原生bin证据；网络名义8833参数、同init/data/row顺序/BCE/32epoch/最终checkpoint。消融是在看到完整头结果后选择的Development探索，不是预注册确认。

## ideal全批结果

原492场景×K4×13输出帧f3–15；HEAD/BODY接触各688事件，及时f3–13、f14–15晚。联合clear4576格、352 clips，阈值使用完整joint-clear格整tie、nearest46，残差保留。下表两高度分母均688，gain/loss是相对原M3的逐scene×replica配对。pass分母为存在pass且无任何contact的88场景×K4=352 clips；physical contact-any-height分母1264。clear段／clip、pass成本与物理及时检出均独立报告，同clear46不表示全部误报成本不变。

| 阶段／臂 | HEAD/688 | BODY/688 | clear/4576 | clear segments | clear clips | pass/352 | physical timely/1264 | HEAD gain/loss | BODY gain/loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M3 frozen | 517 | 406 | 46 | 25 | 24 | 74 | 849 | — | — |
| frozen5slot fusion | 543 | 461 | 46 | 25 | 23 | 74 | 921 | 28/2 | 56/1 |
| pilot12 / center_bce | 479 | 352 | 46 | 40 | 36 | 59 | 779 | 38/76 | 37/91 |
| pilot12 / extent_bce | 468 | 339 | 46 | 31 | 30 | 72 | 750 | 38/87 | 44/111 |
| pilot12 / center_pair | 423 | 317 | 46 | 38 | 33 | 45 | 692 | 23/117 | 28/117 |
| pilot12 / extent_pair | 448 | 330 | 46 | 30 | 28 | 65 | 726 | 39/108 | 39/115 |
| fit64 / center_bce | 427 | 330 | 46 | 35 | 30 | 60 | 708 | 26/116 | 31/107 |
| fit64 / extent_bce | 437 | 341 | 46 | 35 | 33 | 72 | 723 | 34/114 | 50/115 |
| fit64 / center_pair | 394 | 292 | 46 | 33 | 27 | 35 | 643 | 22/145 | 23/137 |
| fit64 / extent_pair | 420 | 314 | 46 | 35 | 32 | 69 | 678 | 25/122 | 33/125 |
| signed-boundary32 / boundary_center | 552 | 496 | 46 | 36 | 31 | 138 | 971 | 65/30 | 107/17 |
| signed-boundary32 / boundary_extent | 543 | 485 | 46 | 36 | 33 | 135 | 960 | 61/35 | 101/22 |
| matched-ablation32 / boundary_no_signed_faces | 578 | 510 | 46 | 33 | 28 | 139 | 1001 | 77/16 | 117/13 |
| new5slot + fullcenter | 558 | 476 | 46 | 26 | 25 | 118 | 956 | 43/2 | 72/2 |
| new5slot + ablation | 575 | 498 | 46 | 26 | 25 | 126 | 991 | 59/1 | 93/1 |
| old5slot OR fullcenter | 552 | 468 | 46 | 25 | 23 | 96 | 936 | 36/1 | 63/1 |
| old5slot OR ablation | 562 | 481 | 46 | 25 | 23 | 101 | 959 | 46/1 | 76/1 |

中心signed-boundary相比extent净多HEAD9、BODY11；角域扩展在ideal检出上没有胜过中心射线。不把得到552/496的条件结果解释为角度—距离联合空间重建已成功。匹配full−ablation的HEAD救9损35、BODY救8损22，净−26/−14；signed-face增量解释撤回。

四个融合点均按已确定的单点公式执行，没有扫描候选预算或阈值：new5slot让M3释放5格后加入新头；old5slot OR新头保留全部旧报警，新头阈值设为ideal剩余clear最大分数以上的nextafter。该阈值在已消费Development上选择，zero-extra仅指本批ideal没有新增joint-clear格，不是部署零误报证书。

| 融合 | 分支 | 相对旧5格HEAD gain/loss | 相对旧5格BODY gain/loss |
| --- | --- | --- | --- |
| new5slot + fullcenter | ideal | 26/11 | 32/17 |
| new5slot + fullcenter | query_yaw_plus3_fixed_threshold | 23/19 | 20/23 |
| new5slot + ablation | ideal | 42/10 | 52/15 |
| new5slot + ablation | query_yaw_plus3_fixed_threshold | 34/12 | 44/21 |
| old5slot OR fullcenter | ideal | 9/0 | 7/0 |
| old5slot OR fullcenter | query_yaw_plus3_fixed_threshold | 6/0 | 4/0 |
| old5slot OR ablation | ideal | 19/0 | 20/0 |
| old5slot OR ablation | query_yaw_plus3_fixed_threshold | 14/0 | 13/0 |

暗4cm横杆接触每高度56事件。new5slot+fullcenter虽整体增加，却BODY2/56低于旧局部读出7/56，不能全面替代旧融合。保留原报警的OR候选保存该能力：

| 臂 | 暗4cm HEAD/56 | 暗4cm BODY/56 |
| --- | --- | --- |
| M3 | 8 | 1 |
| frozen5slot local | 12 | 7 |
| new5slot + fullcenter | 8 | 2 |
| new5slot + ablation | 14 | 4 |
| old5slot OR fullcenter | 12 | 7 |
| old5slot OR ablation | 12 | 7 |

## 训练外48单位排序

同生成器221001–221071中mode0/1，共48个训练外unit；7横移×K4×13帧，17472行。原last5因果平滑；目标query按scene等权macroAUC；主域1.2≤front<2.1m，侵入1/2cm正，外10/15/20cm负。外10cm在训练被mask屏蔽但作为原评价负类，保留该原口径。这些单位以前已消费，不能称独立确认；不能与aligned及时率互换解释。

| 阶段／臂 | macroAUC | 对M3配对均值差 |
| --- | --- | --- |
| M3 frozen | 0.868327 | 0.000000 |
| fit64 / center_bce | 0.688856 | -0.179470 |
| fit64 / center_pair | 0.696982 | -0.171345 |
| fit64 / extent_bce | 0.754732 | -0.113595 |
| fit64 / extent_pair | 0.745672 | -0.122655 |
| signed-boundary32 / boundary_center | 0.845468 | -0.022859 |
| signed-boundary32 / boundary_extent | 0.852141 | -0.016186 |
| matched-ablation32 / boundary_no_signed_faces | 0.839265 | -0.029062 |

bin-token完整头和消融缩小覆盖头的描述性排序缺口，extent排序略优于center，但均未超过M3；消融.839265还低于完整center .845468。该证据不支持“有限形状提升已迁移到一般新场景”。

## +3°query扰动：固定ideal阈值的实际成本

完全同physical.hist和物理truth，只有public query左乘Ry(+3°)；使用Qprime[f]@inverse(sensor[f])@sensor[past]。原M3缓存query_3_+1核对HEAD467/BODY375、clear176/4576。下面沿用各候选ideal阈值，无扰动后重校准，gain/loss相对相同扰动M3。

| 阶段／臂 | HEAD/688 | BODY/688 | clear/4576 | clear segments | clear clips | pass/352 | physical timely/1264 | HEAD gain/loss | BODY gain/loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M3 original threshold | 467 | 375 | 176 | 57 | 52 | 99 | 787 | — | — |
| frozen5slot fusion original threshold | 500 | 409 | 175 | 57 | 52 | 108 | 842 | — | — |
| fit64 / center_bce | 406 | 318 | 85 | 61 | 49 | 88 | 683 | 34/95 | 39/96 |
| fit64 / extent_bce | 393 | 331 | 101 | 60 | 49 | 96 | 678 | 29/103 | 41/85 |
| fit64 / center_pair | 367 | 284 | 81 | 53 | 43 | 77 | 606 | 20/120 | 25/116 |
| fit64 / extent_pair | 394 | 328 | 106 | 61 | 49 | 102 | 675 | 32/105 | 39/86 |
| signed-boundary32 / boundary_center | 509 | 415 | 169 | 77 | 56 | 126 | 858 | 63/21 | 72/32 |
| signed-boundary32 / boundary_extent | 493 | 429 | 213 | 110 | 84 | 140 | 857 | 52/26 | 79/25 |
| matched-ablation32 / boundary_no_signed_faces | 537 | 443 | 227 | 97 | 68 | 148 | 901 | 79/9 | 93/25 |
| new5slot + fullcenter | 504 | 406 | 201 | 75 | 64 | 115 | 845 | 42/5 | 33/2 |
| new5slot + ablation | 522 | 432 | 234 | 93 | 75 | 127 | 887 | 57/2 | 58/1 |
| old5slot OR fullcenter | 506 | 413 | 176 | 58 | 53 | 109 | 852 | 43/4 | 40/2 |
| old5slot OR ablation | 514 | 422 | 191 | 66 | 60 | 117 | 865 | 51/4 | 49/2 |

中心signed-boundary的clear格169低于176，但段77高于57、clips56高于52；extent的格213、段110、clips84都高于M3。消融单头clear227，new5slot+消融234，旧融合OR消融191；旧融合OR完整center的clear176虽然与M3相同，segments58/clips53仍高于M3的57/52。不能只按格成本称更少提示或更安全。人工单轴3°压力不是实测误差分布。

仅作补充：在此已消费扰动批上重新选nearest176阈值的描述结果如下；它不代替上表固定ideal阈值结果。

| 阶段／臂 | HEAD/688 | BODY/688 | clear/4576 | 对176残差 |
| --- | --- | --- | --- | --- |
| fit64 / center_bce | 452 | 374 | 176 | 0 |
| fit64 / extent_bce | 439 | 374 | 176 | 0 |
| fit64 / center_pair | 428 | 349 | 176 | 0 |
| fit64 / extent_pair | 434 | 356 | 176 | 0 |
| signed-boundary32 / boundary_center | 510 | 419 | 176 | 0 |
| signed-boundary32 / boundary_extent | 482 | 412 | 176 | 0 |

## 拟合与失败结果保留

| 阶段／臂 | 最终online hard BCE | pooled valid train AUC |
| --- | --- | --- |
| pilot12 / center_bce | 0.352616 | NOT_RUN |
| pilot12 / center_pair | 0.349429 | NOT_RUN |
| pilot12 / extent_bce | 0.350674 | NOT_RUN |
| pilot12 / extent_pair | 0.347592 | NOT_RUN |
| fit64 / center_bce | 0.195214 | 0.931325 |
| fit64 / center_pair | 0.205170 | 0.933086 |
| fit64 / extent_bce | 0.185500 | 0.936682 |
| fit64 / extent_pair | 0.194062 | 0.936932 |
| signed-boundary32 / boundary_center | 0.345397 | NOT_RUN |
| signed-boundary32 / boundary_extent | 0.337926 | NOT_RUN |
| matched-ablation32 / boundary_no_signed_faces | 0.352832 | NOT_RUN |

online BCE含训练期间参数变化；训练pooled AUC是train拟合诊断，与held scene-equal AUC和及时事件不是同指标。12→64轮覆盖头loss下降不伴随能力改善；本轮收敛，不再训练，不删除旧负结果。未来不同监督和机制仍开放。执行失败记录数为2；算法负结果不被记为执行错误。所有模型、raw、loss、指标与事件ledger分别保留。

## 执行成本与证据边界

| GPU命令阶段 | 墙钟秒 |
| --- | --- |
| pilot12 | 52.328 |
| fit64 | 65.234 |
| fit64 supplement | 23.672 |
| signed-boundary | 198.031 |
| signed-face ablation | 58.344 |

成功GPU实验阶段墙钟**397.609秒**；两次启动前失败**5.704秒**也计入预算，合计**403.313/1200秒**。启动失败分别为括号语法和utf8-sig编码alias错误，均未分配GPU，原source/log已保留、修复后成功；这些耗时含特征、训练、推理与阶段内分析，非CUDA核函数纯耗时或GPU计费时长。使用本地RTX5060 Laptop、float32网络运算、FP16缓存、TF32关闭；本报告只读JSON/绘图，没有再跑网络。

最终独立审计PASS：NumPy重算平滑、阈值整tie、完整指标、逐事件keys与ledger、held每scene AUC及四融合OR公式；输入SHA/源码边界/消融公平性审查通过。11项模型检查通过，验证墙钟9.993秒另列；CUDA清单确认无任务Python常驻进程。

全部为模拟AABB、固定有限背景和相关K/帧Development；一seed新模型；matched thresholds在已消费批选择。本轮保留native距离证据及查询关系，不读取evaluator目标尺寸、反射率、背景模板或类别为网络特征。角度footprint是查询代理，不是恢复subzone目标位置、物理传感器测量或定位证明。未执行实机、用户、安全验证、保护test或480确认重跑；M3/CNH实机效果仍未建立。

## 下一决定

推荐继续探索简化bin级token/readout与既有局部读出的互补；当前优先旧5格OR消融候选：ideal562/481，相对旧融合救19/20损0/0，保留暗4cm能力，但pass101/352、+3°clear191/4576以及held排序低于M3限定其用途。原M3和5格融合继续冻结，本轮训练结束；这不是未来加seed/epoch或不同机制的永久停止规则。下一步检查和补足训练中的“相似原生观测、不同身体查询结果”监督覆盖，明确train-only反事实配对，然后在不同结构／背景且不参与选择的验证数据上看同成本救回及损失是否保留。真正反事实干预必须与现有H/B querycontrast分开报告；角域扩展需胜过匹配center才能归因其价值。

载荷位置：`artifacts.local/work/cnh-delayed-query-dev-20261009/`；v1、fit64、boundary_tokens各自`evaluation/metrics.json`及event_ledger.csv；fit64/supplement、boundary_tokens、boundary_ablation的yaw/fresh raw和metrics；四融合metrics/ledger、阶段PLAN/receipt/loss、启动失败source/log、source_snapshot及final_audit。

派生特征缓存已保存逐文件SHA/bytes到feature_cache_manifest.json后清理：16文件，20,457,457,664 bytes（19.05 GiB），删除并验证。原始hist、normalized histories、模型、raw、ledger与失败记录全部保留；本报告生成无需feature缓存。

## 源码与复现入口

模型：[覆盖头](cnh_delayed_query_model.py)、[bin-token头](cnh_boundary_token_model.py)；运行：[pilot](cnh_delayed_query_dev.py)、[fit64](cnh_delayed_query_fit_dev.py)、[token对照](cnh_boundary_token_dev.py)、[消融](cnh_boundary_token_ablation_dev.py)；评价：[ideal](cnh_delayed_query_eval.py)、[补充](cnh_delayed_query_supplement.py)。聚焦检查使用同目录两份test_cnh_*_model.py的unittest入口。

已有PLAN和完成收据由脚本保护，不直接覆盖重跑。复算指标可读取保留的scores.npz与冻结category；最终独立复算源码、四融合脚本及build_report.py均在上述载荷目录。特征重建需CUDA与继承输入：训练/held读取旧cnh-temporal-readout-20261004的inputs/train或inputs/fresh_evaluation中的histories、transforms、length；aligned/yaw分别调用cnh_delayed_query_dev.aligned_observations(0.)/(3.)。覆盖缓存调用prepare_features，token缓存调用cnh_boundary_token_dev.features，按center/extent及feature_cache_manifest.json原路径重建；重建不需要新的光子采样。特征重建和后续推理属于新执行，需要自己的预算，不能改写本轮收据。

失败记录：

| 文件 | 墙钟秒 |
| --- | --- |
| boundary_ablation\startup_failure.json | 0.5275822 |
| boundary_ablation\startup_failure_encoding.json | 5.1760734 |
