# ToF S集成 @1.25×：冻结任务成本HGB的新生成留出确认

2026-10-10 · CONFIRM_DEVELOPMENT · 任务 `CNH_S_ENSEMBLE_CONFIRM_DEV_20261010`。无训练；固定S955/956/957每高度HGB及47维实现，事先把1.25倍cal预算设为主工作点。

**保留原5格。** 四项预声明判据未全部通过，失败项：b, c。新hold原5格及时HEAD 398/512、BODY 315/512；S集成主点分别救/损 60/0、91/0，合计净增 151/1024。加权成本 223→320；far＋clear满额通知 133→174。

## 确认通道、冻结与访问顺序

按 `research/WORKFLOW.md`，固定方法、固定比较与明确主张的重新检验属于Confirm；本run采用正式注册的最低要求：预注册、固定判据/访问/失败规则、新family与物理键排除、独立复算。对象仍为合成Development，不访问保护FINAL；候选推荐不直接替换App或主线。

[已提交协议](CNH_S_ENSEMBLE_CONFIRM_PROTOCOL_20261010.md)在渲染前提交至Git `69e021496c519f9b897a5ea6df6f64b2f574a407`；控制器逐字节核验已提交协议后才生成光子与前向。[协议快照](../../../../artifacts.local/work/cnh-s-ensemble-confirm-dev-20261010/protocol_snapshot.md) SHA256 `c5728bb891c271a7c84f7228ec9e7cc9aa0609c80a92297266119ac3a22725ab`；[PLAN](../../../../artifacts.local/work/cnh-s-ensemble-confirm-dev-20261010/PLAN.json) SHA256 `7f209b8e99cd6ed3ef9d7ea916576d89e4fb4c4fe78f136ee4bbf94fb28e01a5`。场景与参数先写PLAN，失败不换源。

新cal只以负例联合加权成本选点；所有负例ties及端点完整枚举，不假设gap1成本单调，取满足成本≤原5格cal×1.25的最低阈值，不读取contact收益选点。

cal封存时间 `2026-10-10T14:09:54.110939+00:00`、SHA256 `50e82fb02e3e578d62f2a9b96a0983258ec4111ef3cf0c0d3831c802df87198f`；[sealed_calibration.json](../../../../artifacts.local/work/cnh-s-ensemble-confirm-dev-20261010/sealed_calibration.json)绑定cal分数、完整ties、诊断、冻结模型清单及执行清单。只有封存后才渲染/读hold，hold不再选点。旧次点精确tau `2.4142577648162846`直接迁移，仅描述漂移；不是四项主判据的候选。

## 数据与有限范围独立性

| split | physical scene | family | contact H/B scene | contact H/B事件 | pass每层scene | clear scene |
| --- | --- | --- | --- | --- | --- | --- |
| cal | 768 | 2 | 128/128 | 256/256 | 64/64/64/64 | 256 |
| hold | 1536 | 4 | 256/256 | 512/512 | 128/128/128/128 | 512 |

每physical scene K2。hold1536scene来自4新family/8背景实例，contact每高度512事件来自256scene；HEAD＋BODY1024事件来自512scene。pass四层各256clip/128scene，far合计512clip/256scene；clear1024clip/512scene。所有描述按physical scene聚类，K、帧、seed相关，不能当独立样本放大分母。

| split | 宽m | 厚m | rho | pass间隙cm | clear间隙cm |
| --- | --- | --- | --- | --- | --- |
| cal | 0.38917/0.84917 | 0.01417/0.09417 | 0.16117/0.64117 | 2.417/7.417/16.417/27.417 | 44.417/56.417 |
| hold | 0.46923/0.92923 | 0.02023/0.10423 | 0.24123/0.58123 | 3.423/8.423/18.423/32.423 | 48.423/60.423 |

cal family：`sconfirm_detached_side_pockets`、`sconfirm_reverse_step_columns`。

hold family：`sconfirm_offset_shelf_triplets`、`sconfirm_side_braced_notches`、`sconfirm_side_zigzag_slabs`、`sconfirm_upper_lower_shingled_blocks`。

参数完整列表，包括标牌edge、突出物depth/width/height、竖杆height与contact侵入深度，见PLAN `source_parameters`。横杆、竖杆、标牌边缘、突出物及暗薄层均覆盖；作者类别与几何仅用于生成、评价，不进入模型输入。

新cal/hold互相及与27份可访问已列元数据库存中16936历史物理键、56背景几何键及对应family零交集，含上轮新train/cal/hold；目标轴向尺寸/厚度/rho按10⁻¹⁰精度逐项排除，见[库存身份核对](../../../../artifacts.local/work/cnh-s-ensemble-confirm-dev-20261010/inventory.json)、[源核验](../../../../artifacts.local/work/cnh-s-ensemble-confirm-dev-20261010/prepare_receipt.json)。该排除证明限于可访问且具有完整AABB键/target_box的已列库存；保护来源只保留身份，真实/UE mesh等无同构键来源单列，不宣称全球历史背景排除。

## 字节冻结的特征、模型与工作点

S实际47维：ordinary平滑margin、raw趋势slope/residual共3维，current峰22项和有效性/invalidity22项。M3/local用于原5格强档锁，joint-parent用于E门控，均未额外增入S列；没有新均值/方差特征。逐字节复用上轮 `features_and_labels`、`predict_s` 和6颗HGB，集成评分按原实现float32三seed均值，随后混合归档可提升存储dtype。本run没有fit、微调、早停或特征改动。

| 冻结实现 | SHA256 |
| --- | --- |
| cnh_task_cost_retrain_20261010.py | b732c500f857741ab754a0ebbcc133f7b5a823360b7cba00acc310e2f4a05f2a |
| cnh_task_cost_train_20261010.py | 8b203d0204090c5958d2ebdc3b8fb16dc4a6206ba78863283969fea69aa971b3 |
| cnh_task_cost_metrics_20261010.py | b68764d994ef60ed8f08c541a7a3ee8135fd4190d677b61cad9c8ac1d341e54d |

| HGB权重 | SHA256 |
| --- | --- |
| S_2026100955_BODY.pickle | 824a93352542c6f8648e864385b9bae8c9c31194bd265e59ddabbe6a6d59a196 |
| S_2026100955_HEAD.pickle | b588beef916dbf052b945d602c7ace7720c163b0460e6aab4fbdcbdad7af82b3 |
| S_2026100956_BODY.pickle | dc0804a9693a4488055bc86fa21a25aa7e28bec44fb35104e1002595ce049b96 |
| S_2026100956_HEAD.pickle | 38f95121a15a56c07a1c64103c83feacc705a89428e5bf5efdfcc8ad22952071 |
| S_2026100957_BODY.pickle | 7b7812b893bb85518da9fd66f89c1ebac6518581ad8608b3bf0e814135724e60 |
| S_2026100957_HEAD.pickle | dc968fe2853eca555960b233e13997df635b471304067e73cdf5b0b9354077d2 |

模型沿用上轮训练产物，[冻结模型清单](../../../../artifacts.local/work/cnh-s-ensemble-confirm-dev-20261010/frozen_spec.json)与[执行输入清单](../../../../artifacts.local/work/cnh-s-ensemble-confirm-dev-20261010/execution_manifest.json)核对全部继承字节。本run不复制或覆盖旧权重。

原5格强档逐slot固定；所有新臂仅在非强slot追加轻提醒，强转轻为0。gap1逐query、同帧HEAD/BODY最高通知等级联合计数。near-pass最近表面到共向走廊±0.30m边缘间隙≤10cm：轻0.25/强1；far-pass和clear任何通知1。contact f3–13及时、f14–15晚、全窗无通知静默。原M3和原both955保持自己的冻结结局。E沿用 `max(mean ordinary-single margin, mean joint-parent margin)`，按同一1.25倍cal成本规则选点。

| 臂 | 阈值 | cal成本 | 原5格cal成本 | cal上限 | 候选阈值数 | 负例ties |
| --- | --- | --- | --- | --- | --- | --- |
| E | 0.03070745 | 171.25 | 137 | 171.25 | 366 | 364 |
| S955 | 2.452556 | 170.5 | 137 | 171.25 | 25905 | 25903 |
| S956 | 2.194922 | 171.25 | 137 | 171.25 | 25727 | 25725 |
| S957 | 2.282271 | 171.25 | 137 | 171.25 | 25768 | 25766 |
| Sensemble | 2.045032 | 170.75 | 137 | 171.25 | 26261 | 26259 |

[完整cal负例ties](../../../../artifacts.local/work/cnh-s-ensemble-confirm-dev-20261010/calibration_curve.csv)；迁移tau精确值来自上轮封存secondary记录并核对其SHA256，不使用四舍五入值2.41426重新实现。

## 四项事前判据的实测

| 项 | 固定门槛 | 实测 | 通过 |
| --- | --- | --- | --- |
| a | 总净增≥80/1024；HEAD/BODY各≥11/11（各/512） | 总151/1024；H/B 60/91 | True |
| b | far＋clear≤146.3（原5格133×1.10） | 174 | False |
| c | 成本≤292.6875（原5格223×1.25×1.05） | 320 | False |
| d | S每seed总净增≥40/1024 | {'S955': 145, 'S956': 142, 'S957': 140} | True |

总门槛40/512等比例为80/1024=7.8125%；每高度2%向上取整为11/512；每seed门槛为总门槛一半40/1024。联合通知允许非整数比值上限，实际计数仍为整数。

## hold各臂及时救/损与成本

每高度分母512事件/256scene；每格救/损均为及时事件。与原5格、原both955分别配对复算，不能用总净增替代个体救损。

| 臂 | H vs5救/损 | B vs5救/损 | H+B净增 | H vsboth救/损 | B vsboth救/损 | 加权成本 | far+clear满额 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed/m3 | 2/19 | 2/30 | -45 | 0/54 | 0/104 | 227 | 145 |
| fixed/old5 | 0/0 | 0/0 | 0 | 0/37 | 0/76 | 223 | 133 |
| both955 | 37/0 | 76/0 | 113 | 0/0 | 0/0 | 382 | 212 |
| E/main | 41/0 | 79/0 | 120 | 6/2 | 5/2 | 313.75 | 180 |
| S955/main | 59/0 | 86/0 | 145 | 22/0 | 13/3 | 303.25 | 161 |
| S956/main | 55/0 | 87/0 | 142 | 18/0 | 13/2 | 315.25 | 168 |
| S957/main | 48/0 | 92/0 | 140 | 13/2 | 18/2 | 310.5 | 166 |
| Sensemble/main | 60/0 | 91/0 | 151 | 23/0 | 17/2 | 320 | 174 |
| Sensemble/transfer | 47/0 | 86/0 | 133 | 10/0 | 13/3 | 285.5 | 149 |

### 暗薄层、标牌边缘

每格vs原5格救/损；括号内vs原both955救/损。暗薄与标牌边缘分组可重叠，不能相加。

| 臂 | HEAD_dark_thin /96事件/48scene | BODY_dark_thin /96事件/48scene | HEAD_sign_edge /128事件/64scene | BODY_sign_edge /128事件/64scene |
| --- | --- | --- | --- | --- |
| fixed/m3 | 0/2 (0/23) | 0/1 (0/27) | 0/10 (0/25) | 0/18 (0/45) |
| fixed/old5 | 0/0 (0/21) | 0/0 (0/26) | 0/0 (0/15) | 0/0 (0/27) |
| both955 | 21/0 (0/0) | 26/0 (0/0) | 15/0 (0/0) | 27/0 (0/0) |
| E/main | 25/0 (4/0) | 27/0 (1/0) | 15/0 (0/0) | 26/0 (0/1) |
| S955/main | 28/0 (7/0) | 26/0 (2/2) | 15/0 (0/0) | 27/0 (1/1) |
| S956/main | 27/0 (6/0) | 26/0 (1/1) | 15/0 (0/0) | 27/0 (1/1) |
| S957/main | 25/0 (5/1) | 27/0 (1/0) | 15/0 (0/0) | 27/0 (2/2) |
| Sensemble/main | 29/0 (8/0) | 26/0 (1/1) | 15/0 (0/0) | 28/0 (2/1) |
| Sensemble/transfer | 24/0 (3/0) | 25/0 (1/2) | 15/0 (0/0) | 26/0 (1/2) |

### 及时/晚/静默与physical scene

| 臂 | 高度 | 及时/晚/静默 /512 | 有及时scene | 有救scene | 有损scene |
| --- | --- | --- | --- | --- | --- |
| fixed/m3 | HEAD | 381/59/72 | 205/256 | 2 | 18 |
| fixed/m3 | BODY | 287/15/210 | 164/256 | 2 | 25 |
| fixed/old5 | HEAD | 398/45/69 | 207/256 | 0 | 0 |
| fixed/old5 | BODY | 315/22/175 | 174/256 | 0 | 0 |
| both955 | HEAD | 435/34/43 | 225/256 | 32 | 0 |
| both955 | BODY | 391/9/112 | 205/256 | 55 | 0 |
| E/main | HEAD | 439/37/36 | 227/256 | 35 | 0 |
| E/main | BODY | 394/7/111 | 206/256 | 57 | 0 |
| S955/main | HEAD | 457/13/42 | 235/256 | 43 | 0 |
| S955/main | BODY | 401/8/103 | 211/256 | 62 | 0 |
| S956/main | HEAD | 453/13/46 | 234/256 | 42 | 0 |
| S956/main | BODY | 402/7/103 | 211/256 | 63 | 0 |
| S957/main | HEAD | 446/17/49 | 230/256 | 38 | 0 |
| S957/main | BODY | 407/8/97 | 216/256 | 68 | 0 |
| Sensemble/main | HEAD | 458/13/41 | 235/256 | 43 | 0 |
| Sensemble/main | BODY | 406/7/99 | 214/256 | 66 | 0 |
| Sensemble/transfer | HEAD | 445/16/51 | 230/256 | 38 | 0 |
| Sensemble/transfer | BODY | 401/7/104 | 211/256 | 62 | 0 |

## 通知原始计数、clip率与scene率

通知为gap1联合次数，每格轻/强。逐slot强档锁定不等于发出强通知次数恒定：轻档可能改变gap1状态，因此报告实际通知。

| 臂 | 0-5cm | 5-10cm | 10-20cm | 20-35cm | clear |
| --- | --- | --- | --- | --- | --- |
| fixed/m3 | 0/60 | 0/22 | 0/20 | 0/23 | 0/102 |
| fixed/old5 | 0/71 | 0/19 | 0/17 | 0/19 | 0/97 |
| both955 | 124/103 | 60/21 | 15/18 | 8/19 | 55/97 |
| E/main | 136/63 | 71/19 | 10/17 | 1/19 | 36/97 |
| S955/main | 162/66 | 67/19 | 6/17 | 2/19 | 20/97 |
| S956/main | 168/66 | 81/19 | 8/17 | 6/19 | 21/97 |
| S957/main | 158/67 | 76/19 | 6/17 | 2/19 | 25/97 |
| Sensemble/main | 163/66 | 81/19 | 6/17 | 4/19 | 31/97 |
| Sensemble/transfer | 151/66 | 55/19 | 3/17 | 1/19 | 12/97 |

### 有任意通知clip率

far-pass“误报率”是作者pass clip有任意通知的比例，限定这些固定模拟几何；K2不作为独立重复证明。

| 臂 | 0-5cm | 5-10cm | 10-20cm | 20-35cm | clear | far合计 |
| --- | --- | --- | --- | --- | --- | --- |
| fixed/m3 | 55/256 (21.48%) | 21/256 (8.20%) | 20/256 (7.81%) | 22/256 (8.59%) | 100/1024 (9.77%) | 42/512 (8.20%) |
| fixed/old5 | 62/256 (24.22%) | 19/256 (7.42%) | 17/256 (6.64%) | 18/256 (7.03%) | 94/1024 (9.18%) | 35/512 (6.84%) |
| both955 | 145/256 (56.64%) | 64/256 (25.00%) | 30/256 (11.72%) | 26/256 (10.16%) | 146/1024 (14.26%) | 56/512 (10.94%) |
| E/main | 152/256 (59.38%) | 71/256 (27.73%) | 27/256 (10.55%) | 19/256 (7.42%) | 124/1024 (12.11%) | 46/512 (8.98%) |
| S955/main | 163/256 (63.67%) | 74/256 (28.91%) | 23/256 (8.98%) | 20/256 (7.81%) | 110/1024 (10.74%) | 43/512 (8.40%) |
| S956/main | 168/256 (65.62%) | 83/256 (32.42%) | 25/256 (9.77%) | 22/256 (8.59%) | 111/1024 (10.84%) | 47/512 (9.18%) |
| S957/main | 163/256 (63.67%) | 82/256 (32.03%) | 21/256 (8.20%) | 20/256 (7.81%) | 115/1024 (11.23%) | 41/512 (8.01%) |
| Sensemble/main | 169/256 (66.02%) | 86/256 (33.59%) | 23/256 (8.98%) | 21/256 (8.20%) | 119/1024 (11.62%) | 44/512 (8.59%) |
| Sensemble/transfer | 156/256 (60.94%) | 63/256 (24.61%) | 20/256 (7.81%) | 19/256 (7.42%) | 103/1024 (10.06%) | 39/512 (7.62%) |

### 有任意通知physical-scene率

| 臂 | 0-5cm | 5-10cm | 10-20cm | 20-35cm | clear | far合计 |
| --- | --- | --- | --- | --- | --- | --- |
| fixed/m3 | 39/128 (30.47%) | 19/128 (14.84%) | 19/128 (14.84%) | 21/128 (16.41%) | 94/512 (18.36%) | 40/256 (15.62%) |
| fixed/old5 | 42/128 (32.81%) | 17/128 (13.28%) | 16/128 (12.50%) | 18/128 (14.06%) | 87/512 (16.99%) | 34/256 (13.28%) |
| both955 | 81/128 (63.28%) | 52/128 (40.62%) | 29/128 (22.66%) | 26/128 (20.31%) | 134/512 (26.17%) | 55/256 (21.48%) |
| E/main | 81/128 (63.28%) | 53/128 (41.41%) | 25/128 (19.53%) | 19/128 (14.84%) | 116/512 (22.66%) | 44/256 (17.19%) |
| S955/main | 90/128 (70.31%) | 52/128 (40.62%) | 21/128 (16.41%) | 20/128 (15.62%) | 102/512 (19.92%) | 41/256 (16.02%) |
| S956/main | 94/128 (73.44%) | 58/128 (45.31%) | 24/128 (18.75%) | 22/128 (17.19%) | 103/512 (20.12%) | 46/256 (17.97%) |
| S957/main | 88/128 (68.75%) | 56/128 (43.75%) | 20/128 (15.62%) | 20/128 (15.62%) | 107/512 (20.90%) | 40/256 (15.62%) |
| Sensemble/main | 93/128 (72.66%) | 59/128 (46.09%) | 22/128 (17.19%) | 21/128 (16.41%) | 111/512 (21.68%) | 43/256 (16.80%) |
| Sensemble/transfer | 85/128 (66.41%) | 45/128 (35.16%) | 19/128 (14.84%) | 19/128 (14.84%) | 95/512 (18.55%) | 38/256 (14.84%) |

### 成本固定权重敏感性

0、0.5只描述同一封存阈值，不重新选点。

| 臂 | w=0 | w=.25主 | w=.5 |
| --- | --- | --- | --- |
| fixed/m3 | 227 | 227 | 227 |
| fixed/old5 | 223 | 223 | 223 |
| both955 | 336 | 382 | 428 |
| E/main | 262 | 313.75 | 365.5 |
| S955/main | 246 | 303.25 | 360.5 |
| S956/main | 253 | 315.25 | 377.5 |
| S957/main | 252 | 310.5 | 369 |
| Sensemble/main | 259 | 320 | 381 |
| Sensemble/transfer | 234 | 285.5 | 337 |

## 结论与漂移

主点收益净增14.75%，HEAD/BODY分别11.72%/17.77%；三单seed都超过收益门槛。成本侧未复现：far＋clear从133到174，增30.83%；总加权成本增43.50%，超过允许的31.25%。新cal的1.25倍成本可行，不保证新hold仍满足同一倍率及5%漂移容差。

总成本增量97中，near-pass加权增量56，far-pass满额通知增10、clear增31。新增成本并未全部集中于用户认可的near轻提醒，远pass/clear约束本身也失败。

精确迁移tau的及时净增133/1024、成本285.5、far＋clear149；仅作新域漂移描述，新cal主点和迁移点结果都完整保留。

**保留原5格。** 本次固定确认未通过项：b, c；不以迁移阈值或对照臂表现替换预声明S主点判据，不因hold结果另选阈值。S任务标签轻档的收益及成本证据保留；后续如改变机制或口径需明确为新授权run。

## 独立核验、预算与交付

逐事件 [event_ledger.csv](../../../../artifacts.local/work/cnh-s-ensemble-confirm-dev-20261010/event_ledger.csv)、逐通知 [notification_ledger.csv](../../../../artifacts.local/work/cnh-s-ensemble-confirm-dev-20261010/notification_ledger.csv)包含physical key、split/K、作者分组、等级/权重、paired救损及原始结局；完整分组/scene计数见[metrics.json](../../../../artifacts.local/work/cnh-s-ensemble-confirm-dev-20261010/metrics.json)。

独立核验：[independent_audit.json](../../../../artifacts.local/work/cnh-s-ensemble-confirm-dev-20261010/independent_audit.json)，状态 `PASS`，检查细项以receipt为准。

独立实现复算6权重及3实现hash、27库存/16936历史物理键、全部cal负例ties与最低可行切点、slot强档锁、gap1联合成本、82944逐事件及22594逐通知、分组/scene聚类统计和四项判据。独立核验本身无渲染、前向、训练或保护访问；command-wall 22.906s。

保守预算扣账：渲染＋冻结前向GPU 200/1200s；评价/审计/整合command-wall 800/1200s。这些为向上取整扣账，包含CPU段与启动/整合余量，非核函数时间。GPU数据阶段实际整段receipt cal71.594s＋hold98.703s=170.297s；本run训练为0。无科学执行失败或换源，失败lineage若有非科学审计/整合修正以对应receipt保留。

任务渲染/前向Python与GPU进程已释放，共享hardware-bringup任务保留。登记用close-experiment关闭为completed：terminal_status=GATE_NOT_MET，evidence_verdict=mixed（收益/seed复现，成本未复现）。本run不改变主线/受管复用inheritance；S仅保留未晋升研究组件，不能将旧次点或本迁移点恢复为确认通过。协议快照、正式登记、报告、实现/独立审计脚本及CURRENT/RUNS/CURRENT_DECISION正常直推master；commit/远端一致性见[delivery_receipt.json](../../../../artifacts.local/work/cnh-s-ensemble-confirm-dev-20261010/delivery_receipt.json)，预算扣账见[budget_delivery_receipt.json](../../../../artifacts.local/work/cnh-s-ensemble-confirm-dev-20261010/budget_delivery_receipt.json)。

本run为有限作者AABB合成Development确认。物理键、参数、family无交集不代表真实世界分布独立，slot/帧/seed不作iid样本；新hold现已消费。没有训练、保护480/test访问、已消费数据选点，不作实机安全、硬件极限或完整RGB/CNH同步系统证明。旧run和各自停止规则保留。
