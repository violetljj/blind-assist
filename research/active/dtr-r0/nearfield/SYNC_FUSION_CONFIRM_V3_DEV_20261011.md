# 融合确认 v3：FAIL

主中带确认未通过。
协议 `ec252d68` 在新数据前提交并推送。新 cal6/eval12、576帧，30个旧消费visit一次训练；六臂等权HGB D′、逻辑回归B′、native-only审计(b)均删除显式query位置与带别输入，16传感器特征及最低证据OR门控。分带只用于阈值。

## 主判据：半循环中带

D′ W1696/2077，最佳单源RGB-only W1609/2077，净+87，救213/损126；visit聚类95%净增区间 -0.308–9.155pp。中带FREE 4/689=0.581%，上限7.5%。判定 **FAIL**。

| 模型 | W/POS | 最佳单源净增 | 救/损 | 净增95%区间 | F/FREE | 同口径判定 |
|---|---:|---:|---:|---:|---:|---|
| D′ | 1696/2077 | +87 | 213/126 | -0.308–9.155pp | 4/689 (0.581%) | FAIL |
| B′ | 1695/2077 | +86 | 210/124 | -0.519–9.557pp | 8/689 (1.161%) | FAIL |
| 审计(b) | 1518/2077 | -91 | 219/310 | -11.354–2.535pp | 0/689 (0.000%) | FAIL |

主D′各子条件：{"calibrated_and_denominators": true, "positive_difference": true, "interval_lower_positive": false, "FREE_tolerance_pass": true}。

## 次判据：近带退化臂

| 臂 | 模型 | W/POS | RGB W/POS | 净增 | F/FREE | 判定 |
|---|---|---:|---:|---:|---:|---|
| faro_rho030_ambient1 | D′ | 260/273 | 262/273 | -2 | 15/105 (14.286%) | FAIL |
| faro_rho030_ambient1 | B′ | 260/273 | 262/273 | -2 | 15/105 (14.286%) | FAIL |
| faro_rho030_ambient1 | 审计(b) | 260/273 | 262/273 | -2 | 15/105 (14.286%) | FAIL |
| faro_rho030_ambient10 | D′ | 1188/1643 | 1246/1643 | -58 | 121/1702 (7.109%) | FAIL |
| faro_rho030_ambient10 | B′ | 1188/1643 | 1246/1643 | -58 | 121/1702 (7.109%) | FAIL |
| faro_rho030_ambient10 | 审计(b) | 1188/1643 | 1246/1643 | -58 | 121/1702 (7.109%) | FAIL |

D′次判据整体：FAIL。各臂均要求相对RGB净增≥0及FREE≤4%；观察到不减不构成统计非劣证明。主/次判据独立，近/远带数字完整报告，远带只报告不作主张。
faro_rho030_ambient1：{"calibrated_and_denominators": true, "nonnegative_net": false, "FREE_tolerance_pass": false}。
faro_rho030_ambient10：{"calibrated_and_denominators": true, "nonnegative_net": false, "FREE_tolerance_pass": false}。

## 四类证据：半循环中带

| 模型 | 证据类 | POS | W | F | U | 救/损 | 净增vs最佳单源 |
|---|---|---:|---:|---:|---:|---:|---:|
| D′ | both | 845 | 829 | 0 | 35 | 0/16 | -16 |
| D′ | tof_only | 331 | 288 | 0 | 14 | 213/4 | +209 |
| D′ | rgb_only | 605 | 579 | 4 | 425 | 0/26 | -26 |
| D′ | neither | 296 | 0 | 0 | 0 | 0/80 | -80 |
| B′ | both | 845 | 826 | 0 | 35 | 0/19 | -19 |
| B′ | tof_only | 331 | 283 | 0 | 20 | 210/6 | +204 |
| B′ | rgb_only | 605 | 586 | 8 | 502 | 0/19 | -19 |
| B′ | neither | 296 | 0 | 0 | 0 | 0/80 | -80 |
| 审计(b) | both | 845 | 826 | 0 | 35 | 0/19 | -19 |
| 审计(b) | tof_only | 331 | 295 | 0 | 20 | 219/3 | +216 |
| 审计(b) | rgb_only | 605 | 397 | 0 | 397 | 0/208 | -208 |
| 审计(b) | neither | 296 | 0 | 0 | 0 | 0/80 | -80 |

全部6臂/3带中，D′/B′/(b)无最低证据支持总数为 **0**。最低证据：未裁剪K2原ToF margin≥0或当前带RGB raw margin≥0；任一K缺失时ToF raw=-inf。四类净和复现总净，属于条件分解，不是因果归因；去先验指删除显式位置/带别特征，保留传感器区间相对几何量。

## 新数据、校准与封存

按官方metadata物理顺序接受18 visit，历史排除143 visit；候选终态 {"SKIP_REFERENCE_GATE": 24, "ACCEPTED": 18}。每visit两段16帧5Hz，visit级train/cal/eval隔离，门槛近POS≥16、中远严格FREE≥32；窗口先于标签固定，不合格visit不换capture/窗口。
- eval native_perturbed：384帧/12贡献visit，10368 query；ToF raw缺测3837/10368。
- eval faro_rho015_ambient1：384帧/12贡献visit，10368 query；ToF raw缺测5146/10368。
- eval faro_rho060_ambient1：384帧/12贡献visit，10368 query；ToF raw缺测3258/10368。
- eval faro_rho030_ambient3：384帧/12贡献visit，10368 query；ToF raw缺测4796/10368。
- eval faro_rho030_ambient1：43帧/5贡献visit，1161 query；ToF raw缺测7/1161。
- eval faro_rho030_ambient10：384帧/12贡献visit，10368 query；ToF raw缺测5779/10368。

新cal六臂原始行混合，每方法/带一个阈值，近2%/中5%/远10%；21切点、54566ties独立枚举复算。最大W、再最小F、再最高阈值；零POS/FREE全拒绝。9权重、16列、cal cache/全部ties/代码/阈值先封存，再一次OPEN；eval之后只读冻结逐query cache进行统计与独立审计，没有再读eval reference或修改模型/阈值。

## 全部评价数字

W=预测query阳性且POS，F=阳性且严格sampled FREE，U=阳性且UNKNOWN；UNKNOWN不当负例。
| 臂 | 带 | 方法 | W/POS | F/FREE | U/UNKNOWN |
|---|---|---|---:|---:|---:|
| native_perturbed | 0.3-0.8m | D′ | 1245/1643 | 48/1702 (2.820%) | 53/111 |
| native_perturbed | 0.3-0.8m | B′ | 1244/1643 | 48/1702 (2.820%) | 53/111 |
| native_perturbed | 0.3-0.8m | 审计(b) | 1245/1643 | 48/1702 (2.820%) | 53/111 |
| native_perturbed | 0.3-0.8m | ToF-only | 906/1643 | 52/1702 (3.055%) | 51/111 |
| native_perturbed | 0.3-0.8m | RGB-only | 1246/1643 | 62/1702 (3.643%) | 66/111 |
| native_perturbed | 0.3-0.8m | OR | 1279/1643 | 53/1702 (3.114%) | 57/111 |
| native_perturbed | 0.3-0.8m | AND | 938/1643 | 44/1702 (2.585%) | 46/111 |
| native_perturbed | 0.8-1.5m | D′ | 1696/2077 | 4/689 (0.581%) | 474/690 |
| native_perturbed | 0.8-1.5m | B′ | 1695/2077 | 8/689 (1.161%) | 557/690 |
| native_perturbed | 0.8-1.5m | 审计(b) | 1518/2077 | 0/689 (0.000%) | 452/690 |
| native_perturbed | 0.8-1.5m | ToF-only | 854/2077 | 0/689 (0.000%) | 2/690 |
| native_perturbed | 0.8-1.5m | RGB-only | 1609/2077 | 67/689 (9.724%) | 611/690 |
| native_perturbed | 0.8-1.5m | OR | 1461/2077 | 11/689 (1.597%) | 431/690 |
| native_perturbed | 0.8-1.5m | AND | 1197/2077 | 25/689 (3.628%) | 449/690 |
| native_perturbed | 1.5-3m | D′ | 1836/1988 | 107/341 (31.378%) | 422/1127 |
| native_perturbed | 1.5-3m | B′ | 1832/1988 | 65/341 (19.062%) | 406/1127 |
| native_perturbed | 1.5-3m | 审计(b) | 1836/1988 | 83/341 (24.340%) | 419/1127 |
| native_perturbed | 1.5-3m | ToF-only | 881/1988 | 8/341 (2.346%) | 28/1127 |
| native_perturbed | 1.5-3m | RGB-only | 1923/1988 | 174/341 (51.026%) | 806/1127 |
| native_perturbed | 1.5-3m | OR | 1789/1988 | 90/341 (26.393%) | 340/1127 |
| native_perturbed | 1.5-3m | AND | 1313/1988 | 23/341 (6.745%) | 526/1127 |
| faro_rho015_ambient1 | 0.3-0.8m | D′ | 1193/1643 | 123/1702 (7.227%) | 52/111 |
| faro_rho015_ambient1 | 0.3-0.8m | B′ | 1193/1643 | 122/1702 (7.168%) | 52/111 |
| faro_rho015_ambient1 | 0.3-0.8m | 审计(b) | 1193/1643 | 122/1702 (7.168%) | 52/111 |
| faro_rho015_ambient1 | 0.3-0.8m | ToF-only | 690/1643 | 153/1702 (8.989%) | 22/111 |
| faro_rho015_ambient1 | 0.3-0.8m | RGB-only | 1246/1643 | 62/1702 (3.643%) | 66/111 |
| faro_rho015_ambient1 | 0.3-0.8m | OR | 1224/1643 | 129/1702 (7.579%) | 56/111 |
| faro_rho015_ambient1 | 0.3-0.8m | AND | 765/1643 | 73/1702 (4.289%) | 22/111 |
| faro_rho015_ambient1 | 0.8-1.5m | D′ | 1590/2077 | 5/689 (0.726%) | 489/690 |
| faro_rho015_ambient1 | 0.8-1.5m | B′ | 1610/2077 | 8/689 (1.161%) | 564/690 |
| faro_rho015_ambient1 | 0.8-1.5m | 审计(b) | 1408/2077 | 1/689 (0.145%) | 421/690 |
| faro_rho015_ambient1 | 0.8-1.5m | ToF-only | 557/2077 | 0/689 (0.000%) | 34/690 |
| faro_rho015_ambient1 | 0.8-1.5m | RGB-only | 1609/2077 | 67/689 (9.724%) | 611/690 |
| faro_rho015_ambient1 | 0.8-1.5m | OR | 1307/2077 | 11/689 (1.597%) | 448/690 |
| faro_rho015_ambient1 | 0.8-1.5m | AND | 1012/2077 | 14/689 (2.032%) | 344/690 |
| faro_rho015_ambient1 | 1.5-3m | D′ | 1794/1988 | 103/341 (30.205%) | 419/1127 |
| faro_rho015_ambient1 | 1.5-3m | B′ | 1792/1988 | 64/341 (18.768%) | 400/1127 |
| faro_rho015_ambient1 | 1.5-3m | 审计(b) | 1794/1988 | 81/341 (23.754%) | 415/1127 |
| faro_rho015_ambient1 | 1.5-3m | ToF-only | 373/1988 | 5/341 (1.466%) | 32/1127 |
| faro_rho015_ambient1 | 1.5-3m | RGB-only | 1923/1988 | 174/341 (51.026%) | 806/1127 |
| faro_rho015_ambient1 | 1.5-3m | OR | 1737/1988 | 89/341 (26.100%) | 342/1127 |
| faro_rho015_ambient1 | 1.5-3m | AND | 899/1988 | 17/341 (4.985%) | 404/1127 |
| faro_rho060_ambient1 | 0.3-0.8m | D′ | 1199/1643 | 124/1702 (7.286%) | 53/111 |
| faro_rho060_ambient1 | 0.3-0.8m | B′ | 1199/1643 | 123/1702 (7.227%) | 53/111 |
| faro_rho060_ambient1 | 0.3-0.8m | 审计(b) | 1199/1643 | 124/1702 (7.286%) | 53/111 |
| faro_rho060_ambient1 | 0.3-0.8m | ToF-only | 708/1643 | 156/1702 (9.166%) | 25/111 |
| faro_rho060_ambient1 | 0.3-0.8m | RGB-only | 1246/1643 | 62/1702 (3.643%) | 66/111 |
| faro_rho060_ambient1 | 0.3-0.8m | OR | 1229/1643 | 131/1702 (7.697%) | 57/111 |
| faro_rho060_ambient1 | 0.3-0.8m | AND | 788/1643 | 77/1702 (4.524%) | 23/111 |
| faro_rho060_ambient1 | 0.8-1.5m | D′ | 1637/2077 | 9/689 (1.306%) | 487/690 |
| faro_rho060_ambient1 | 0.8-1.5m | B′ | 1629/2077 | 11/689 (1.597%) | 562/690 |
| faro_rho060_ambient1 | 0.8-1.5m | 审计(b) | 1477/2077 | 11/689 (1.597%) | 436/690 |
| faro_rho060_ambient1 | 0.8-1.5m | ToF-only | 710/2077 | 4/689 (0.581%) | 37/690 |
| faro_rho060_ambient1 | 0.8-1.5m | RGB-only | 1609/2077 | 67/689 (9.724%) | 611/690 |
| faro_rho060_ambient1 | 0.8-1.5m | OR | 1386/2077 | 15/689 (2.177%) | 447/690 |
| faro_rho060_ambient1 | 0.8-1.5m | AND | 1184/2077 | 33/689 (4.790%) | 346/690 |
| faro_rho060_ambient1 | 1.5-3m | D′ | 1878/1988 | 110/341 (32.258%) | 446/1127 |
| faro_rho060_ambient1 | 1.5-3m | B′ | 1872/1988 | 68/341 (19.941%) | 425/1127 |
| faro_rho060_ambient1 | 1.5-3m | 审计(b) | 1878/1988 | 100/341 (29.326%) | 441/1127 |
| faro_rho060_ambient1 | 1.5-3m | ToF-only | 1130/1988 | 15/341 (4.399%) | 102/1127 |
| faro_rho060_ambient1 | 1.5-3m | RGB-only | 1923/1988 | 174/341 (51.026%) | 806/1127 |
| faro_rho060_ambient1 | 1.5-3m | OR | 1840/1988 | 98/341 (28.739%) | 377/1127 |
| faro_rho060_ambient1 | 1.5-3m | AND | 1386/1988 | 67/341 (19.648%) | 454/1127 |
| faro_rho030_ambient3 | 0.3-0.8m | D′ | 1196/1643 | 121/1702 (7.109%) | 52/111 |
| faro_rho030_ambient3 | 0.3-0.8m | B′ | 1196/1643 | 122/1702 (7.168%) | 52/111 |
| faro_rho030_ambient3 | 0.3-0.8m | 审计(b) | 1196/1643 | 122/1702 (7.168%) | 52/111 |
| faro_rho030_ambient3 | 0.3-0.8m | ToF-only | 708/1643 | 155/1702 (9.107%) | 24/111 |
| faro_rho030_ambient3 | 0.3-0.8m | RGB-only | 1246/1643 | 62/1702 (3.643%) | 66/111 |
| faro_rho030_ambient3 | 0.3-0.8m | OR | 1228/1643 | 128/1702 (7.521%) | 56/111 |
| faro_rho030_ambient3 | 0.3-0.8m | AND | 776/1643 | 75/1702 (4.407%) | 23/111 |
| faro_rho030_ambient3 | 0.8-1.5m | D′ | 1611/2077 | 6/689 (0.871%) | 489/690 |
| faro_rho030_ambient3 | 0.8-1.5m | B′ | 1629/2077 | 8/689 (1.161%) | 568/690 |
| faro_rho030_ambient3 | 0.8-1.5m | 审计(b) | 1430/2077 | 1/689 (0.145%) | 424/690 |
| faro_rho030_ambient3 | 0.8-1.5m | ToF-only | 633/2077 | 0/689 (0.000%) | 35/690 |
| faro_rho030_ambient3 | 0.8-1.5m | RGB-only | 1609/2077 | 67/689 (9.724%) | 611/690 |
| faro_rho030_ambient3 | 0.8-1.5m | OR | 1342/2077 | 11/689 (1.597%) | 449/690 |
| faro_rho030_ambient3 | 0.8-1.5m | AND | 1068/2077 | 18/689 (2.612%) | 346/690 |
| faro_rho030_ambient3 | 1.5-3m | D′ | 1809/1988 | 102/341 (29.912%) | 424/1127 |
| faro_rho030_ambient3 | 1.5-3m | B′ | 1808/1988 | 62/341 (18.182%) | 405/1127 |
| faro_rho030_ambient3 | 1.5-3m | 审计(b) | 1809/1988 | 81/341 (23.754%) | 421/1127 |
| faro_rho030_ambient3 | 1.5-3m | ToF-only | 532/1988 | 3/341 (0.880%) | 46/1127 |
| faro_rho030_ambient3 | 1.5-3m | RGB-only | 1923/1988 | 174/341 (51.026%) | 806/1127 |
| faro_rho030_ambient3 | 1.5-3m | OR | 1755/1988 | 88/341 (25.806%) | 352/1127 |
| faro_rho030_ambient3 | 1.5-3m | AND | 1030/1988 | 22/341 (6.452%) | 410/1127 |
| faro_rho030_ambient1 | 0.3-0.8m | D′ | 260/273 | 15/105 (14.286%) | 2/9 |
| faro_rho030_ambient1 | 0.3-0.8m | B′ | 260/273 | 15/105 (14.286%) | 2/9 |
| faro_rho030_ambient1 | 0.3-0.8m | 审计(b) | 260/273 | 15/105 (14.286%) | 2/9 |
| faro_rho030_ambient1 | 0.3-0.8m | ToF-only | 211/273 | 30/105 (28.571%) | 2/9 |
| faro_rho030_ambient1 | 0.3-0.8m | RGB-only | 262/273 | 6/105 (5.714%) | 6/9 |
| faro_rho030_ambient1 | 0.3-0.8m | OR | 262/273 | 16/105 (15.238%) | 2/9 |
| faro_rho030_ambient1 | 0.3-0.8m | AND | 212/273 | 20/105 (19.048%) | 3/9 |
| faro_rho030_ambient1 | 0.8-1.5m | D′ | 156/159 | 0/7 (0.000%) | 211/221 |
| faro_rho030_ambient1 | 0.8-1.5m | B′ | 156/159 | 0/7 (0.000%) | 221/221 |
| faro_rho030_ambient1 | 0.8-1.5m | 审计(b) | 155/159 | 0/7 (0.000%) | 221/221 |
| faro_rho030_ambient1 | 0.8-1.5m | ToF-only | 105/159 | 0/7 (0.000%) | 1/221 |
| faro_rho030_ambient1 | 0.8-1.5m | RGB-only | 135/159 | 1/7 (14.286%) | 221/221 |
| faro_rho030_ambient1 | 0.8-1.5m | OR | 129/159 | 0/7 (0.000%) | 147/221 |
| faro_rho030_ambient1 | 0.8-1.5m | AND | 147/159 | 1/7 (14.286%) | 160/221 |
| faro_rho030_ambient1 | 1.5-3m | D′ | 48/48 | 0/0 (N/E) | 76/339 |
| faro_rho030_ambient1 | 1.5-3m | B′ | 48/48 | 0/0 (N/E) | 75/339 |
| faro_rho030_ambient1 | 1.5-3m | 审计(b) | 48/48 | 0/0 (N/E) | 76/339 |
| faro_rho030_ambient1 | 1.5-3m | ToF-only | 43/48 | 0/0 (N/E) | 1/339 |
| faro_rho030_ambient1 | 1.5-3m | RGB-only | 46/48 | 0/0 (N/E) | 200/339 |
| faro_rho030_ambient1 | 1.5-3m | OR | 48/48 | 0/0 (N/E) | 47/339 |
| faro_rho030_ambient1 | 1.5-3m | AND | 48/48 | 0/0 (N/E) | 142/339 |
| faro_rho030_ambient10 | 0.3-0.8m | D′ | 1188/1643 | 121/1702 (7.109%) | 52/111 |
| faro_rho030_ambient10 | 0.3-0.8m | B′ | 1188/1643 | 121/1702 (7.109%) | 52/111 |
| faro_rho030_ambient10 | 0.3-0.8m | 审计(b) | 1188/1643 | 121/1702 (7.109%) | 52/111 |
| faro_rho030_ambient10 | 0.3-0.8m | ToF-only | 682/1643 | 159/1702 (9.342%) | 24/111 |
| faro_rho030_ambient10 | 0.3-0.8m | RGB-only | 1246/1643 | 62/1702 (3.643%) | 66/111 |
| faro_rho030_ambient10 | 0.3-0.8m | OR | 1221/1643 | 130/1702 (7.638%) | 56/111 |
| faro_rho030_ambient10 | 0.3-0.8m | AND | 733/1643 | 71/1702 (4.172%) | 22/111 |
| faro_rho030_ambient10 | 0.8-1.5m | D′ | 1546/2077 | 5/689 (0.726%) | 486/690 |
| faro_rho030_ambient10 | 0.8-1.5m | B′ | 1565/2077 | 9/689 (1.306%) | 560/690 |
| faro_rho030_ambient10 | 0.8-1.5m | 审计(b) | 1352/2077 | 1/689 (0.145%) | 418/690 |
| faro_rho030_ambient10 | 0.8-1.5m | ToF-only | 425/2077 | 1/689 (0.145%) | 24/690 |
| faro_rho030_ambient10 | 0.8-1.5m | RGB-only | 1609/2077 | 67/689 (9.724%) | 611/690 |
| faro_rho030_ambient10 | 0.8-1.5m | OR | 1243/2077 | 12/689 (1.742%) | 443/690 |
| faro_rho030_ambient10 | 0.8-1.5m | AND | 884/2077 | 9/689 (1.306%) | 339/690 |
| faro_rho030_ambient10 | 1.5-3m | D′ | 1780/1988 | 101/341 (29.619%) | 417/1127 |
| faro_rho030_ambient10 | 1.5-3m | B′ | 1778/1988 | 61/341 (17.889%) | 398/1127 |
| faro_rho030_ambient10 | 1.5-3m | 审计(b) | 1780/1988 | 76/341 (22.287%) | 413/1127 |
| faro_rho030_ambient10 | 1.5-3m | ToF-only | 127/1988 | 2/341 (0.587%) | 19/1127 |
| faro_rho030_ambient10 | 1.5-3m | RGB-only | 1923/1988 | 174/341 (51.026%) | 806/1127 |
| faro_rho030_ambient10 | 1.5-3m | OR | 1720/1988 | 87/341 (25.513%) | 340/1127 |
| faro_rho030_ambient10 | 1.5-3m | AND | 680/1988 | 12/341 (3.519%) | 398/1127 |

## 独立核验、资源与交付

协议/训练/cal封存/冻结eval审计均PASS。独立核验53001逐query、371007决策、126指标、108配对、504证据格、432四类配对分解、2000次/12visit bootstrap；cal两固定帧压力数组与16列读出样本也已复算。统计不重新训练，区间反映固定模型/阈值的visit抽样变异。
下载 3,889,213,192 /4,000,000,000B；GPU 211.577/600s；测得阶段CPU 1974.665s，含杂项保守上界 2209.665/2400 command-wall s。内部data规划额度重分配记录保留，总硬上限未改。来源先完整目标，再官方注册siblings固定前缀；最终18目标+23注册capture，下一组件超剩余字节预算而停止，未提供整visit完整几何保证。原source墙限、render partial及机械continuation记录保留。模型/GPU/网络进程已释放。

真实RGB＋半合成ToF的Development确认；半循环不是独立ToF真值，FARO只过门帧，标签门槛富集、名义共位/45°与轨迹缺测限制保留。不是实机、整身体清空或导航安全证明；不读取保护480/test，不升级App/M3/5格默认读出。

完整payload：`F:\ba-data\blindassist-artifacts-20260805\work\sync-fusion-confirm-v3-dev-20261011`。
- 新manifest：`dataset_manifest_v3_complete.json`（仓库薄manifest `SYNC_FUSION_CONFIRM_V3_MANIFEST_DEV_20261011.json`）。
- 权重hash：`train/train_seal.json`；阈值与全ties：`cal/cal_seal.json`、`cal/ties/`。
- eval逐query：`eval/per_query.csv`、`eval/cache.npz`；bootstrap/全部指标：`eval/summary.json`。
- 证据拆分：`eval/independent_evidence_decomposition.json`；四阶段独立核验：`independent_*_audit.json`。
- 仓库收据：`SYNC_FUSION_CONFIRM_V3_RECEIPTS_DEV_20261011.json`，逐文件SHA256、9权重、21切点与资源。
