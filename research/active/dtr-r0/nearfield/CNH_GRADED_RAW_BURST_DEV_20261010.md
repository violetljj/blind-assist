# 固定阈值短促 raw 证据对照

2026-10-10，EXPLORE，已消费 ideal/共向模拟 Development。承接[剩余检出诊断](CNH_GRADED_REMAINING_DETECTION_DEV_20261010.md)，固定 head50 主工作点及 both、baseline 对照。

**单帧 raw 追加能挽回一部分晚检和静默，但 clear/pass 成本明显增加，不默认采用。** Validation head50 的 HEAD 及时救回 8/5/9，BODY 救回 9/15/7，各高度每 seed /384；clear 新受提醒 clip 增加 51/87/45，各 /512。同阈值 2-of-3 几乎不救回，故“加持续性就能低成本保留短促收益”在此对照不成立。大部分低反射横细杆仍静默，主要瓶颈继续交给 allbins 的 M3/ordinary 分支及独立 top8 joint 追加分支的目标证据保留诊断。

## 固定机制与边界

在输出前声明主方案 raw_single：当前 ordinary raw 分数 ≥ 已有 ordinary single 阈值，仅在父 grade0 处追加 grade1。对照 raw_k2of3：过去至多 3 个已观测时刻中至少 2 次 raw ≥ 同一阈值，同样只追加 grade1。另保留 fixed 身份对照。三个机制 × baseline/both/head50 × 3 seeds × cal/validation，共 54 cells。阈值不变，没有拟合、按成本重校准、挑 seed、窗口扫描或通知层操作。

每个 scene/replica/query 在缓存 f3 空历史启动；raw_single 可于 f3 触发，raw_k2of3 最早 f4，且只使用 f3/f4。不同场景、replica、query 没有历史串联；原 grade1/grade2 全部保留，强提醒逐 slot 一致。零损失/零推迟来自追加构造，非统计非劣证明；raw≥single 也不是目标对应证据的证明。

输入均为已有分数/grades 缓存。每 split 384 scenes × 4 replicas × 13 frames × 2 queries；每高度 384 contact 流，每目标族 96；purepass 256 clips/3328 union slots，clear 512 clips/6656 union slots。f3..13 为及时，f14..15 为晚检，f3..15 全无提醒为静默。时序只报帧，不将名义 200ms 代理解释成设备延迟。每个 seed 独立报告，955/956/957 是完整 seed 2026100955/956/957 的简称。

## Validation 主工作点效果

表内三元组均为 955/956/957；及时/晚检/静默各高度的分母均 /384。救回、提前与完整窗救回为逐 scene/replica/contact query 配对；提前只数两边都及时的流，完整窗救回只数父完整窗静默而候选完整窗有提醒。

| head50 方案 | HEAD 及时 | HEAD 晚检 | HEAD 静默 | BODY 及时 | BODY 晚检 | BODY 静默 |
|---|---|---|---|---|---|---|
| fixed | 352/357/351 | 15/14/17 | 17/13/16 | 311/316/320 | 12/10/12 | 61/58/52 |
| raw_single | 360/362/360 | 8/9/9 | 16/13/15 | 320/331/327 | 10/8/8 | 54/45/49 |
| raw_k2of3 | 353/357/351 | 14/14/17 | 17/13/16 | 311/317/320 | 12/11/12 | 61/56/52 |

| 对 fixed 配对 | HEAD及时救回 | BODY及时救回 | HEAD共同及时提前 | BODY共同及时提前 | HEAD完整窗救回 | BODY完整窗救回 |
|---|---|---|---|---|---|---|
| raw_single | 8/5/9 | 9/15/7 | 93/75/89 | 62/89/62 | 1/0/1 | 7/13/3 |
| raw_k2of3 | 1/0/0 | 0/1/0 | 8/9/8 | 4/3/2 | 0/0/0 | 0/2/0 |

全部 54 cells 的及时/完整窗损失与两边都检出时的推迟均 0，强提醒不变。raw_single 的物理 contact-query union 及时 663/673/671 → 680/693/687，各 /768，救 17/20/16，提前 155/164/151 个共同及时事件。其提前分布保留在 metrics：validation HEAD 共同及时提前 1..7/1..7/1..7 帧，BODY 1..6/1..7/1..6 帧；共同及时全部配对的中位提前为 0 帧，不能只拿提前子集表达总体。

原晚检 → 及时：raw_single HEAD 8/15、5/14、8/17，BODY 3/12、3/10、4/12。原静默 → 及时：HEAD 0/17、0/13、1/16，BODY 6/61、12/58、3/52；另有 HEAD 1/0/0、BODY 1/1/0 从静默变晚检。前诊断中的 raw_cross_without_smooth 只是机会计数；这里才是完整父策略下的实际配对结果。

BODY 横杆原静默 35/34/33 各 /96，raw_single 后仍静默 30/25/31；其中 rho=.25 原静默 32/30/30 各 /48，救回及时 2/6/2，仍静默 30/24/28。标牌边缘原静默 21/20/17 各 /96，救回及时 1/3/1；仍静默 20/16/16，seed956 另有 1 件变晚检。短促路径能救部分弱目标，但不会消除主要静默群。

## 追加轻提醒成本

以下为 HEAD/BODY union，按每个 clip 在同一帧有任一 query 提醒计一个 slot；grade1 的追加不升级为 grade2。clear 分母 slots /6656、clips /512；purepass slots /3328、clips /256。新 clip 指父完整窗没有提醒、候选完整窗出现提醒，另保留已有 clip 首报提前数。

| head50 方案 | clear slots | clear clips | purepass slots | purepass clips |
|---|---|---|---|---|
| fixed | 89/131/102 | 55/84/65 | 711/871/783 | 150/176/164 |
| raw_single | 162/273/175 | 106/171/110 | 865/1022/932 | 190/203/202 |
| raw_k2of3 | 99/149/115 | 57/87/68 | 772/906/830 | 154/176/166 |

raw_single 新 clear slots 73/142/73、新 clear clips 51/87/45，已有 clear clips 首报提前 5/15/8；新 purepass slots 154/151/149、新 clips 40/27/38，已有 purepass clips 首报提前 63/78/68。raw_k2of3 新 clear slots 10/18/13、新 clips 2/3/3；新 purepass slots 61/35/47、新 clips 4/0/2。这些是完整窗成本，不能把单帧短促提醒记作零成本。HEAD/BODY 分开、强/轻分开、slots/clips/segments/最长持续帧均在 metrics 保存。

## cal 与控制工作点

cal 不用于挑方案或阈值。head50 raw_single 及时救回 HEAD 12/8/10、BODY 18/17/11；完整窗救回 HEAD 4/1/4、BODY 11/14/8。其新 clear slots 175/180/146、clips 92/119/92；新 purepass slots 160/150/143、clips 31/29/41。cal raw_k2of3 及时救回 HEAD 0/1/0、BODY 2/2/0。

validation both 的 raw_single 及时救回 HEAD 6/5/9、BODY 9/15/7；baseline 为 HEAD 8/14/14、BODY 13/21/15。both 新 clear slots 70/140/71、clips 48/85/43，purepass slots 153/145/148、clips 40/24/38；baseline 新 clear slots 80/150/86、clips 57/93/57，purepass slots 164/167/160、clips 41/35/42。全部控制、cal、2-of-3、query 时钟及前后分层均保留，未因 head50 更低成本而省略其它工作点。

## 决定、核验与交付

保留 raw_single 作为可回放的短促机会对照，不改默认分级/检测策略；raw_k2of3 本固定 recipe 停止，不继续调窗口/阈值。当前最强方向仍是评价侧目标证据的分支归因：allbins 分别进入 M3 finite-volume 投影/累积与 ordinary token 读出，独立 expanded-gate/localpeak/top8 路径进入 joint addition，top8 不喂给 M3；先说明弱目标证据在哪个分支和环节衰减，再做针对该环节的保留机制。已有 raw 旁路证明平滑会压制一些有用证据，但 clear 大幅增加说明 raw 过线同时包含较多非 contact 提醒，尚未形成低成本替代方案。

独立核验未导入 producer：用移位递推重建 2-of-3、独立 minimum clock/成本段统计复算全部 54 grade cells、及时/完整窗配对、物理 contact union、强/轻/query/union 成本及 165888 ledger 行；全部 PASS。输入和源 hash、父 grades 重建一致，首次 f3/f4、跨 stream reset 与间隔单脉冲 fixture 通过。一次主执行内部 2.203s、主命令 3.343s；一次独立核验内部 2.297s、核验命令 3.534s。预算 main120 + verify90 + report90 = 300 CPU command-wall seconds，GPU0；其它定向读写与报告保守按 90s 上限记账，共 96.877/300s，此为保守记账而非精确总耗时。没有任务常驻进程或付费分配。

源：`cnh_graded_raw_burst_dev.py`、`audit_cnh_graded_raw_burst_dev.py`。载荷 `artifacts.local/work/cnh-graded-raw-burst-dev-20261010/`（canonical junction 落 F:）：PLAN、metrics、summary、cohorts、ledger、cal/validation grades、receipt 和 verification。旧输出不覆盖，固定对照执行完即停止。仅已消费受控模拟 Development、constant contact truth 与短缓存；seed/replica/frame 相关，诊断后选机制，不构成新结构独立确认、物体身份、实机、用户收益或安全证明。原 M3/5格/L2/480/weak_pass 及各自旧 stop 保留。
