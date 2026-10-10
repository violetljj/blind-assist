# BODY定向追加的检出与打扰取舍

2026-10-10；EXPLORE，起始 `59696df6`。用户“推进”接续[匹配诊断](CNH_GRADED_PEAK_MATCH_DEV_20261010.md)，执行固定评分与阈值的BODY定向追加对照。

**BODY定向追加保留4/6/8件BODY救回及53/31/44件提前，将新增clear从23/40/27降至12/12/18格，但放弃HEAD救回4/14/6件及提前31/59/40件。被放弃的救回全部是横向细障碍或标牌边缘，本批没有其他接触高度的及时提醒覆盖这些救回。不能把HEAD新增提醒整体视为无用成本；保留两高度联合候选，BODY-only仅作为明确的成本取舍，不默认采用。**

## 固定比较

全部ideal cal/validation各384 scenes×4噪声replicas×13帧×2查询，ordinary三seed955/956/957分别报告，不集成。运行时只用原强/轻分数、缓存score_current及query高度；标签与作者分组仅评价。

原参照baseline为ordinary_OR强档：oldfusion或ordinary≥addition；非强且ordinary≥single给轻档。both复用[联合评分](CNH_GRADED_PEAK_JOINT_DEV_20261010.md)中档c15_p64，仅在baseline0且有限score_current≥原cutoff处追加轻档；body_only只允许BODY追加，HEAD逐slot等于baseline。三seed原cutoff为1.335735804933452、.8851012650937939、1.090820299871486，不重校准、不再预测、不拟合。

先写PLAN再统计18cells。继承24输入hash、全部原strong和已有light；BODY-only的BODY逐slot与both相同。上述相等/原参照无损是规则构造与实现检查，不是独立非劣证据。固定f3–13为及时，f14–15为晚；ledger完整首时刻与及时首时刻分列。只报告帧提前，不换算提醒全链路时延。

预算为main90＋audit90＋整合120＝300 CPU command-wall seconds，GPU/拟合/预测/新raw/新采样0。范围仅固定三策略的完整检出、时钟、成本和分组比较及实现修复；完成后停止本配方扫阈值/重新分配预算。

## Validation效果

以下依次seed955/956/957；HEAD与BODY接触各384事件，clear512 clips/6656 joint slots，纯pass256 clips/3328 joint slots。slot是scene×replica×帧的两高度并集，不是实际提醒次数。

| 项目 | 原参照 | 两高度追加 | BODY定向追加 |
| --- | --- | --- | --- |
| HEAD及时检出 /384 | 351/344/346 | 355/358/352 | 351/344/346 |
| BODY及时检出 /384 | 307/310/312 | 311/316/320 | 311/316/320 |
| 相对原参照HEAD救回 | 0/0/0 | 4/14/6 | 0/0/0 |
| 相对原参照BODY救回 | 0/0/0 | 4/6/8 | 4/6/8 |
| 原已及时HEAD事件提前 | 0/0/0 | 31/59/40 | 0/0/0 |
| 原已及时BODY事件提前 | 0/0/0 | 53/31/44 | 53/31/44 |
| 真实contact高度联合及时 /768 | 658/654/658 | 666/674/672 | 662/660/666 |
| clear总slots /6656 | 73/107/80 | 96/147/107 | 85/119/98 |
| clear总clips /512 | 40/65/47 | 61/96/70 | 51/73/61 |
| pass总slots /3328 | 691/831/753 | 716/888/789 | 709/853/776 |
| pass总clips /256 | 149/168/158 | 150/180/165 | 149/175/162 |

BODY定向相对both减少clear11/28/9 slots、10/23/9 clips，pass7/35/13 slots、1/5/3 clips；相对baseline仍新增clear12/12/18 slots、11/8/14 clips，pass18/22/23 slots、0/7/4 clips。没有设定用一个检出交换多少误提醒的效用权重，所以这些数字不构成单一胜负判定。

BODY提前事件中正提前帧的中位数2/1/2，范围1–7/1–8/1–7；把未提前事件也包含的both-timely中位数各为0。仅比较已及时交集不能代替救回计数。

BODY-only相对原参照两高度及时损失/推迟均0。相对both则HEAD及时损失4/14/6、共同及时推迟31/59/40，BODY损失/推迟0；所有被关闭的仅是新增轻提醒。真实contact查询并集也损4/14/6件并推迟31/59/40件。本批768接触事件每scene只一个contact高度，不能由此推广到两个高度都接触的场景。旧E.summary的physical_contact_any_height还会计非接触高度的提醒，本报告另算contact查询上的联合指标，两者不能互换。

### 弱细目标分组

以ledger中category=contact且baseline及时首时刻缺失、both及时首时刻存在定义新增救回，不将pass/clear的新增提醒计入。被放弃的HEAD救回：horizontal3/10/4、sign_edge1/4/2，合计4/14/6；vertical/protrusion各0。保留的BODY救回：horizontal0/3/1、vertical2/2/2、sign_edge2/1/5，合计4/6/8。

两高度的弱细目标均有收益，简单按HEAD整轴关闭会直接放弃横向细障碍和边缘的收益。完整family/background分组检出与原oldfusion配对保存在metrics.json；上述新增救回按完整baseline配对，参照不同，不混算贡献。

## 强轻提醒与持续打扰

三策略所有strong slots/首次时刻相同；及时strong HEAD315/313/323、BODY267/257/282。strong clear各58slots、28clips，strong pass355/362/461slots、89/103/115clips。ordinary_OR强转轻为0；历史oldfusion与完整single参照身份保留。

BODY-only及时light-only contact HEAD36/31/23、BODY44/59/38，各/384；这是及时轻档且截至f13无强档的事件，与strong及时检出互斥。query轻检出、联合light成本和any成本分别存储，不把有强档覆盖的另一query轻档算成联合轻提醒。

| BODY-only联合最高等级light成本 | 955 | 956 | 957 |
| --- | ---: | ---: | ---: |
| clear slots /6656 | 27 | 61 | 40 |
| clear clips /512 | 23 | 47 | 35 |
| pass slots /3328 | 354 | 491 | 315 |
| pass clips /256 | 138 | 163 | 146 |
| clear最长light连续帧 | 3 | 4 | 2 |
| pass最长light连续帧 | 5 | 7 | 5 |

强/轻clips可以重叠，不能相加得到any clips。BODY新增candidate本身clear12/12/18slots落在12/12/17clips，最长各1帧；pass18/22/23slots落在15/19/22clips，最长各2帧。与baseline相比实际新passclips仅0/7/4，但已有passclip首提醒提前10/5/12件，因此“新clip0”不能读成“无额外打扰”。已有clearclip首提醒提前1/1/2件；每一项分母与时间口径在载荷保留。

## Cal与证据身份

整个已消费cal含旧fit与旧cutoff两部分，本轮不重新选择阈值。BODY-only cal timely HEAD352/344/351、BODY324/325/329，BODY救回4/5/11、提前63/39/79。原both HEAD救回5/13/12、提前16/72/39；限制后这些全部放弃。

cal中both新增clear18/26/22slots、13/19/19clips，BODY-only12/6/10slots、9/3/9clips；both新增pass17/38/31slots、5/6/4clips，BODY-only9/18/19slots、4/4/4clips。这不是cal9/11选择时的c15/p64上限口径，也不声称validation等成本。

BODY-only候选是在看过validation匹配成本后提出的探索，仍使用已消费Development；不能称新验证或独立确认。三seed、K噪声重复及帧相关，不合并成独立样本池。

## 核验与下一决定

独立audit未import主程序，独立重算first/min时钟、逐帧连续段、强轻/any成本、全部分组、query与contact高度联合救损/早晚、追加与实际新增成本、18summary及55296ledger；共2982897 scalar/array checks，24输入hash前后核对PASS。主程序内部1.016s、命令2.869s；audit内部1.546s、命令2.567s。本轮CPU命令保守计main及统计30/90s、audit3/90s、整合全额120/120s，合计153/300s，GPU0。无常驻任务进程。

源码cnh_graded_peak_body_only_dev.py、audit_cnh_graded_peak_body_only_dev.py；载荷 `artifacts.local/work/cnh-graded-peak-body-only-dev-20261010/` 含预声明PLAN、24input hashes、两split grade、metrics、summary、ledger、receipt及audit/result。无科学条件修订或实验失败重跑；检查与文档记录随交付完成。

保留两高度current联合候选与完整原强轻提醒，BODY-only作为成本取舍对照，不默认关闭HEAD或采用静默。下一优先分高度calibration：保留BODY候选，在cal侧控制HEAD的实际新增成本，完整报告HEAD弱细目标救回与提前的保留程度；这是后续候选，尚未执行。旧高度filter/replace负结果仍属于旧配方，本轮没有恢复旧模型、预算配方或停止试验。

原M3/5格/L2/body truth/fullbin/480、weak_pass和各旧run停止身份保留；未接入App。所有结果仅受控模拟Development，不是物体归因、free/coverage、实机、安全或真实用户收益证明。
