# 固定工作点的弱/细目标漏检、晚检与首次证据诊断

2026-10-10；EXPLORE，起始 `b3da23a9`。用户同意从通知层回到检测本身。共向ideal，固定head50主工作点、both高检出对照和baseline；未改任何模型、阈值或提醒策略。

**最优先是低反射率BODY横细杆的整窗静默，其次BODY标牌边缘；HEAD标牌边缘和BODY弱竖杆主要是晚检。现有走廊内峰“是否存在”已经饱和，不能解释这些差异；部分晚检另有原始分数过线但5帧平滑后不过线的线索。下一应做评价侧的目标bin支持与保留链定位，再决定检测改法。**

## 固定范围与完成结果

复用既有cal/validation，各384scenes×4replicas×13帧×2query；只诊断真实contact高度。三个seed955/956/957（完整2026100955/956/957）、baseline/both/head50全部保留，共13824contact流、55296快照、4608工作点配对和2346分层摘要。没有选择表现最好seed、重新采样、fit、forward或raw直方图重建。

f3–13为及时，f14–15为晚检，f3–15没有正grade为整窗静默。每split/seed/工作点/高度contact分母384，每形状族96；rho分层每族48。首次分数取f3，这是缓存输出的左边界，不是目标第一次可见；200ms/帧仅名义clock，不能作设备端时延。

Validation head50如下，三个数字均按955/956/957：

| 接触高度及形状 | 及时 | 晚检 | 整窗无提醒 |
| --- | --- | --- | --- |
| BODY横细杆 | 61/62/63 | 0/0/0 | 35/34/33 |
| BODY标牌边缘 | 74/75/78 | 1/1/1 | 21/20/17 |
| BODY竖杆 | 80/83/83 | 11/9/11 | 5/4/2 |
| HEAD横细杆 | 71/75/69 | 8/8/11 | 17/13/16 |
| HEAD标牌边缘 | 89/90/90 | 7/6/6 | 0/0/0 |

protrusion两高度和HEAD竖杆均96/96及时。总体HEAD及时352/357/351、晚15/14/17、静默17/13/16；BODY及时311/316/320、晚12/10/12、静默61/58/52，各/384。

BODY在both与head50逐slot相同，故其漏检不能归因于HEAD降成本。both的HEAD及时355/358/352；head50少3/1/1件，并推迟15/26/22件共同及时HEAD事件，这项既有检出与打扰取舍单列，未把它当新模型问题。通知层不参与本轮数值生成，固定grade首报语义沿用。

## 首次分数与峰支持揭示了什么

**弱反射率比单纯贴边更能定位BODY横杆问题。** rho=.25时BODY横杆48件中静默32/30/30；rho=.65为3/4/3。BODY标牌边缘弱组静默16/17/16，高rho为5/3/1。BODY竖杆的全部晚检/静默位于rho=.25；高rho竖杆48/48及时。rho为模拟作者参数，不是实测材质反射率。

HEAD横杆全部静默落在in1cm浅侵入；BODY静默在in1/in4/in12cm均存在，总体每档17–23/128件，不能把BODY问题只解释成接触/擦边精确分界。完整形状、rho、placement、background分层见summary.json。

**通用峰支持的存在性失去区分力。** 所有contact流的及时11帧均有至少一个既有top8 inner候选，首次inner为f3；失败流也一样。静默BODY及时窗最强inner幅值中位1.447/1.432/1.417，HEAD为1.504/1.504/1.490（正signed-log与query membership加权的缓存单位）。这证明已有通用峰描述始终有数值输入，不能证明弱目标回波被保留，更不能证明目标检出、free或coverage。背景也能提供这些峰，支持extent是angular-node几何范围，不是物体尺寸。

**首次和最强及时分数都落在工作点以下。** 静默BODY固定joint score_current的f3 margin中位−1.863/−1.357/−1.599；及时窗最佳margin仍为−1.459/−.996/−1.286。HEAD对应f3 −2.607/−1.372/−2.164，及时最佳−2.176/−.811/−1.880。margin为各自固定阈值之差，不是概率，不跨模型或seed比较尺度；baseline没有启用joint追加，其joint margin仅作证据参照。

失败流的M3/local/ordinary平滑路径全部低于固定阈值；both/head50的joint追加也低于各自阈值。源码重建固定grades与父缓存逐值一致，排除了本轮连接错阈值或通知抑制导致首报丢失。负margin本身是失败定义的结果，诊断价值在首次/最佳距离、分层和原始/平滑差异，不能据此笼统建议降阈值。

**平滑对部分晚检可能有影响，但不能包办整窗漏检。** HEAD晚检15/14/17件中，ordinary原始分数在及时窗过single线、全部及时平滑分数却不过线者8/5/8件；BODY为3/3/4件。静默HEAD只有0/0/1件符合，BODY为6/12/3件。原M3或local原始分数过old-fusion线但平滑后不过线的晚检HEAD4/5/7、BODY10/8/8，和ordinary组可重叠，不能相加。这里只是同流原始与既有5帧加权平滑的差异；没有评价直接报raw、缩窗或新时序规则，短促raw过线可能也带来pass/clear成本。

每流保留f3、first_inner、及时窗joint分数最佳帧和首次提醒四个快照，含ordinary raw/smooth margin、joint margin、rank0/inner幅值、内支持比例/count、深度和extent；无首次提醒的快照显式缺失。原始13帧数组继续由输入hash指向，未重写源缓存。

独立补充对全部1226条静默流检查完整13帧，保存75组摘要：validation head50静默流仍13/13帧有inner支持。BODY横杆整窗最佳joint margin中位−1.369/−.969/−1.140，BODY边缘−1.643/−.772/−1.339，HEAD横杆−2.176/−.811/−1.679。多数静默流并非只在f13截止前差最后一次峰；延长到当前完整f15窗口也没有过固定工作点。完整窗补充见verification/silent_full_window.csv及summary，未把窗口延长到新观测。

## 下一步与边界

推荐先把低rho BODY横杆与标牌边缘的静默流送入**评价侧目标bin证据审计**：用已知模拟目标及遮挡几何定位其应有原生zone/bin，检查目标相容证据是否出现、是否进入top8、投影/累积后是否保留，再关联ordinary/M3输出。该目标几何只允许评价与归因，不得进入检测输入。它能区分“通用峰全是背景”“目标弱峰被筛掉”“目标相容支持保留但读出低分”；目前三者未被分开。这比继续给饱和的“有inner峰”指标加权更有决策价值。

并行的次优先是对上述raw过线而smooth不过线的晚检流做固定短促证据对照，逐query报告及时救/损、首次时刻和完整pass/clear成本，保留无过线静默流作为适用范围对照。下一步为机制提议，本轮没有运行新检测规则或目标归因，不把raw过线数当已救回件数。

本轮仅已消费模拟Development；shape/rho/placement/background和contact为评价侧分层，未进运行时特征。相关scene/K/帧与seed不当独立确认。M3/原5格/L2/body truth/fullbin/480/weak_pass和各旧stop保持；没有Android/硬件/用户安全结论。

载荷 `artifacts.local/work/cnh-graded-remaining-detection-dev-20261010/`：PLAN、events、snapshots、working_point_pairs、summary、receipt；源代码同目录的cnh_graded_remaining_detection_dev.py。预算main120＋核验120＋整合180＝420 CPU command-wall seconds，GPU0；主调用5.7965s、内部4.625s。主执行与定向读取保守按30s、核验与整合分别按其全额120/180s扣额，共330/420s；这不是精确总耗时。所有计算为短CPU缓存统计，没有常驻任务资源。

独立核验直接读取manifest与缓存，未import主诊断程序或原分析模块；独立重算平滑、固定grade、全部13824事件/55296快照/4608pairs及旧contact首时刻账本，1248768项字段核对通过，39项继承hash与直接读入hash前后一致。内部10.562s，完整命令墙时未单独捕获，核验阶段按全额120s计。另从events重算全部2346摘要的成员/结果/直方图/过线计数及每字段分母与中位数，96186项通过，内部1.657s、Python命令1.9558s。完整窗静默补充单列，没有重复主科学运行。默认Python缺NumPy的两次环境尝试留在execution_receipt，后复用现有科学runtime，未安装依赖。验证代码与收据保存在载荷verification/；文档索引15热文件/131链接通过。
