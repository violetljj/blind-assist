# 保留BODY追加、校准HEAD增量成本

2026-10-10；EXPLORE，起始 `abda0a31`（接续CNH交付 `d150c8fe`，其后为独立RGB更新）。用户“继续”接续[BODY定向对照](CNH_GRADED_PEAK_BODY_ONLY_DEV_20261010.md)。

**预声明50%档保留HEAD救回1/13/5件及提前16/33/19件，BODY救回4/6/8、提前53/31/44全部保留；新增clear由23/40/27降至16/24/22格，新增pass片段由1/12/7降至1/8/6。比直接关闭HEAD保留了更多细障碍收益，但相对原两高度追加仍失去HEAD救回3/1/1件、推迟15/26/22件。保留head50作为降低打扰的候选工作点，原both仍作高检出对照，不据本轮默认替换或升级App。**

## 校准规则与证据身份

复用固定score_current分数、原c15_p64阈值与全部原strong/fullsingle。BODY逐slot与原both相同；HEAD只提高追加阈值，不撤原参照强轻档。运行时HEAD追加仅要求原HEAD grade0、分数有限且≥新阈值；不读取标签、作者分组或未来帧。

先PLAN，再仅使用原cal背景9/11选阈值；8/10仍为旧fit部分，不用于本轮定参。把原both相对body_only实际新增的联合clear、纯pass slots作为HEAD原贡献，分别取floor(25%/50%/75%×贡献)为上限。成本计算只在BODY-only两高度均静默的slot计HEAD增量；这条静默条件只用于cal计费，不是运行时筛选条件。阈值为≥原阈值且满足两上限的最低完整ties边界，使用被排除分数的nextafter。无可计费分数时该单轴不约束；预算为0时仍保留零联合成本的HEAD追加，不等于HEAD全关。

三档全部报告，50%为预声明主档，未根据validation挑胜者。原baseline/both/body_only三对照精确继承。ideal cal/validation各384×4×13×2，共36cells/9cuts/110592ledger；f3–13及时，f14–15晚。三seed分别报告，不投票或合成独立样本。

cal9/11 clear/pass分母为3328/1664 union slots；原HEAD贡献clear4/13/9、pass7/13/10。主50%目标clear2/6/4、pass3/6/5，实际clear2/6/1、pass0/5/5，剩余预算也保留。主HEAD阈值1.6571274825139004、1.1000813278886314、1.3657576015849209，BODY仍用原1.335735804933452、.8851012650937939、1.090820299871486。cal上限不保证validation等比例成本。

本轮是在已有validation成本与BODY-only损失后提出的探索，不能称独立确认。旧height filter/replace负结果保留；本轮是固定BODY追加后的HEAD增量校准，不将不同配方混作同一机制结论。

## Validation完整取舍

表中均按seed955/956/957。检出每高度384接触事件；clear512clips/6656slots，纯pass256clips/3328slots。救回/提前相对完整原baseline，不把pass/clear新增提醒计入接触救回。

| 策略 | HEAD救回 | HEAD原已及时提前 | 新增clear slots | 新增clear clips | 新增pass slots | 新增pass clips |
| --- | --- | --- | --- | --- | --- | --- |
| both原两高度追加 | 4/14/6 | 31/59/40 | 23/40/27 | 21/31/23 | 25/57/36 | 1/12/7 |
| BODY-only | 0/0/0 | 0/0/0 | 12/12/18 | 11/8/14 | 18/22/23 | 0/7/4 |
| HEAD25% | 1/10/5 | 16/20/15 | 16/20/22 | 15/15/18 | 20/37/29 | 1/8/5 |
| HEAD50%（主档） | 1/13/5 | 16/33/19 | 16/24/22 | 15/19/18 | 20/40/30 | 1/8/6 |
| HEAD75% | 1/14/5 | 16/45/24 | 16/30/25 | 15/22/21 | 20/49/30 | 1/9/6 |

三个HEAD档及both、BODY-only均保留BODY救回4/6/8、提前53/31/44；这是BODY策略逐slot相等的构造事实。955三个档位在validation上述汇总相同，不能从小cal计数推出连续稳定曲线；956高档保留全部14件HEAD救回，但仍推迟14件共同及时事件。

主50% timely HEAD352/357/351、BODY311/316/320；contact真实高度联合663/673/671各/768。相对both少3/1/1件HEAD及时检出，共同及时HEAD推迟15/26/22件，BODY损失/推迟0；相对原baseline所有query及时损失/推迟0。这里每scene仅一个contact高度，联合损失恰等于HEAD损失；不推广到双高度接触场景，也不与旧any-height指标互换。

主50%保留HEAD新增horizontal1/9/3、sign_edge0/4/2；相对both放弃horizontal2/1/1、sign_edge1/0/0。损失仍落在当前优先的细障碍，而非无用提醒。低成本候选和高检出对照之间没有用户给定效用权重，不宣称统一最优。

主50% clear总89/131/102slots、55/84/65clips，pass总150/176/164clips。相比both省clear7/16/5slots、6/12/5clips，pass5/17/6slots、0/4/1clips；全部原strong及已有light保持，不采用静默。

## 分级与持续打扰

强档及首次时刻在全部策略完全相同：HEAD315/313/323、BODY267/257/282及时，strong clear58slots/28clips，strong pass355/362/461slots、89/103/115clips。ordinary_OR强转轻0。主档及时light-only HEAD37/44/28、BODY44/59/38各/384，含义是及时轻提醒且截至f13无强提醒。

主档联合最高等级light成本clear31/73/44slots、27/58/39clips，pass356/509/322slots、139/165/148clips；clear最长连续3/4/2帧、pass5/7/5帧。strong/light clips可重叠，不能相加为any。已有passclip首提醒还提前11/19/16件，已有clearclip提前1/2/2件；新增clip计数不能代表全部打扰成本。全部细项、首时刻及帧差直方图保存在metrics/ledger，帧格并非实际声振次数或真实时延。

## 核验与交付

预算main90＋audit90＋整合120＝300 CPU command-wall seconds；本轮保守main/统计30、audit4、整合全额120，合计154/300s。主内部1.828s、命令4.209s，audit内部2.781s、命令3.891s；GPU/fit/预测/新raw/新采样0。本轮完成固定三比例配方并停止继续扫阈值/比例，不扩展旧停止run。

独立audit未import主程序，独立枚举完整score ties验证最低可行阈值与两轴cut/cap/unused，独立重算cal切分、36cells/summary、110592ledger、强轻/成本/分组/物理联合/时钟，6336255项检查与30input hashes前后一致PASS；三旧控制的grades与旧schema全部metrics精确相同。没有科学条件修订或实验失败重跑。源码cnh_graded_peak_head_cal_dev.py、audit_cnh_graded_peak_head_cal_dev.py；载荷 `artifacts.local/work/cnh-graded-peak-head-cal-dev-20261010/` 保存PLAN/cal_partition/9cutoffs/两split grades/metrics/summary/ledger/receipt/audit。无常驻任务进程。

保留head50成本候选及both高检出对照，不默认升级。下一优先在固定两工作点核对首次轻提醒→强提醒的事件链，分别看contact、pass、clear是否先得到有用轻提醒、是否后来升级，以及持续/碎片化提醒成本，再决定事件级“先报后升”是否需要不同于已有8帧累积与5帧平滑的机制；这是后续候选，尚未执行。当前深度差仍无静默依据，temporal不默认追加。

原M3/5格/L2/body truth/fullbin/480、weak_pass及各旧stop保留。全部仅已消费受控模拟Development；未接入App，不是free/coverage、实机、安全或真实用户收益证明。
