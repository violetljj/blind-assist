# CNH校准尾部、中心锚定OR与高度混合读出（Development）

**中心锚定OR保住原报警并增加部分+3°接触命中，但“零额外校准成本”没有延伸为零额外validation成本；高度混合读出仍有双条件取舍。** OR全部政策格对center无损失来自报警集合超集，不是风险证明。hybrid的BODY虽然读取center raw，两高度共享的校准θ发生变化，BODY报警也会下降。M3、原独立weak_pass候选与旧停止结果保留，不从validation选择赢家或升级基线。

40%是三个预先声明点中的展示例。weak OR在+3°相对center，总体HEAD救/损3/0、1/0、5/0，BODY4/0、8/0、9/0（各/384）；1 cm HEAD3/0、0/0、5/0，BODY2/0、4/0、4/0（各/128）。clear clip却从148/120/140升至155/133/150各/512，pass clip从136/122/122变为136/124/128各/256。ideal验证条件也不等于cal：该点HEAD最多救1个，BODY无新增，pass仍增加4/1/2个。

| seed | 40% +3°weak策略 | H/B及时各/384 | clear clip/512 | pass clip/256 |
| --- | --- | --- | --- | --- |
| 955 | center | 306/278 | 148 | 136 |
| 955 | OR | 309/282 | 155 | 136 |
| 955 | hybrid | 302/275 | 139 | 135 |
| 956 | center | 289/258 | 120 | 122 |
| 956 | OR | 290/266 | 133 | 124 |
| 956 | hybrid | 287/253 | 131 | 125 |
| 957 | center | 299/259 | 140 | 122 |
| 957 | OR | 304/268 | 150 | 128 |
| 957 | hybrid | 300/259 | 144 | 127 |

weak hybrid在同一40% +3°点的1 cm HEAD救/损为3/1、4/1、4/0，BODY为0/0、0/2、0/0；总体HEAD为3/7、5/7、4/3，BODY为0/3、0/5、0/0。这些浅HEAD收益值得讨论，但不意味着整体命中与成本均优于center，不能以OR结构性无损失为门槛否定hybrid。全20/30/40%、三seed、两arm和两分支结果见完整表；0…551之类clear增量单位是slot/6656，不是clip/512。

下一优先保留center作为锚、OR作为明确成本的附加召回候选：先在有依据的方向误差/背景证据中复核附加报警成本，而非用cal的零增量名称暗示实物低误报。机制是保留原报警，仅让侧方向证据补充；潜在收益是浅边界救回，最薄弱假设是cal尾部足以约束新背景/姿态的报警。最小下一检查可冻结本轮阈值与方向合同，在独立背景和方向误差片段上并列center与OR的1 cm救回及clear/pass clip增量；hybrid继续作为共享阈值取舍参照，不再按本轮validation调θ、权重或选点。该后续未运行，尚未建立实物、导航或安全结论。


## 冻结规则与分母

原始身体接触truth保持；方向合同沿用人工±3°，ideal候选−3/0/+3°，+3°候选0/+3/+6°，未认证实物误差范围。复用上一轮全部raw分数，checkpoint不回放，无模型forward、训练、阈值重选于validation、GPU、新光子或采集。

center沿用上一轮single的每模型/seed/20/30/40% ideal-cal阈值。锚定OR使用center≥θsingle或off≥θoff；off逐帧取lower/upper raw最大值后沿用因果last5平滑，不能替换成已平滑分数的最大值。θoff只从ideal cal中center未占用的joint-clear slot及整段未报警pass clip的限制值选择：取全部限制值最大值的nextafter(+inf)，整组tie拒绝；无eligible单位时θoff=+inf，退化为center。本轮只做零增量校准，不利用剩余预算。

OR零损失来自center报警集合的超集结构；ideal-cal clear slots/pass clips应与center精确相同，validation成本仍需实测分别报告。这不是给hybrid增加零损失准入门槛。hybrid逐帧HEAD=.25/.5/.25 raw加权，BODY=center raw；继承相同平滑，两高度仅一个共享θ，ideal-cal依原joint预算选最低可行whole-tie分数level，不分头阈值、不contact优化。

成本均覆盖f3–15全部13输出；contact及时只计f3–13，f14–15只计late。总体及时HEAD/BODY各/384；1 cm各/128=32scene×K4；clear clips/512、pass clips/256、clear slots/6656和报警段分别列出，不折算相减。108政策cell包括36center参照及72新policy，所有seed、arm、分支、20/30/40点保留；cal clear slot预算65、pass clip预算51/76/102，校准相同预算不等于validation相同成本。


## 全政策科学图

[OR/hybrid全三seed救损与分开的clip代价](../../../../artifacts.local/work/cnh-direction-anchor-dev-20261009/policy_tradeoffs.png)

每行是arm×分支，前9格为OR、后9格为hybrid，均按955/956/957的20/30/40点排列。及时救与损不抵消或折算clear/pass clip。OR全格损失0是center超集结构，不是实物安全或对hybrid的门槛。

## 全108 validation政策格

36center参照+72新政策。clearClip/512、passClip/256、clearSlot/6656与段独立报告；θ以6位小数展示，精确值见JSON。OR对应center/off双阈值，hybrid仅一个共享标量θ。

### ideal

| arm | seed | 点% | policy | 状态 | θ（OR为center/off） | H/B各384 | clearClip/passClip | clearSlot/段 | 1cmH/B各128 | 1cm vscenter H；B救/损(净) | 1cm vsM3 H；B救/损(净) | 整体vscenter H；B救/损(净) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 955 | 20 | center_single | CALIBRATED | 4.444754 | 264/227 | 0/53 | 0/0 | 68/56 | 0/0(+0)；0/0(+0) | 16/23(-7)；11/11(+0) | 0/0(+0)；0/0(+0) |
| control | 955 | 20 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 4.444754 / 11.726794 | 264/227 | 0/53 | 0/0 | 68/56 | 0/0(+0)；0/0(+0) | 16/23(-7)；11/11(+0) | 0/0(+0)；0/0(+0) |
| control | 955 | 20 | head_weighted_body_center | CALIBRATED | 5.268620 | 251/206 | 0/47 | 0/0 | 60/52 | 1/9(-8)；0/4(-4) | 12/27(-15)；9/13(-4) | 1/14(-13)；0/21(-21) |
| control | 955 | 30 | center_single | CALIBRATED | 2.590881 | 294/265 | 0/80 | 0/0 | 79/78 | 0/0(+0)；0/0(+0) | 24/20(+4)；28/6(+22) | 0/0(+0)；0/0(+0) |
| control | 955 | 30 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 2.590881 / 10.096044 | 295/265 | 0/82 | 0/0 | 80/78 | 1/0(+1)；0/0(+0) | 24/19(+5)；28/6(+22) | 1/0(+1)；0/0(+0) |
| control | 955 | 30 | head_weighted_body_center | CALIBRATED | 2.989800 | 288/259 | 0/74 | 0/0 | 79/73 | 1/1(+0)；0/5(-5) | 23/19(+4)；25/8(+17) | 3/9(-6)；0/6(-6) |
| control | 955 | 40 | center_single | CALIBRATED | 1.874375 | 310/280 | 1/96 | 1/1 | 81/88 | 0/0(+0)；0/0(+0) | 24/18(+6)；35/3(+32) | 0/0(+0)；0/0(+0) |
| control | 955 | 40 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 1.874375 / 10.096044 | 311/280 | 1/96 | 1/1 | 82/88 | 1/0(+1)；0/0(+0) | 24/17(+7)；35/3(+32) | 1/0(+1)；0/0(+0) |
| control | 955 | 40 | head_weighted_body_center | CALIBRATED | 2.234261 | 303/271 | 1/94 | 1/1 | 81/82 | 1/1(+0)；0/6(-6) | 24/18(+6)；31/5(+26) | 2/9(-7)；0/9(-9) |
| control | 956 | 20 | center_single | CALIBRATED | 6.057990 | 259/182 | 0/51 | 0/0 | 67/49 | 0/0(+0)；0/0(+0) | 13/21(-8)；6/13(-7) | 0/0(+0)；0/0(+0) |
| control | 956 | 20 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 6.057990 / 14.243341 | 259/182 | 0/52 | 0/0 | 67/49 | 0/0(+0)；0/0(+0) | 13/21(-8)；6/13(-7) | 0/0(+0)；0/0(+0) |
| control | 956 | 20 | head_weighted_body_center | CALIBRATED | 6.092933 | 253/181 | 0/51 | 0/0 | 63/49 | 0/4(-4)；0/0(+0) | 11/23(-12)；6/13(-7) | 0/6(-6)；0/1(-1) |
| control | 956 | 30 | center_single | CALIBRATED | 4.371076 | 280/219 | 0/73 | 0/0 | 77/58 | 0/0(+0)；0/0(+0) | 18/16(+2)；12/10(+2) | 0/0(+0)；0/0(+0) |
| control | 956 | 30 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 4.371076 / 11.669246 | 280/219 | 0/73 | 0/0 | 77/58 | 0/0(+0)；0/0(+0) | 18/16(+2)；12/10(+2) | 0/0(+0)；0/0(+0) |
| control | 956 | 30 | head_weighted_body_center | CALIBRATED | 4.383284 | 279/218 | 0/71 | 0/0 | 77/58 | 2/2(+0)；0/0(+0) | 19/17(+2)；12/10(+2) | 2/3(-1)；0/1(-1) |
| control | 956 | 40 | center_single | CALIBRATED | 2.736755 | 304/246 | 1/106 | 1/1 | 85/69 | 0/0(+0)；0/0(+0) | 23/13(+10)；20/7(+13) | 0/0(+0)；0/0(+0) |
| control | 956 | 40 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 2.736755 / 9.317640 | 304/246 | 1/107 | 1/1 | 85/69 | 0/0(+0)；0/0(+0) | 23/13(+10)；20/7(+13) | 0/0(+0)；0/0(+0) |
| control | 956 | 40 | head_weighted_body_center | CALIBRATED | 2.913773 | 300/244 | 1/104 | 1/1 | 82/67 | 0/3(-3)；0/2(-2) | 23/16(+7)；18/7(+11) | 3/7(-4)；0/2(-2) |
| control | 957 | 20 | center_single | CALIBRATED | 6.693934 | 260/211 | 0/55 | 0/0 | 69/53 | 0/0(+0)；0/0(+0) | 16/22(-6)；8/11(-3) | 0/0(+0)；0/0(+0) |
| control | 957 | 20 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 6.693934 / 15.271625 | 260/211 | 0/56 | 0/0 | 69/53 | 0/0(+0)；0/0(+0) | 16/22(-6)；8/11(-3) | 0/0(+0)；0/0(+0) |
| control | 957 | 20 | head_weighted_body_center | CALIBRATED | 7.031885 | 259/209 | 0/51 | 0/0 | 67/53 | 2/4(-2)；0/0(+0) | 16/24(-8)；8/11(-3) | 3/4(-1)；0/2(-2) |
| control | 957 | 30 | center_single | CALIBRATED | 5.054212 | 273/236 | 0/72 | 0/0 | 76/64 | 0/0(+0)；0/0(+0) | 21/20(+1)；17/9(+8) | 0/0(+0)；0/0(+0) |
| control | 957 | 30 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 5.054212 / 13.946710 | 273/236 | 0/75 | 0/0 | 76/64 | 0/0(+0)；0/0(+0) | 21/20(+1)；17/9(+8) | 0/0(+0)；0/0(+0) |
| control | 957 | 30 | head_weighted_body_center | CALIBRATED | 4.811739 | 279/243 | 0/76 | 0/0 | 77/66 | 2/1(+1)；2/0(+2) | 22/20(+2)；19/9(+10) | 7/1(+6)；7/0(+7) |
| control | 957 | 40 | center_single | CALIBRATED | 4.000479 | 287/259 | 0/98 | 0/0 | 80/77 | 0/0(+0)；0/0(+0) | 23/18(+5)；29/8(+21) | 0/0(+0)；0/0(+0) |
| control | 957 | 40 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 4.000479 / 13.946710 | 287/259 | 0/99 | 0/0 | 80/77 | 0/0(+0)；0/0(+0) | 23/18(+5)；29/8(+21) | 0/0(+0)；0/0(+0) |
| control | 957 | 40 | head_weighted_body_center | CALIBRATED | 3.946529 | 292/260 | 0/96 | 0/0 | 80/78 | 1/1(+0)；1/0(+1) | 24/19(+5)；30/8(+22) | 7/2(+5)；1/0(+1) |
| weak_pass | 955 | 20 | center_single | CALIBRATED | 2.189399 | 274/245 | 0/54 | 0/0 | 69/65 | 0/0(+0)；0/0(+0) | 16/22(-6)；17/8(+9) | 0/0(+0)；0/0(+0) |
| weak_pass | 955 | 20 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 2.189399 / 8.811336 | 274/245 | 0/54 | 0/0 | 69/65 | 0/0(+0)；0/0(+0) | 16/22(-6)；17/8(+9) | 0/0(+0)；0/0(+0) |
| weak_pass | 955 | 20 | head_weighted_body_center | CALIBRATED | 2.225964 | 268/244 | 0/55 | 0/0 | 67/64 | 3/5(-2)；0/1(-1) | 17/25(-8)；16/8(+8) | 3/9(-6)；0/1(-1) |
| weak_pass | 955 | 30 | center_single | CALIBRATED | 1.339579 | 299/269 | 3/76 | 3/3 | 77/80 | 0/0(+0)；0/0(+0) | 21/19(+2)；28/4(+24) | 0/0(+0)；0/0(+0) |
| weak_pass | 955 | 30 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 1.339579 / 6.983053 | 299/269 | 3/81 | 3/3 | 77/80 | 0/0(+0)；0/0(+0) | 21/19(+2)；28/4(+24) | 0/0(+0)；0/0(+0) |
| weak_pass | 955 | 30 | head_weighted_body_center | CALIBRATED | 1.608958 | 291/263 | 1/70 | 1/1 | 77/76 | 2/2(+0)；0/4(-4) | 22/20(+2)；26/6(+20) | 3/11(-8)；0/6(-6) |
| weak_pass | 955 | 40 | center_single | CALIBRATED | 0.851776 | 316/282 | 8/93 | 8/8 | 85/89 | 0/0(+0)；0/0(+0) | 24/14(+10)；36/3(+33) | 0/0(+0)；0/0(+0) |
| weak_pass | 955 | 40 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 0.851776 / 6.527635 | 317/282 | 8/97 | 8/8 | 86/89 | 1/0(+1)；0/0(+0) | 24/13(+11)；36/3(+33) | 1/0(+1)；0/0(+0) |
| weak_pass | 955 | 40 | head_weighted_body_center | CALIBRATED | 1.096427 | 306/273 | 5/88 | 5/5 | 79/83 | 0/6(-6)；0/6(-6) | 23/19(+4)；30/3(+27) | 1/11(-10)；0/9(-9) |
| weak_pass | 956 | 20 | center_single | CALIBRATED | 2.459946 | 289/234 | 0/59 | 0/0 | 74/60 | 0/0(+0)；0/0(+0) | 15/16(-1)；13/9(+4) | 0/0(+0)；0/0(+0) |
| weak_pass | 956 | 20 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 2.459946 / 10.502448 | 289/234 | 0/60 | 0/0 | 74/60 | 0/0(+0)；0/0(+0) | 15/16(-1)；13/9(+4) | 0/0(+0)；0/0(+0) |
| weak_pass | 956 | 20 | head_weighted_body_center | CALIBRATED | 2.753832 | 279/229 | 0/54 | 0/0 | 71/59 | 2/5(-3)；0/1(-1) | 14/18(-4)；13/10(+3) | 2/12(-10)；0/5(-5) |
| weak_pass | 956 | 30 | center_single | CALIBRATED | 1.919977 | 298/249 | 1/77 | 1/1 | 79/69 | 0/0(+0)；0/0(+0) | 19/15(+4)；19/6(+13) | 0/0(+0)；0/0(+0) |
| weak_pass | 956 | 30 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 1.919977 / 9.311522 | 298/249 | 1/78 | 1/1 | 79/69 | 0/0(+0)；0/0(+0) | 19/15(+4)；19/6(+13) | 0/0(+0)；0/0(+0) |
| weak_pass | 956 | 30 | head_weighted_body_center | CALIBRATED | 2.124835 | 296/241 | 1/74 | 1/1 | 79/64 | 1/1(+0)；0/5(-5) | 20/16(+4)；15/7(+8) | 1/3(-2)；0/8(-8) |
| weak_pass | 956 | 40 | center_single | CALIBRATED | 1.377748 | 309/264 | 5/97 | 5/5 | 86/75 | 0/0(+0)；0/0(+0) | 22/11(+11)；25/6(+19) | 0/0(+0)；0/0(+0) |
| weak_pass | 956 | 40 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 1.377748 / 8.098515 | 309/264 | 5/98 | 5/5 | 86/75 | 0/0(+0)；0/0(+0) | 22/11(+11)；25/6(+19) | 0/0(+0)；0/0(+0) |
| weak_pass | 956 | 40 | head_weighted_body_center | CALIBRATED | 1.554868 | 305/257 | 5/92 | 6/5 | 84/72 | 1/3(-2)；0/3(-3) | 23/14(+9)；22/6(+16) | 2/6(-4)；0/7(-7) |
| weak_pass | 957 | 20 | center_single | CALIBRATED | 3.198637 | 264/233 | 0/55 | 0/0 | 66/60 | 0/0(+0)；0/0(+0) | 14/23(-9)；13/9(+4) | 0/0(+0)；0/0(+0) |
| weak_pass | 957 | 20 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 3.198637 / 11.381917 | 264/233 | 0/55 | 0/0 | 66/60 | 0/0(+0)；0/0(+0) | 14/23(-9)；13/9(+4) | 0/0(+0)；0/0(+0) |
| weak_pass | 957 | 20 | head_weighted_body_center | CALIBRATED | 3.321946 | 261/228 | 0/49 | 0/0 | 64/59 | 1/3(-2)；0/1(-1) | 14/25(-11)；12/9(+3) | 2/5(-3)；0/5(-5) |
| weak_pass | 957 | 30 | center_single | CALIBRATED | 2.022180 | 286/261 | 0/75 | 0/0 | 76/78 | 0/0(+0)；0/0(+0) | 19/18(+1)；28/6(+22) | 0/0(+0)；0/0(+0) |
| weak_pass | 957 | 30 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 2.022180 / 8.094853 | 286/261 | 0/75 | 0/0 | 76/78 | 0/0(+0)；0/0(+0) | 19/18(+1)；28/6(+22) | 0/0(+0)；0/0(+0) |
| weak_pass | 957 | 30 | head_weighted_body_center | CALIBRATED | 2.047352 | 287/260 | 0/72 | 0/0 | 76/78 | 2/2(+0)；0/0(+0) | 21/20(+1)；28/6(+22) | 3/2(+1)；0/1(-1) |
| weak_pass | 957 | 40 | center_single | CALIBRATED | 1.384948 | 307/279 | 2/93 | 2/2 | 83/85 | 0/0(+0)；0/0(+0) | 24/16(+8)；34/5(+29) | 0/0(+0)；0/0(+0) |
| weak_pass | 957 | 40 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 1.384948 / 7.698571 | 307/279 | 2/95 | 2/2 | 83/85 | 0/0(+0)；0/0(+0) | 24/16(+8)；34/5(+29) | 0/0(+0)；0/0(+0) |
| weak_pass | 957 | 40 | head_weighted_body_center | CALIBRATED | 1.406277 | 307/279 | 2/94 | 2/2 | 80/85 | 1/4(-3)；0/0(+0) | 22/17(+5)；34/5(+29) | 6/6(+0)；0/0(+0) |

### yaw_plus3

| arm | seed | 点% | policy | 状态 | θ（OR为center/off） | H/B各384 | clearClip/passClip | clearSlot/段 | 1cmH/B各128 | 1cm vscenter H；B救/损(净) | 1cm vsM3 H；B救/损(净) | 整体vscenter H；B救/损(净) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 955 | 20 | center_single | CALIBRATED | 4.444754 | 268/211 | 87/96 | 182/105 | 70/58 | 0/0(+0)；0/0(+0) | 12/10(+2)；13/26(-13) | 0/0(+0)；0/0(+0) |
| control | 955 | 20 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 4.444754 / 11.726794 | 279/234 | 128/111 | 555/153 | 79/65 | 9/0(+9)；7/0(+7) | 13/2(+11)；13/19(-6) | 11/0(+11)；23/0(+23) |
| control | 955 | 20 | head_weighted_body_center | CALIBRATED | 5.268620 | 253/186 | 64/84 | 117/70 | 68/44 | 4/6(-2)；0/14(-14) | 9/9(+0)；11/38(-27) | 6/21(-15)；0/25(-25) |
| control | 955 | 30 | center_single | CALIBRATED | 2.590881 | 298/262 | 137/120 | 650/190 | 79/78 | 0/0(+0)；0/0(+0) | 14/3(+11)；21/14(+7) | 0/0(+0)；0/0(+0) |
| control | 955 | 30 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 2.590881 / 10.096044 | 300/269 | 139/120 | 904/179 | 81/82 | 2/0(+2)；4/0(+4) | 14/1(+13)；22/11(+11) | 2/0(+2)；7/0(+7) |
| control | 955 | 30 | head_weighted_body_center | CALIBRATED | 2.989800 | 293/254 | 132/119 | 695/168 | 79/75 | 1/1(+0)；0/3(-3) | 13/2(+11)；20/16(+4) | 1/6(-5)；0/8(-8) |
| control | 955 | 40 | center_single | CALIBRATED | 1.874375 | 307/272 | 142/132 | 789/205 | 80/83 | 0/0(+0)；0/0(+0) | 14/2(+12)；23/11(+12) | 0/0(+0)；0/0(+0) |
| control | 955 | 40 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 1.874375 / 10.096044 | 308/276 | 143/132 | 992/192 | 81/85 | 1/0(+1)；2/0(+2) | 14/1(+13)；24/10(+14) | 1/0(+1)；4/0(+4) |
| control | 955 | 40 | head_weighted_body_center | CALIBRATED | 2.234261 | 302/266 | 138/125 | 888/172 | 80/81 | 1/1(+0)；0/2(-2) | 13/1(+12)；22/12(+10) | 1/6(-5)；0/6(-6) |
| control | 956 | 20 | center_single | CALIBRATED | 6.057990 | 214/122 | 7/63 | 7/7 | 56/22 | 0/0(+0)；0/0(+0) | 4/16(-12)；1/50(-49) | 0/0(+0)；0/0(+0) |
| control | 956 | 20 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 6.057990 / 14.243341 | 215/122 | 10/63 | 10/10 | 57/22 | 1/0(+1)；0/0(+0) | 4/15(-11)；1/50(-49) | 1/0(+1)；0/0(+0) |
| control | 956 | 20 | head_weighted_body_center | CALIBRATED | 6.092933 | 219/122 | 10/69 | 11/10 | 58/22 | 3/1(+2)；0/0(+0) | 3/13(-10)；1/50(-49) | 12/7(+5)；0/0(+0) |
| control | 956 | 30 | center_single | CALIBRATED | 4.371076 | 265/186 | 59/83 | 104/75 | 68/47 | 0/0(+0)；0/0(+0) | 9/9(+0)；12/36(-24) | 0/0(+0)；0/0(+0) |
| control | 956 | 30 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 4.371076 / 11.669246 | 272/197 | 109/107 | 255/141 | 73/51 | 5/0(+5)；4/0(+4) | 10/5(+5)；12/32(-20) | 7/0(+7)；11/0(+11) |
| control | 956 | 30 | head_weighted_body_center | CALIBRATED | 4.383284 | 276/186 | 82/90 | 177/111 | 74/47 | 6/0(+6)；0/0(+0) | 11/5(+6)；12/36(-24) | 13/2(+11)；0/0(+0) |
| control | 956 | 40 | center_single | CALIBRATED | 2.736755 | 293/242 | 127/122 | 450/187 | 77/69 | 0/0(+0)；0/0(+0) | 12/3(+9)；23/25(-2) | 0/0(+0)；0/0(+0) |
| control | 956 | 40 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 2.736755 / 9.317640 | 296/264 | 146/124 | 799/201 | 80/79 | 3/0(+3)；10/0(+10) | 13/1(+12)；24/16(+8) | 3/0(+3)；22/0(+22) |
| control | 956 | 40 | head_weighted_body_center | CALIBRATED | 2.913773 | 295/240 | 138/122 | 575/199 | 80/68 | 3/0(+3)；0/1(-1) | 13/1(+12)；22/25(-3) | 3/1(+2)；0/2(-2) |
| control | 957 | 20 | center_single | CALIBRATED | 6.693934 | 262/169 | 82/85 | 166/91 | 66/41 | 0/0(+0)；0/0(+0) | 7/9(-2)；5/35(-30) | 0/0(+0)；0/0(+0) |
| control | 957 | 20 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 6.693934 / 15.271625 | 278/205 | 129/112 | 767/188 | 79/52 | 13/0(+13)；11/0(+11) | 12/1(+11)；6/25(-19) | 16/0(+16)；36/0(+36) |
| control | 957 | 20 | head_weighted_body_center | CALIBRATED | 7.031885 | 254/160 | 104/99 | 229/118 | 70/35 | 5/1(+4)；0/6(-6) | 6/4(+2)；4/40(-36) | 8/16(-8)；0/9(-9) |
| control | 957 | 30 | center_single | CALIBRATED | 5.054212 | 282/226 | 123/108 | 400/157 | 76/62 | 0/0(+0)；0/0(+0) | 10/2(+8)；12/21(-9) | 0/0(+0)；0/0(+0) |
| control | 957 | 30 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 5.054212 / 13.946710 | 286/244 | 131/114 | 881/143 | 80/74 | 4/0(+4)；12/0(+12) | 13/1(+12)；18/15(+3) | 4/0(+4)；18/0(+18) |
| control | 957 | 30 | head_weighted_body_center | CALIBRATED | 4.811739 | 285/232 | 131/115 | 687/173 | 79/65 | 3/0(+3)；3/0(+3) | 12/1(+11)；14/20(-6) | 4/1(+3)；6/0(+6) |
| control | 957 | 40 | center_single | CALIBRATED | 4.000479 | 292/250 | 133/121 | 619/178 | 78/73 | 0/0(+0)；0/0(+0) | 11/1(+10)；16/14(+2) | 0/0(+0)；0/0(+0) |
| control | 957 | 40 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 4.000479 / 13.946710 | 294/261 | 135/122 | 909/153 | 80/80 | 2/0(+2)；7/0(+7) | 13/1(+12)；21/12(+9) | 2/0(+2)；11/0(+11) |
| control | 957 | 40 | head_weighted_body_center | CALIBRATED | 3.946529 | 293/251 | 137/121 | 859/149 | 80/74 | 2/0(+2)；1/0(+1) | 13/1(+12)；16/13(+3) | 2/1(+1)；1/0(+1) |
| weak_pass | 955 | 20 | center_single | CALIBRATED | 2.189399 | 272/236 | 91/95 | 213/118 | 64/69 | 0/0(+0)；0/0(+0) | 7/11(-4)；17/19(-2) | 0/0(+0)；0/0(+0) |
| weak_pass | 955 | 20 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 2.189399 / 8.811336 | 284/256 | 130/115 | 472/165 | 74/76 | 10/0(+10)；7/0(+7) | 8/2(+6)；17/12(+5) | 12/0(+12)；20/0(+20) |
| weak_pass | 955 | 20 | head_weighted_body_center | CALIBRATED | 2.225964 | 276/235 | 117/109 | 342/141 | 73/68 | 9/0(+9)；0/1(-1) | 8/3(+5)；17/20(-3) | 11/7(+4)；0/1(-1) |
| weak_pass | 955 | 30 | center_single | CALIBRATED | 1.339579 | 294/264 | 134/123 | 513/199 | 73/79 | 0/0(+0)；0/0(+0) | 12/7(+5)；22/14(+8) | 0/0(+0)；0/0(+0) |
| weak_pass | 955 | 30 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 1.339579 / 6.983053 | 301/274 | 143/130 | 857/196 | 80/84 | 7/0(+7)；5/0(+5) | 13/1(+12)；22/9(+13) | 7/0(+7)；10/0(+10) |
| weak_pass | 955 | 30 | head_weighted_body_center | CALIBRATED | 1.608958 | 293/254 | 131/126 | 585/181 | 76/74 | 4/1(+3)；0/5(-5) | 11/3(+8)；19/16(+3) | 4/5(-1)；0/10(-10) |
| weak_pass | 955 | 40 | center_single | CALIBRATED | 0.851776 | 306/278 | 148/136 | 739/233 | 78/84 | 0/0(+0)；0/0(+0) | 14/4(+10)；23/10(+13) | 0/0(+0)；0/0(+0) |
| weak_pass | 955 | 40 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 0.851776 / 6.527635 | 309/282 | 155/136 | 1009/215 | 81/86 | 3/0(+3)；2/0(+2) | 14/1(+13)；23/8(+15) | 3/0(+3)；4/0(+4) |
| weak_pass | 955 | 40 | head_weighted_body_center | CALIBRATED | 1.096427 | 302/275 | 139/135 | 823/195 | 80/84 | 3/1(+2)；0/0(+0) | 13/1(+12)；23/10(+13) | 3/7(-4)；0/3(-3) |
| weak_pass | 956 | 20 | center_single | CALIBRATED | 2.459946 | 254/209 | 66/87 | 111/71 | 60/52 | 0/0(+0)；0/0(+0) | 0/8(-8)；11/30(-19) | 0/0(+0)；0/0(+0) |
| weak_pass | 956 | 20 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 2.459946 / 10.502448 | 254/209 | 66/87 | 111/71 | 60/52 | 0/0(+0)；0/0(+0) | 0/8(-8)；11/30(-19) | 0/0(+0)；0/0(+0) |
| weak_pass | 956 | 20 | head_weighted_body_center | CALIBRATED | 2.753832 | 252/196 | 70/90 | 124/75 | 61/48 | 2/1(+1)；0/4(-4) | 0/7(-7)；10/33(-23) | 4/6(-2)；0/13(-13) |
| weak_pass | 956 | 30 | center_single | CALIBRATED | 1.919977 | 268/234 | 100/104 | 214/127 | 63/64 | 0/0(+0)；0/0(+0) | 0/5(-5)；18/25(-7) | 0/0(+0)；0/0(+0) |
| weak_pass | 956 | 30 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 1.919977 / 9.311522 | 268/240 | 104/107 | 228/134 | 63/67 | 0/0(+0)；3/0(+3) | 0/5(-5)；18/22(-4) | 0/0(+0)；6/0(+6) |
| weak_pass | 956 | 30 | head_weighted_body_center | CALIBRATED | 2.124835 | 274/225 | 105/110 | 265/126 | 66/60 | 4/1(+3)；0/4(-4) | 2/4(-2)；16/27(-11) | 10/4(+6)；0/9(-9) |
| weak_pass | 956 | 40 | center_single | CALIBRATED | 1.377748 | 289/258 | 120/122 | 397/186 | 68/78 | 0/0(+0)；0/0(+0) | 3/3(+0)；23/16(+7) | 0/0(+0)；0/0(+0) |
| weak_pass | 956 | 40 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 1.377748 / 8.098515 | 290/266 | 133/124 | 457/201 | 68/82 | 0/0(+0)；4/0(+4) | 3/3(+0)；23/12(+11) | 1/0(+1)；8/0(+8) |
| weak_pass | 956 | 40 | head_weighted_body_center | CALIBRATED | 1.554868 | 287/253 | 131/125 | 474/184 | 71/76 | 4/1(+3)；0/2(-2) | 4/1(+3)；22/17(+5) | 5/7(-2)；0/5(-5) |
| weak_pass | 957 | 20 | center_single | CALIBRATED | 3.198637 | 259/202 | 34/79 | 53/40 | 61/49 | 0/0(+0)；0/0(+0) | 3/10(-7)；10/32(-22) | 0/0(+0)；0/0(+0) |
| weak_pass | 957 | 20 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 3.198637 / 11.381917 | 272/215 | 126/111 | 317/134 | 70/57 | 9/0(+9)；8/0(+8) | 4/2(+2)；10/24(-14) | 13/0(+13)；13/0(+13) |
| weak_pass | 957 | 20 | head_weighted_body_center | CALIBRATED | 3.321946 | 258/192 | 99/100 | 204/106 | 66/45 | 6/1(+5)；0/4(-4) | 2/4(-2)；8/34(-26) | 10/11(-1)；0/10(-10) |
| weak_pass | 957 | 30 | center_single | CALIBRATED | 2.022180 | 285/248 | 109/106 | 316/148 | 69/72 | 0/0(+0)；0/0(+0) | 7/6(+1)；16/15(+1) | 0/0(+0)；0/0(+0) |
| weak_pass | 957 | 30 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 2.022180 / 8.094853 | 293/260 | 138/121 | 867/229 | 77/77 | 8/0(+8)；5/0(+5) | 10/1(+9)；17/11(+6) | 8/0(+8)；12/0(+12) |
| weak_pass | 957 | 30 | head_weighted_body_center | CALIBRATED | 2.047352 | 288/247 | 132/120 | 597/171 | 75/72 | 6/0(+6)；0/0(+0) | 8/1(+7)；16/15(+1) | 6/3(+3)；0/1(-1) |
| weak_pass | 957 | 40 | center_single | CALIBRATED | 1.384948 | 299/259 | 140/122 | 582/199 | 75/76 | 0/0(+0)；0/0(+0) | 10/3(+7)；17/12(+5) | 0/0(+0)；0/0(+0) |
| weak_pass | 957 | 40 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 1.384948 / 7.698571 | 304/268 | 150/128 | 981/229 | 80/80 | 5/0(+5)；4/0(+4) | 13/1(+12)；19/10(+9) | 5/0(+5)；9/0(+9) |
| weak_pass | 957 | 40 | head_weighted_body_center | CALIBRATED | 1.406277 | 300/259 | 144/127 | 877/191 | 79/76 | 4/0(+4)；0/0(+0) | 12/1(+11)；17/12(+5) | 4/3(+1)；0/0(+0) |

## 本轮54 cal记录

| arm | seed | 点% | policy | 状态 | centerθ（OR） | θ/offθ | calH/B各384 | cal clearClip/passClip/slot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 955 | 20 | center_single | CALIBRATED | — | 4.444754 | 257/241 | 0/51/0 |
| control | 955 | 20 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 4.444754 | 11.726794 | 257/241 | 0/51/0 |
| control | 955 | 20 | head_weighted_body_center | CALIBRATED | — | 5.268620 | 249/220 | 0/51/0 |
| control | 955 | 30 | center_single | CALIBRATED | — | 2.590881 | 293/275 | 1/76/2 |
| control | 955 | 30 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 2.590881 | 10.096044 | 293/275 | 1/76/2 |
| control | 955 | 30 | head_weighted_body_center | CALIBRATED | — | 2.989800 | 290/272 | 5/76/5 |
| control | 955 | 40 | center_single | CALIBRATED | — | 1.874375 | 314/284 | 10/102/13 |
| control | 955 | 40 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 1.874375 | 10.096044 | 314/284 | 10/102/13 |
| control | 955 | 40 | head_weighted_body_center | CALIBRATED | — | 2.234261 | 315/280 | 22/102/29 |
| control | 956 | 20 | center_single | CALIBRATED | — | 6.057990 | 256/182 | 0/51/0 |
| control | 956 | 20 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 6.057990 | 14.243341 | 256/182 | 0/51/0 |
| control | 956 | 20 | head_weighted_body_center | CALIBRATED | — | 6.092933 | 255/181 | 0/51/0 |
| control | 956 | 30 | center_single | CALIBRATED | — | 4.371076 | 277/213 | 1/76/1 |
| control | 956 | 30 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 4.371076 | 11.669246 | 277/213 | 1/76/1 |
| control | 956 | 30 | head_weighted_body_center | CALIBRATED | — | 4.383284 | 281/213 | 1/76/1 |
| control | 956 | 40 | center_single | CALIBRATED | — | 2.736755 | 303/260 | 2/102/3 |
| control | 956 | 40 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 2.736755 | 9.317640 | 303/260 | 2/102/3 |
| control | 956 | 40 | head_weighted_body_center | CALIBRATED | — | 2.913773 | 302/257 | 2/102/3 |
| control | 957 | 20 | center_single | CALIBRATED | — | 6.693934 | 262/213 | 0/51/0 |
| control | 957 | 20 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 6.693934 | 15.271625 | 262/213 | 0/51/0 |
| control | 957 | 20 | head_weighted_body_center | CALIBRATED | — | 7.031885 | 254/205 | 0/51/0 |
| control | 957 | 30 | center_single | CALIBRATED | — | 5.054212 | 272/252 | 0/76/0 |
| control | 957 | 30 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 5.054212 | 13.946710 | 272/252 | 0/76/0 |
| control | 957 | 30 | head_weighted_body_center | CALIBRATED | — | 4.811739 | 286/258 | 0/76/0 |
| control | 957 | 40 | center_single | CALIBRATED | — | 4.000479 | 286/274 | 0/102/0 |
| control | 957 | 40 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 4.000479 | 13.946710 | 286/274 | 0/102/0 |
| control | 957 | 40 | head_weighted_body_center | CALIBRATED | — | 3.946529 | 295/275 | 0/102/0 |
| weak_pass | 955 | 20 | center_single | CALIBRATED | — | 2.189399 | 273/262 | 1/51/2 |
| weak_pass | 955 | 20 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 2.189399 | 8.811336 | 273/262 | 1/51/2 |
| weak_pass | 955 | 20 | head_weighted_body_center | CALIBRATED | — | 2.225964 | 272/258 | 1/51/2 |
| weak_pass | 955 | 30 | center_single | CALIBRATED | — | 1.339579 | 298/286 | 7/76/10 |
| weak_pass | 955 | 30 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 1.339579 | 6.983053 | 298/286 | 7/76/10 |
| weak_pass | 955 | 30 | head_weighted_body_center | CALIBRATED | — | 1.608958 | 295/278 | 11/76/14 |
| weak_pass | 955 | 40 | center_single | CALIBRATED | — | 0.851776 | 315/293 | 16/102/28 |
| weak_pass | 955 | 40 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 0.851776 | 6.527635 | 315/293 | 16/102/28 |
| weak_pass | 955 | 40 | head_weighted_body_center | CALIBRATED | — | 1.096427 | 312/289 | 27/102/43 |
| weak_pass | 956 | 20 | center_single | CALIBRATED | — | 2.459946 | 283/236 | 2/51/3 |
| weak_pass | 956 | 20 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 2.459946 | 10.502448 | 283/236 | 2/51/3 |
| weak_pass | 956 | 20 | head_weighted_body_center | CALIBRATED | — | 2.753832 | 275/230 | 1/51/2 |
| weak_pass | 956 | 30 | center_single | CALIBRATED | — | 1.919977 | 298/257 | 5/76/6 |
| weak_pass | 956 | 30 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 1.919977 | 9.311522 | 298/257 | 5/76/6 |
| weak_pass | 956 | 30 | head_weighted_body_center | CALIBRATED | — | 2.124835 | 291/250 | 3/76/5 |
| weak_pass | 956 | 40 | center_single | CALIBRATED | — | 1.377748 | 313/272 | 9/102/12 |
| weak_pass | 956 | 40 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 1.377748 | 8.098515 | 313/272 | 9/102/12 |
| weak_pass | 956 | 40 | head_weighted_body_center | CALIBRATED | — | 1.554868 | 307/266 | 14/102/18 |
| weak_pass | 957 | 20 | center_single | CALIBRATED | — | 3.198637 | 264/241 | 0/51/0 |
| weak_pass | 957 | 20 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 3.198637 | 11.381917 | 264/241 | 0/51/0 |
| weak_pass | 957 | 20 | head_weighted_body_center | CALIBRATED | — | 3.321946 | 257/236 | 0/51/0 |
| weak_pass | 957 | 30 | center_single | CALIBRATED | — | 2.022180 | 289/277 | 3/76/5 |
| weak_pass | 957 | 30 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 2.022180 | 8.094853 | 289/277 | 3/76/5 |
| weak_pass | 957 | 30 | head_weighted_body_center | CALIBRATED | — | 2.047352 | 294/275 | 2/76/4 |
| weak_pass | 957 | 40 | center_single | CALIBRATED | — | 1.384948 | 308/291 | 8/102/12 |
| weak_pass | 957 | 40 | anchored_zero_extra | CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS | 1.384948 | 7.698571 | 308/291 | 8/102/12 |
| weak_pass | 957 | 40 | head_weighted_body_center | CALIBRATED | — | 1.406277 | 315/290 | 14/102/21 |

## 上轮54阈值尾部与score／threshold分解

54个前驱中53个由pass clip约束阻断、1个由clear slot阻断；18个max阈值相对single全升，18个前驱均因pass clip受限。weighted有12个θ上升、6个下降。唯一clear阻断是weak955 weighted40：选中clear slot65、前驱66，pass clip均97低于预算102。本轮18个OR θoff的全局limiter为17个pass、1个clear。零增量OR比完整剩余预算更严格；本轮未运行budget-OR，不能由此说剩余预算不存在。

上轮single/max/weighted各18个ideal-cal阈值，共54；不在本轮重新改这些阈值。前驱是完整distinct score-grid紧邻的更低level，whole ties全部进入；下表保留被违反的clear slot/pass clip约束。query曝光的contact/clear/pass列不是joint成本单位。

| arm | seed | 聚合 | 点% | θ | θ−singleθ | 前驱θ | 选中slot/pass | 前驱slot/pass | 阻断约束 | selected level C/F/P query曝光 | 前驱level C/F/P query曝光 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 955 | single | 20 | 4.444754 | 0.000000 | 4.442563 | 0/51 | 0/52 | pass_clips | 0/0/1 | 0/0/1 |
| control | 955 | single | 30 | 2.590881 | 0.000000 | 2.585702 | 2/76 | 2/77 | pass_clips | 1/0/0 | 0/0/1 |
| control | 955 | single | 40 | 1.874375 | 0.000000 | 1.871938 | 13/102 | 13/103 | pass_clips | 1/0/0 | 0/0/1 |
| control | 955 | max | 20 | 10.962576 | 6.517823 | 10.961953 | 0/51 | 0/52 | pass_clips | 1/0/0 | 0/0/1 |
| control | 955 | max | 30 | 8.737247 | 6.146365 | 8.736451 | 1/76 | 1/77 | pass_clips | 1/0/0 | 0/0/1 |
| control | 955 | max | 40 | 7.320071 | 5.445696 | 7.318205 | 15/102 | 15/103 | pass_clips | 1/0/0 | 0/0/1 |
| control | 955 | weighted | 20 | 5.268620 | 0.823866 | 5.267897 | 0/51 | 0/52 | pass_clips | 1/0/0 | 0/0/1 |
| control | 955 | weighted | 30 | 3.219801 | 0.628920 | 3.219376 | 11/76 | 11/77 | pass_clips | 1/0/0 | 0/0/1 |
| control | 955 | weighted | 40 | 2.365273 | 0.490898 | 2.364952 | 51/102 | 51/103 | pass_clips | 0/0/1 | 0/0/1 |
| control | 956 | single | 20 | 6.057990 | 0.000000 | 6.055250 | 0/51 | 0/52 | pass_clips | 1/0/0 | 0/0/1 |
| control | 956 | single | 30 | 4.371076 | 0.000000 | 4.367076 | 1/76 | 1/77 | pass_clips | 1/0/0 | 0/0/1 |
| control | 956 | single | 40 | 2.736755 | 0.000000 | 2.734305 | 3/102 | 3/103 | pass_clips | 0/0/1 | 0/0/1 |
| control | 956 | max | 20 | 12.155198 | 6.097208 | 12.151700 | 0/51 | 0/52 | pass_clips | 0/0/1 | 0/0/1 |
| control | 956 | max | 30 | 9.952164 | 5.581087 | 9.947417 | 0/76 | 0/77 | pass_clips | 1/0/0 | 0/0/1 |
| control | 956 | max | 40 | 7.000095 | 4.263340 | 6.998688 | 2/102 | 2/103 | pass_clips | 1/0/0 | 0/0/1 |
| control | 956 | weighted | 20 | 6.018121 | -0.039869 | 6.016894 | 0/51 | 0/52 | pass_clips | 1/0/0 | 0/0/1 |
| control | 956 | weighted | 30 | 4.322787 | -0.048289 | 4.320337 | 1/76 | 1/77 | pass_clips | 1/0/0 | 0/0/1 |
| control | 956 | weighted | 40 | 3.052595 | 0.315840 | 3.050588 | 3/102 | 3/103 | pass_clips | 1/0/0 | 0/0/1 |
| control | 957 | single | 20 | 6.693934 | 0.000000 | 6.690738 | 0/51 | 0/52 | pass_clips | 1/0/0 | 0/0/1 |
| control | 957 | single | 30 | 5.054212 | 0.000000 | 5.052196 | 0/76 | 0/77 | pass_clips | 0/0/1 | 0/0/1 |
| control | 957 | single | 40 | 4.000479 | 0.000000 | 4.000239 | 0/102 | 0/103 | pass_clips | 0/0/1 | 0/0/1 |
| control | 957 | max | 20 | 14.261351 | 7.567417 | 14.249602 | 0/51 | 0/52 | pass_clips | 1/0/0 | 0/0/1 |
| control | 957 | max | 30 | 11.648523 | 6.594311 | 11.647056 | 1/76 | 1/77 | pass_clips | 1/0/0 | 0/0/1 |
| control | 957 | max | 40 | 10.479319 | 6.478840 | 10.473372 | 7/102 | 7/103 | pass_clips | 1/0/0 | 0/0/1 |
| control | 957 | weighted | 20 | 6.929418 | 0.235484 | 6.927847 | 0/51 | 0/52 | pass_clips | 1/0/0 | 0/0/1 |
| control | 957 | weighted | 30 | 4.813806 | -0.240406 | 4.811532 | 0/76 | 0/77 | pass_clips | 1/0/0 | 0/0/1 |
| control | 957 | weighted | 40 | 3.804759 | -0.195720 | 3.803899 | 2/102 | 2/103 | pass_clips | 1/0/0 | 0/0/1 |
| weak_pass | 955 | single | 20 | 2.189399 | 0.000000 | 2.188590 | 2/51 | 2/52 | pass_clips | 1/0/0 | 0/0/1 |
| weak_pass | 955 | single | 30 | 1.339579 | 0.000000 | 1.338981 | 10/76 | 10/77 | pass_clips | 0/0/1 | 0/0/1 |
| weak_pass | 955 | single | 40 | 0.851776 | 0.000000 | 0.851366 | 28/102 | 28/103 | pass_clips | 1/0/0 | 0/0/1 |
| weak_pass | 955 | max | 20 | 7.095061 | 4.905662 | 7.092150 | 0/51 | 0/52 | pass_clips | 0/0/1 | 0/0/1 |
| weak_pass | 955 | max | 30 | 5.706940 | 4.367361 | 5.705307 | 3/76 | 3/77 | pass_clips | 1/0/0 | 0/0/1 |
| weak_pass | 955 | max | 40 | 4.800785 | 3.949009 | 4.800686 | 18/102 | 18/103 | pass_clips | 0/1/0 | 0/0/1 |
| weak_pass | 955 | weighted | 20 | 2.478829 | 0.289430 | 2.478026 | 5/51 | 5/52 | pass_clips | 0/0/1 | 0/0/1 |
| weak_pass | 955 | weighted | 30 | 1.704555 | 0.364976 | 1.696693 | 30/76 | 30/77 | pass_clips | 1/0/0 | 0/0/1 |
| weak_pass | 955 | weighted | 40 | 1.227904 | 0.376128 | 1.227382 | 65/97 | 66/97 | clear_slots | 0/0/1 | 0/1/0 |
| weak_pass | 956 | single | 20 | 2.459946 | 0.000000 | 2.459724 | 3/51 | 3/52 | pass_clips | 0/0/1 | 0/0/1 |
| weak_pass | 956 | single | 30 | 1.919977 | 0.000000 | 1.916497 | 6/76 | 6/77 | pass_clips | 0/0/1 | 0/0/1 |
| weak_pass | 956 | single | 40 | 1.377748 | 0.000000 | 1.377364 | 12/102 | 12/103 | pass_clips | 1/0/0 | 0/0/1 |
| weak_pass | 956 | max | 20 | 8.494318 | 6.034372 | 8.485382 | 0/51 | 0/52 | pass_clips | 1/0/0 | 0/0/1 |
| weak_pass | 956 | max | 30 | 6.872428 | 4.952451 | 6.872286 | 0/76 | 0/77 | pass_clips | 1/0/0 | 0/0/1 |
| weak_pass | 956 | max | 40 | 4.862711 | 3.484963 | 4.861478 | 2/102 | 2/103 | pass_clips | 0/1/0 | 0/0/1 |
| weak_pass | 956 | weighted | 20 | 2.868013 | 0.408067 | 2.866130 | 2/51 | 2/52 | pass_clips | 0/0/1 | 0/0/1 |
| weak_pass | 956 | weighted | 30 | 2.159290 | 0.239313 | 2.158587 | 6/76 | 6/77 | pass_clips | 1/0/0 | 0/0/1 |
| weak_pass | 956 | weighted | 40 | 1.514323 | 0.136575 | 1.514033 | 23/102 | 23/103 | pass_clips | 1/0/0 | 0/0/1 |
| weak_pass | 957 | single | 20 | 3.198637 | 0.000000 | 3.195897 | 0/51 | 0/52 | pass_clips | 1/0/0 | 0/0/1 |
| weak_pass | 957 | single | 30 | 2.022180 | 0.000000 | 2.020756 | 5/76 | 5/77 | pass_clips | 1/0/0 | 0/0/1 |
| weak_pass | 957 | single | 40 | 1.384948 | 0.000000 | 1.381787 | 12/102 | 12/103 | pass_clips | 1/0/0 | 0/0/1 |
| weak_pass | 957 | max | 20 | 8.324782 | 5.126145 | 8.318885 | 0/51 | 0/52 | pass_clips | 0/0/1 | 0/0/1 |
| weak_pass | 957 | max | 30 | 6.972768 | 4.950587 | 6.970092 | 0/76 | 0/77 | pass_clips | 1/0/0 | 0/0/1 |
| weak_pass | 957 | max | 40 | 5.790536 | 4.405587 | 5.790513 | 1/102 | 1/103 | pass_clips | 1/0/0 | 0/0/1 |
| weak_pass | 957 | weighted | 20 | 3.197102 | -0.001535 | 3.195872 | 0/51 | 0/52 | pass_clips | 1/0/0 | 0/0/1 |
| weak_pass | 957 | weighted | 30 | 1.897053 | -0.125127 | 1.896009 | 9/76 | 9/77 | pass_clips | 0/0/1 | 0/0/1 |
| weak_pass | 957 | weighted | 40 | 1.465925 | 0.080977 | 1.465046 | 22/102 | 22/103 | pass_clips | 1/0/0 | 0/0/1 |

max同θ平滑分数不低于center，全部36个max路径同θquery报警损失为0。换成自身cal θ后可失检，说明本配方的阈值路径需要单列；不能把max自身θ下的损失解释为同θ score降低。weighted没有单调保证：两个路径均保留——center@singleθ→weighted@singleθ→weighted@ownθ；第二段既可增也可减，不能只归因一个阶段。

真正blocking的predecessor-added单位共有54次threshold-record×unit occurrence：pass clip中segmented_backwall21、side_gate_posts32，clear joint-slot中side_gate_posts1。这是记录次数而非独立场景或光子；背景名称不证明反射来源。方向贡献逐个链接至last5 raw lower/center/upper，未由argmax推定正确方向。weighted weak956/30% 1cm分解（各128）补充具体例：ideal H79→79→79、B69→63→62；+3° H63→67→66、B64→74→69。顺序为center@singleθ→weighted@singleθ→weighted@ownθ，说明score和θ两步均有实际变化，不能把第二步成本/损失归到纯score或反之。

完整72条score／threshold路径、432个阶段×分层summary与55296条contact ledger保留。下面40% weak例只展示已声明点的分解；前两高度及时数各/384，救损以同路径上一步为参照，各高度/384。完整1cm分层及所有点见CSV，均不按loss/miss筛掉。

| seed | 聚合 | 分支 | 阶段 | H/B及时 | clearClip/passClip | clearSlot | vs上一步H；B救/损 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 955 | max | ideal | center_at_single_theta | 316/282 | 8/93 | 8 | 0/0；0/0 |
| 955 | max | ideal | aggregator_at_single_theta | 358/318 | 182/218 | 821 | 42/0；36/0 |
| 955 | max | ideal | aggregator_at_own_theta | 270/218 | 2/88 | 2 | 0/88；0/100 |
| 955 | max | yaw_plus3 | center_at_single_theta | 306/278 | 148/136 | 739 | 0/0；0/0 |
| 955 | max | yaw_plus3 | aggregator_at_single_theta | 341/325 | 257/202 | 1729 | 35/0；47/0 |
| 955 | max | yaw_plus3 | aggregator_at_own_theta | 283/256 | 146/113 | 1016 | 0/58；0/69 |
| 955 | weighted | ideal | center_at_single_theta | 316/282 | 8/93 | 8 | 0/0；0/0 |
| 955 | weighted | ideal | aggregator_at_single_theta | 315/279 | 7/94 | 7 | 5/6；2/5 |
| 955 | weighted | ideal | aggregator_at_own_theta | 301/270 | 2/81 | 2 | 0/14；0/9 |
| 955 | weighted | yaw_plus3 | center_at_single_theta | 306/278 | 148/136 | 739 | 0/0；0/0 |
| 955 | weighted | yaw_plus3 | aggregator_at_single_theta | 307/282 | 149/132 | 985 | 5/4；8/4 |
| 955 | weighted | yaw_plus3 | aggregator_at_own_theta | 301/275 | 139/130 | 813 | 0/6；0/7 |
| 956 | max | ideal | center_at_single_theta | 309/264 | 5/97 | 5 | 0/0；0/0 |
| 956 | max | ideal | aggregator_at_single_theta | 354/302 | 143/198 | 438 | 45/0；38/0 |
| 956 | max | ideal | aggregator_at_own_theta | 288/201 | 1/105 | 1 | 0/66；0/101 |
| 956 | max | yaw_plus3 | center_at_single_theta | 289/258 | 120/122 | 397 | 0/0；0/0 |
| 956 | max | yaw_plus3 | aggregator_at_single_theta | 342/322 | 210/189 | 1603 | 53/0；64/0 |
| 956 | max | yaw_plus3 | aggregator_at_own_theta | 290/255 | 148/117 | 1001 | 0/52；0/67 |
| 956 | weighted | ideal | center_at_single_theta | 309/264 | 5/97 | 5 | 0/0；0/0 |
| 956 | weighted | ideal | aggregator_at_single_theta | 307/251 | 5/103 | 7 | 3/5；3/16 |
| 956 | weighted | ideal | aggregator_at_own_theta | 306/248 | 3/94 | 4 | 0/1；0/3 |
| 956 | weighted | yaw_plus3 | center_at_single_theta | 289/258 | 120/122 | 397 | 0/0；0/0 |
| 956 | weighted | yaw_plus3 | aggregator_at_single_theta | 295/278 | 136/127 | 643 | 8/2；21/1 |
| 956 | weighted | yaw_plus3 | aggregator_at_own_theta | 290/276 | 135/127 | 594 | 0/5；0/2 |
| 957 | max | ideal | center_at_single_theta | 307/279 | 2/93 | 2 | 0/0；0/0 |
| 957 | max | ideal | aggregator_at_single_theta | 351/307 | 161/201 | 621 | 44/0；28/0 |
| 957 | max | ideal | aggregator_at_own_theta | 270/184 | 2/100 | 2 | 0/81；0/123 |
| 957 | max | yaw_plus3 | center_at_single_theta | 299/259 | 140/122 | 582 | 0/0；0/0 |
| 957 | max | yaw_plus3 | aggregator_at_single_theta | 337/319 | 218/193 | 1603 | 38/0；60/0 |
| 957 | max | yaw_plus3 | aggregator_at_own_theta | 283/234 | 146/113 | 991 | 0/54；0/85 |
| 957 | weighted | ideal | center_at_single_theta | 307/279 | 2/93 | 2 | 0/0；0/0 |
| 957 | weighted | ideal | aggregator_at_single_theta | 307/265 | 1/92 | 1 | 6/6；1/15 |
| 957 | weighted | ideal | aggregator_at_own_theta | 306/263 | 1/87 | 1 | 0/1；0/2 |
| 957 | weighted | yaw_plus3 | center_at_single_theta | 299/259 | 140/122 | 582 | 0/0；0/0 |
| 957 | weighted | yaw_plus3 | aggregator_at_single_theta | 301/276 | 145/125 | 933 | 5/3；17/0 |
| 957 | weighted | yaw_plus3 | aggregator_at_own_theta | 300/275 | 142/123 | 903 | 0/1；0/1 |

4548个active/tie尾部单位、4548个峰值记录、22336个历史帧贡献记录完整留存；unit_kind区分joint-clear slot与pass clip，active、selectedtie、predecessortie和predecessor_added分别标注。shape、背景族/实例、rho、side和每个last5 raw方向贡献可按单位链接，不把跨模型logit尺度或某个背景名称当光子来源因果。

[54阈值/前驱](../../../../artifacts.local/work/cnh-direction-anchor-dev-20261009/tails/tail_thresholds.csv) · [active与limiting tie元数据](../../../../artifacts.local/work/cnh-direction-anchor-dev-20261009/tails/tail_units.csv) · [峰值定位](../../../../artifacts.local/work/cnh-direction-anchor-dev-20261009/tails/tail_peak_frames.csv) · [last5 raw方向贡献](../../../../artifacts.local/work/cnh-direction-anchor-dev-20261009/tails/contributor_frames.csv)

[完整score/θ分解](../../../../artifacts.local/work/cnh-direction-anchor-dev-20261009/tails/decomposition_summary.csv) · [分解事件ledger](../../../../artifacts.local/work/cnh-direction-anchor-dev-20261009/tails/decomposition_event_ledger.csv) · [同θ单调检查](../../../../artifacts.local/work/cnh-direction-anchor-dev-20261009/tails/same_theta_checks.json)

[本轮原始政策metrics](../../../../artifacts.local/work/cnh-direction-anchor-dev-20261009/policies/metrics.json) · [本轮精确阈值](../../../../artifacts.local/work/cnh-direction-anchor-dev-20261009/policies/thresholds.json) · [全政策CSV](../../../../artifacts.local/work/cnh-direction-anchor-dev-20261009/policies/summary.csv) · [逐事件ledger](../../../../artifacts.local/work/cnh-direction-anchor-dev-20261009/policies/event_ledger.csv) · [报告108cell CSV](../../../../artifacts.local/work/cnh-direction-anchor-dev-20261009/report_policy_cells.csv)

[OR限制记录](../../../../artifacts.local/work/cnh-direction-anchor-dev-20261009/policies/calibration_tail_limits.csv) · [独立audit](../../../../artifacts.local/work/cnh-direction-anchor-dev-20261009/independent_audit.json)

## 范围与保留

tail前驱约束、limit ties的背景/方向描述说明阈值受哪些记录限制，不能当成墙面、纯目标或具体光子来源因果证明。max在同θ时逐帧/平滑单调，损失可能来自抬升θ；weighted不满足同θ单调，score变化与阈值变化必须分开报告。argmax描述不作为方向纠正器。

本轮为已消费模拟Development，有限AABB/背景、人工误差、相关seed/K/帧不是独立确认或实物鲁棒性。M3、原5格融合、L2、原weak_pass候选、480身份与旧停止结果保留；未训练、forward、新观测、硬件或访问保护480。不按validation挑seed/policy/点或升级基线；不同机制可继续探索，OR结构性零损失不变成hybrid的零损失门槛。


## 核验与资源

独立audit状态PASS（首轮执行），完整结构检查与来源见收据。报告仅消费最终诊断/政策结果，未重算科学账本。
8项policy结构检查、5项tail聚焦检查通过。tail首个fixture曾因阈值记录嵌套接口失败，已修复后重跑该检查并保留失败源码/收据；正式科学执行一次。独立复算覆盖108政策格、54本轮cal记录、82944政策接触账本（27648条1cm）、54旧cal阈值及前驱、4548尾部单位/峰值、22336时序贡献、72分解路径/432 summary/55296事件行与87政策输入绑定，均PASS。OR全slot超集、ideal-cal两类成本精确等于center，以及hybrid共享θ最低可行均已核验。

CPU命令实测：policy 7.067s、tail 4.503s（含首次fixture失败）、独立审计7.030s、报告3.149s，合计21.749s；根端盘点/文档核验/Git交付保守扣额40s，总扣额61.749/600s。内部函数计时已包含在命令wall内，不重复相加；40s扣额不是性能测量。GPU0/0，无训练或模型推理。[预算收据](../../../../artifacts.local/work/cnh-direction-anchor-dev-20261009/budget_final.json) · [冻结PLAN](../../../../artifacts.local/work/cnh-direction-anchor-dev-20261009/PLAN.json)。

新政策源码、tail源码与PLAN的SHA在执行和交付时一致；原模型、阈值、输入来源和旧失败保留。小型CSV/JSON/原分数/来源快照、审计源码、报告和已实际查看的一张科学图持久保留用于复算；task计算命令已结束，无常驻模型或GPU占用。未重新运行上一轮0/+3°模型回放一致性，本轮只读其已核验分数与来源；未验证新独立背景、反号偏差、实机或安全效果，未执行剩余预算OR、δ/权重敏感性或误差混合。

交付核验与最终commit/远端一致性另记于payload的delivery_receipt.json；历史CNH与并发RGB正文保留，不将未运行项记为通过。
