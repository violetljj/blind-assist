# 稀疏峰运动对齐读出：未形成低成本及时收益

2026-10-10 · EXPLORE · `CNH_SPARSE_RAY_TRACK_DEV_20261010` · 输入冻结于`8d08aad9`。

结论：按原5格的校准通知预算选择阈值后，新增运动对齐读出没有及时救回或提前。validation HEAD/BODY仍为298/244，各/384；4cm HEAD横杆仍19/64，暗4cm仍1/32，而purepass通知61→63。单独替代原5格时，4cm横杆救2损10、净−8。保留原M3/5格，不采用本配方，也不继续扫描本配方窗口或阈值。信号可对齐不等于低成本风险证据，不否定其它原始直方图读出机制。

本次承接[期望回波oracle](CNH_HEAD_THIN_SIGNAL_ORACLE_DEV_20261010.md)，没有在[已消费的新场景E2E hold](CNH_RGB_FROZEN_E2E_RESULTS_20261010.md)上选择机制或阈值。此次最细横杆为旧开发数据的4cm，不是上次的1.7cm；未建立1.7cm迁移或真实ToF结论。

## 固定机制与比较

从当前帧全部native bins的正值、公共query门控后的径向局部峰生成候选，不限top8、不使用目标真值。中央ray和bin中心给出位置假设，用当前已知sensor pose映到world，再逆投到过去已观测帧的zone/range bin；保留signed历史值，用固定三bin核[.25,.5,.25]累积并除以参考噪声核范数。每个候选在可用1/2/4帧前缀取最大值，乘当前公共expanded-query membership，再跨候选取最大值。

主方案`aligned_adaptive`；不对齐控制`unaligned_adaptive`具有同候选、核、窗口搜索，仅重复当前native zone/bin；`current_only`只用当前帧同核。`aligned_fixed4`只保存描述，不评价选优。f3空历史开始，不使用f0…2或未来帧；每个scene/replica/query独立。中心点只是证据关联假设，不是物体身份；静态world/已知pose是ideal条件。三bin跨度约0.9m，是工程容错核而非标定pulse；分数是参考噪声单位，不是oracle中的含信号方差SNR。

主方案和两个控制都在原5格grade0处追加轻提醒，原5格strong逐slot保留。补充主读出独立方案，正输出全部轻提醒，真实救/损和强转轻单列。所有方案使用相同gap1通知回放：首个positive立即发、同episode只发strong升级、两个连续0重置、同帧HEAD/BODY取max算一次联合通知。

旧consumed ideal cal/validation各384 authored rows×K4×13输出帧。每高度contact384事件，HEAD横杆96，4cm层64、其中rho=.25暗层32。每split有324个不同physical key；64个absence rows实际只有4个背景world。原曝光计数和成本预算保持全部重复行/K4；接触描述CI按physical key聚类，不报负例成本CI，背景设计因素仍相关。

校准只用cal joint-clear/purepass负例，完整枚举每个负例score tie排除点、全纳入点和∞；gap1成本非单调，故不用二分。取最小阈值使clear/purepass联合通知分别不超过原5格39/60；不读取接触收益选工作点，cal封存后才评价validation。预算1800 command-wall秒：main900、evaluation300、audit300、integration300；实现/路径修复和chunk大小可调，机制、核、窗口、成本额度不按结果扫描。判据是实际配对及时收益与成本，而不是分数增强；固定比较完即结束。

## 结果

下表是validation，HEAD/BODY及时各/384；4cm HEAD横杆/64，暗4cm/32；clear/purepass通知分别对应512/256个曝光clip：

| 方案 | HEAD | BODY | 4cm HEAD | 暗4cm | clear通知 | purepass通知 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原M3 | 284 | 221 | 18 | 0 | 30 | 63 |
| 原5格 | 298 | 244 | 19 | 1 | 29 | 61 |
| 5格＋运动对齐 | 298 | 244 | 19 | 1 | 29 | 63 |
| 5格＋不对齐控制 | 298 | 244 | 19 | 1 | 29 | 62 |
| 5格＋当前帧控制 | 298 | 244 | 19 | 1 | 29 | 61 |
| 仅运动对齐轻提醒 | 165 | 99 | 11 | 0 | 0 | 55 |

三个追加方案在cal及时与clear/pass成本也均等于原5格：HEAD293、BODY248、4cm24、暗4cm7，clear39/pass60。其固定阈值分别30.816072、29.553120、24.494898；独立轻提醒阈值14.952094，cal及时HEAD175/BODY93、4cm10/暗0，clear0/pass60。

三个追加方案两split全部HEAD/BODY/4cm/暗4cm及时救回、损失和共同及时提前均为0；零损失来自追加构造，不能称统计非劣。主方案validation新增2次purepass通知落在已有提醒clip，原5格与主方案clear/pass受提醒clip均28/60；校准成本一致没有保证validation成本一致。不能把此结果写成同成本收益，也不能把轻提醒记成零打扰。

独立方案validation 4cm及时救2损10；cal救4损18。validation共同及时的9个4cm事件由strong变成light；不是完整强提醒替代。暗4cm及时0/32，原5格的1件也丢掉。其他高度、完整窗救损、共同及时早晚、逐query强轻、通知/clip和事件ledger保存在payload，不只报净值。

完整f3…15窗口补核验也没有隐藏救回：三个追加方案全部接触层救/损0/0，HEAD354/384、BODY258/384、4cm39/64、暗4cm16/32均保持原5格。独立方案完整窗分别救/损5/89、1/158、2/17、0/10；4cm为39→24/64。独立方案共同及时的9个4cm事件中，8件延后、1件同帧，中位延后3帧；HEAD共同及时157件中147件延后，BODY98件中91件延后，中位均5帧。完整窗与及时窗指标不能混用。

## 失败原因与下一决定

只读诊断固定分数和通知，不重新选阈值。cal最后一个排除tie把clear/pass成本39/61降至39/60，决定性事件为HEAD protrusion纯pass、f13、2帧窗、query深度约0.981m，inner_share=0。validation新增两个grade slot也都是HEAD纯pass、f13、inner_share=0，分别来自vertical和protrusion场景；场景类别不代表峰的物体身份。

validation 4cm HEAD横杆及时窗最大分数，rho=.25层中位5.814、最大8.012；rho=.65层中位14.005、最大17.537。两层各32事件，全部低于主阈值30.816。对齐相对不对齐的配对最大分数差，暗层中位+0.054、亮层0；纯pass差值p90为+2.670。该统计量增强了部分背景/擦边高尾，没有获得区分弱目标的空间选择性。不同方案分别校准，不能用这些分数差代替任务收益。

薄杆及时argmax的inner_share中位同样为0，故直接删除外侧支持会连浅接触证据一起删除，尚无依据采用这种门控。当前中央ray/bin中心假设、粗距核与幅度最大值不能区分浅接触和擦边，这是本固定配方的失败定位，不是传感器物理上限。停止本配方；若继续读出研究，优先检验局部zone/range形状和相邻背景对比是否提供这种区分，最弱假设是粗空间/距离采样仍保留可分形状。先做同一开发集的低成本可分性诊断，再决定实现；不以加窗、降阈值或建同步RGB源绕过此次负结果。

## 核验与失败记录

独立96个标量候选/窗口/anchor/path复算分数误差0；72个独立坐标映射、8个未来hist/pose/query及pre-f3扰动前缀检查、4个signed FP16归一化检查通过。确定性移动点的aligned/current分数比2.0，证明关联方向可增强理想工程fixture，不是目标检出增益。另独立复算36864逐流、12个方案/split单元、8个配对、4条完整校准曲线身份/最小可行点及关注点实际成本；均通过。

初次计算在最终receipt路径序列化失败，原因是canonical junction解析到F盘后仍按E盘repo求相对路径。两split分数已经算完；保存原source snapshot、失败及SHA，修复路径命名并用`--finalize-existing`复核原数值函数/config AST、输入身份和少量prefix后恢复receipt，没有重复全量readout或覆盖分数。主命令13.270s（失败只在receipt），恢复内部0.797s；评价命令19.050s；核心核验命令1.680s，其余核验/预算/资源见delivery回执。

评价源后来出现尚未使用的resume辅助改动；执行产物结构与source SHA匹配原冻结版本b75b8c45，原字节snapshot恢复并核验，数值/cal/count函数AST一致。原after-resume版本只留artifact，无新校准或结果替换。失败、source lineage与旧输出全部保留。

没有训练、网络forward、新光子采样、GPU、保护数据、E2E hold或Android修改；任务进程结束，耐久输出与snapshot保留于canonical artifacts.local。当前证据只来自选择过机制的模拟Development，nominal帧时序/公共姿态、粗距bin和有限背景均非实机性能或安全结论。

源：[读出](cnh_sparse_ray_track_dev_20261010.py) · [评价](cnh_sparse_ray_track_metrics_dev_20261010.py) · [核心核验](audit_cnh_sparse_ray_track_dev_20261010.py) · [指标核验](audit_cnh_sparse_ray_track_metrics_dev_20261010.py) · [完整窗补核验](supplement_cnh_sparse_ray_track_metrics_dev_20261010.py)。

payload：`artifacts.local/work/cnh-sparse-ray-track-dev-20261010/`，PLAN、两split scores/public geometry/grade通知、封存cal和完整曲线、metrics/ledger/补充配对、audit、失败/源码snapshot、delivery receipt。
