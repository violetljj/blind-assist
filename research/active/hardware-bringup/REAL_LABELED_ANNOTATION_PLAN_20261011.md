# 宿舍单次10段 CNH：简单后处理与弱标签

2026-10-11，EXPLORE。一次录10段：1→2→3→4→4b→5→6→7→8→9。4b是同位置暗空通道，替代原另等夜间的10/11；5恢复明档，9也是明档拔插重复。只做准备，不采集、不刷写、不运行后处理。跟随脚本摆放即可，不要求卷尺、照度计或完整mesh审核。一次填大约传感器高度；距离用固件中央/书本代理弱参考，照片可选。

## 按条件对照冻结模型响应

condition_label / expected_response保存摆放目的；per-query label为近似POS/pass/clear/UNKNOWN。source=CONDITION_WEAK_APPROXIMATE、evidence_quality=weak_approximate。可按条件预期比较冻结模型的静态响应，不能称独立真值检出率/误报率。

| 段号与条件 | condition_label | expected_response | per-query label |
|---|---|---|---|
| 1阳台外无近回波 | corridor_empty | silent | 无中央有效距离可条件弱clear；有.3–3m弱表面则UNKNOWN |
| 2近墙约.6m；3同墙约1.5m | plane | calibration | UNKNOWN；只供幅值/峰宽校准，不凭名称给负标签 |
| 4明空通道；4b原位暗空通道；9明档拔插重复 | corridor_empty | silent | 距离未知UNKNOWN；窗帘等弱表面在.3–3m→UNKNOWN；弱距离下界明确>3m才可条件弱clear |
| 5恢复明档落地竖杆约1m；6同杆约2m | corridor_target_present | alarm | 整杆可能跨两query；高度/长度未知则UNKNOWN，条件报警预期保留 |
| 7外侧杆横偏约.5m、距离约1m | outside_target | no_strong_alarm | x误差跨±.30接触线/±.40擦边壳→UNKNOWN，不强报警预期不是几何clear |
| 8额头横杆，距离约1–1.5m | corridor_target_present | alarm | 按估计H与杆高判断；低支架不能按HEAD名称标HEAD POS |

expected_label另保留query层面的意图。质量失败、interrupted/needs_review/superseded/skipped、移动或未知边界均UNKNOWN；保留失败/重录证据。逐queryUNKNOWN不妨碍条件响应对照，不当负例，不填充缺帧。

固件距离及书本代理来自同一传感器，只作弱参考，不是独立真值。空通道、阳台外、无中央回波都不自动等于零ambient、纯噪声或纯串扰；墙回波也不作purebias。每行pure_noise_or_bias_template=false、independent_ground_truth=false。

## 4/4b的噪声与ambient配对

设备和通道位置不变；4明档→4b暗档只改照明，之后5恢复明档。先比较真实CNH ambient分布：**有实际差异时**才讨论ambient与方差变化；开/关灯并不保证传感器NIR ambient改变。两档ambient没有差别，保留这一结果，不声称完成噪声斜率辨识。

若拉帘同时改变通道尽头表面，记录surface混杂，比较时避开受对应表面回波影响的bin，不能把它们当纯噪声真值或把整墙/窗帘回波当串扰bias。哪些bin受到影响需由实际回波和弱距离检查决定；当前任务不运行这项后处理。标签行noise_ambient_pair=4/4b只是配对身份，不是“几何相同/噪声已识别”的证明。

## 近似高度和姿态

冻结坐标X右/Y下/Z前：HEAD x±.30m、y−.20...42m、z.30..3m；BODY x±.30m、y.42...90m、z.30..3m。传感器离地H时，绝对高度分别[H−.42,H+.20]和[H−.90,H−.42]。H=.90m的HEAD=.48–1.10m；H=1.20m的HEAD=.78–1.40m，1.4–1.6m横杆不能只凭条件名称直接HEAD POS。

脚本一次估计H默认1.20m、水平tilt_deg=0；冻结Rx(−10°)对应向下俯10°。估计姿态偏离随结果保留，不声明物理标定。固件通常是径向距离，query Z是轴向，保留误差来源。默认描述误差H±.15m、杆高±.20m、x±.15m、距离±max(.25m,20%)；区间碰±.30/±.40、Z=.30/3或相应高度边界→UNKNOWN。尺寸未知安全返回UNKNOWN，不新增测量门槛。

静态0.9m首次报警及时性始终NOT_EVALUABLE，不生成动态接触、提前量或停步效果。冻结依据：[cnh_cvr_projection.py](../dtr-r0/nearfield/cnh_cvr_projection.py) query_masks、[cnh_cvr_pilot.py](../dtr-r0/nearfield/cnh_cvr_pilot.py) rotation、[cnh_sequence_observed_geometry.py](../dtr-r0/nearfield/cnh_sequence_observed_geometry.py) surface_category。评价沿用±.30接触线及0–10cm擦边（±.40扩展），不按M3训练±.33扩大接触线。

## 以后录完的离线入口

[host/real_labeled_annotation.py](host/real_labeled_annotation.py) approximate读取实际录制根capture.json和各attempt的metadata.json/frames.jsonl，输出每frame/query与summary。不会打开串口、联网、运行模型或访问保护数据；独占创建输出，重跑换文件名。skip无帧只在summary保留。

```powershell
# 以后录完使用，本次准备不运行。
python research/active/hardware-bringup/host/real_labeled_annotation.py approximate --session artifacts.local/work/real-labeled-capture-20261011 --output artifacts.local/work/real-labeled-capture-20261011/weak-labels-v1.jsonl
```

按实际日期替换目录。condition_index支持整数1..9及字符串4b；旧夜间10/11仅为历史数据兼容，不属于当前录制计划。metadata保存sensor_height/source、tilt/source、target_distance/source、distance_reference（stable/有效比例/中央zones/探测路径），杆尺寸及位置可缺省。nominal计划值不是实测，不从条件名补成真值。

frames读sensor.seq/ms、host_received_utc/host_received_monotonic_ns。外层或sensor的invalid/motion_during_capture/pose_changed→UNKNOWN。逐帧弱表面可揭示空条件冲突，但不能将失败/未知稳定距离探测变成已确认远场。旧template/annotate严格入口保留在代码，旧六条件详案可查Git ada649c3；它不是本次快速准备的启动门槛。