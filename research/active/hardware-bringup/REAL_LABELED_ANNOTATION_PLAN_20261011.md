# 宿舍11段 CNH：简单后处理与弱标签

2026-10-11，EXPLORE。白天9段＋夜间重复1/4，共11段。本任务只准备，不采集、不刷写、不运行真实标签。按引导脚本摆放即可，不要求卷尺、照度计或完整mesh审核。一次填传感器大约高度；距离用中央固件距离/书本代理弱参考，照片可选。

## 按条件对照模型响应

每行保留两层信息：condition_label / expected_response记录用户摆放条件与“应静默/应报警/不应强报警”的目的；per-query label是近似POS/pass/clear/UNKNOWN。来源固定CONDITION_WEAK_APPROXIMATE，evidence_quality=weak_approximate。条件预期可以先对照冻结模型的静态响应，不能称独立真值检出率/误报率。

| 段号与条件 | condition_label | expected_response | per-query label |
|---|---|---|---|
| 1阳台外无近回波；10夜间重复 | corridor_empty | silent | 无中央有效距离可条件弱clear；有.3–3m弱表面则UNKNOWN |
| 2近墙约.6m；3同墙约1.5m | plane | calibration | UNKNOWN；只供幅值/峰宽等校准，不凭场景名给负标签 |
| 4朝窗帘空通道；9拔插重复；11夜间重复 | corridor_empty | silent | 距离未知UNKNOWN；窗帘等弱表面落.3–3m→UNKNOWN；弱距离下界明确>3m才可条件弱clear |
| 5落地竖杆约1m；6同杆约2m | corridor_target_present | alarm | 整杆可能跨两query；高度/长度未知则UNKNOWN，条件报警预期仍保留 |
| 7外侧杆，横偏约.5m、距离约1m | outside_target | no_strong_alarm | x误差跨±.30接触线/±.40擦边壳→UNKNOWN，预期不强报警不等于几何clear |
| 8额头横杆，距离约1–1.5m | corridor_target_present | alarm | 根据估计传感器H和杆高判断；低支架不能凭HEAD名称标HEAD POS |

expected_label另保留query层面的条件意图。质量失败、interrupted/needs_review/superseded/skipped、移动与未知边界均UNKNOWN，保留原段/失败数据。per-query未知不妨碍按condition_label比较响应。UNKNOWN单列，不当负例；缺帧不填充。

固件逐zone距离、中央距离与同传感器书本代理只供弱参考，不是独立真值。空通道、户外或无中央回波，不能自动成为零ambient、纯噪声或纯串扰模板；地板/侧墙/远表面仍会贡献回波。墙surface也不能作purebias。每行保存pure_noise_or_bias_template=false和independent_ground_truth=false。

## 高度与姿态的近似边界

冻结坐标X右/Y下/Z前：HEAD x±.30m、y−.20...42m、z.30..3m；BODY x±.30m、y.42...90m、z.30..3m。离地H时绝对高度分别[H−.42,H+.20]和[H−.90,H−.42]。例如H=.90m的HEAD=.48–1.10m，H=1.20m的HEAD=.78–1.40m；1.4–1.6m横杆不能因条件叫HEAD就直接HEAD POS。

脚本一次估计H默认1.20m，水平tilt_deg=0也是估计；冻结模拟Rx(−10°)对应向下俯10°。二者偏离随结果保留，本次没有物理外参/单位标定。固件多给径向距离，query Z是轴向，骨架保留误差来源。

默认描述误差H±.15m、杆高±.20m、x±.15m、距离±max(.25m,20%)。区间碰±.30/±.40、Z=.30/3或相应高度边界→UNKNOWN。尺寸未知不报错，不新增测量门槛，只保存条件预期和逐queryUNKNOWN。静态0.9m首次报警及时性始终NOT_EVALUABLE；不输出动态接触、提前量或停步效果。昼夜只表示照明设置，CNH ambient的实测差异才能说明传感器环境变化。

冻结来源：[cnh_cvr_projection.py](../dtr-r0/nearfield/cnh_cvr_projection.py) query_masks、[cnh_cvr_pilot.py](../dtr-r0/nearfield/cnh_cvr_pilot.py) rotation、[cnh_sequence_observed_geometry.py](../dtr-r0/nearfield/cnh_sequence_observed_geometry.py) surface_category。物理评价沿用±.30接触线和0–10cm擦边（±.40扩展），不按M3训练标签±.33扩大接触线。

## 后处理入口

[host/real_labeled_annotation.py](host/real_labeled_annotation.py)新增approximate，未来读取实际录制根目录capture.json与各attempt的metadata.json/frames.jsonl，输出每帧每query和summary。不会打开串口、联网、运行模型或访问保护数据。输出独占创建，重跑换文件名；skip无帧只在summary保留。

```powershell
# 未来录完后执行；本次准备不运行真实标签。
python research/active/hardware-bringup/host/real_labeled_annotation.py approximate --session artifacts.local/work/real-labeled-capture-20261011 --output artifacts.local/work/real-labeled-capture-20261011/weak-labels-v1.jsonl
```

日期按实际录制目录替换。metadata记录condition_index、sensor_height_m/source、tilt_deg/source、target_distance_m/source、distance_reference（stable/有效比例/中央zones/探测路径），可选target_x_m、target_height_m或target_height_low_m/high_m、rod_orientation/rod_length_m/rod_diameter_m及uncertainty。nominal计划值不是实测；不从条件名自动补尺寸。

frames保留Decoder envelope：读取sensor.seq/ms、外层host_received_utc/host_received_monotonic_ns。人工确认移动可在外层/sensor加invalid/motion_during_capture/pose_changed→UNKNOWN。逐帧有效中央弱表面可提示空条件冲突，但不能将失败/未知稳定距离探测变成已确认远场。

旧template / annotate严格几何入口保留在代码中；旧六条件详案可查Git ada649c3的同名文档，卷尺/完整mesh/照片审核不是本次快速录制门槛。未来有完整实测几何时可选严格入口，当前优先按11条件迅速定位信号与冻结模型响应。