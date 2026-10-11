# 第一次有标注 CNH：跟着脚本录制

2026-10-11，EXPLORE。当前只准备与模拟串口测试，不采集、不刷写、不修改冻结模型。CPU预算≤1200 command-wall秒。操作以[引导脚本](host/real_labeled_capture.py)为主，不制作打印版；[标注方案](REAL_LABELED_ANNOTATION_PLAN_20261011.md)说明后处理边界。

## 开始前准备

准备设备及稳定支架、电脑、卷尺、手机水平仪与相机、胶带、同一大平面（优先家中同一面墙）、固定短杆与支架。杆建议长20cm、直径2–4cm，记实际长度、直径和材质；不能手持。没有短杆可只露出20cm杆段，但遮挡物与支架也要记录，不能假装它们不存在。

找一个前方轴向至少4.8m、最好5m以上无物体的开阔区域；同时检查两侧家具和地板。若家中没有，留空该条件，不能用近墙冒充开阔。每段设备和物体静止，人离开视场；固定杆的支架尽量放在走廊外，拍照记录所有背景。手机每段开始前和结束后各拍一张，保留照片原文件并在脚本中填写路径或照片编号。

设备中心建议离地 **H=1.65m**、光轴向下 **10°**。优先H=1.60–1.70m；场地需要时允许1.20–1.70m，目标高度随H调整。俯角建议8–12°；手机的向下角度记为正。实际值必须实测，不将建议值填成测量。把设备“向前”的方向用地面胶带标出来，量距离时沿地面平行方向量到目标前表面；不要用斜着的光轴距离代替轴向前距Z。若起点只能到外壳，记录起点偏移和至少±2cm误差。

冻结走廊相对设备：横向左右±30cm；HEAD离地H−42cm至H+20cm，BODY离地H−90cm至H−42cm；query前距0.30–3.00m。H=1.65m时，HEAD高1.23–1.85m，BODY高0.75–1.23m。HEAD/BODY名称不等于绝对人体部位。10cm擦边壳到横向±40cm。坐标/俯角来源见[几何合同](REAL_LABELED_ANNOTATION_PLAN_20261011.md#几何合同与摆放建议)；H=1.65m是继承场景的安装假设，并非已完成的物理标定。

## 六种摆放：脚本会逐段提醒

| 顺序 | 摆放与要测的数值 |
|---|---|
| 1 开阔 | 前方>4.8m无物体；量最近前方墙/物体距离，记录地板、侧墙与家具。距离无法测到时填未知，不能编造。 |
| 2 近平面 | 同一墙/板前表面Z=0.64m，正对走廊；推荐宽≥1.5m、高≥1.4m。板中心高H−0.30m（名义1.35m）；墙则记可见边界。 |
| 3 远平面 | 同一材质、同一平面，Z=1.50m；可移动设备到1.50m，重新测H/俯角，不换另一反射材质。 |
| 4 HEAD杆 | 竖直20cm短杆，前表面Z=1.50m，中心横向X=0，中心离地H−0.10m（名义1.55m）。 |
| 5 BODY杆 | 同杆、同Z与X，中心离地H−0.65m（名义1.00m）。 |
| 6 外侧杆 | 同HEAD杆、同Z与高度，中心向右偏X=+0.50m；直径4cm时内侧边在+0.48m，超出±0.40m擦边壳。 |

两档照明：dark=关灯、拉上窗帘；bright=开灯、拉开窗帘。避免阳光直射传感器，记录灯/窗帘设置和可选照度。灯光不一定提高近红外ambient，脚本保留真实CNH ambient；不要求它达到人为比例。

每个独立会话先按1→6各录dark与bright两段。完成第一会话后拔插ToF重新开始，记录是否重新安装；第二会话重复12段。共24段，每段先5秒倒计时，再30秒记录；有效录制12分钟，约3670帧（5.1Hz），加倒计时与自检至少14分钟，测量、照片和摆放通常使总耗时达到30–50分钟。重录额外计时，失败段保留。

## 每段怎么做

1. 看脚本显示的条件，摆放后输入实测前距、设备高、俯角、目标中心高/尺寸、杆直径、横向偏移、材质、ambient档和可选照度；未知就记未知。支架与背景写进备注。
2. 拍开始照片，填路径或编号。确认设备和目标固定、人能在5秒内离开视场，再按提示开始。
3. 倒计时后静止30秒。电脑显示帧率、丢帧与逐zone固件距离/ambient。固件距离异常只是提示检查摆放，不自动改测量或生成真值。
4. 结束后拍照片并填写。脚本质量提示有异常时按提示重录；旧段保留并标superseded。不要为通过检查修改实测量。
5. 中断后用resume命令继续；第一会话结束再按脚本提示拔插，开始第二会话。汇总确认24个条件/照明/会话槽位，照片与实测信息齐全。

未来开始录制时，先关闭占用COM5的台架演示/串口监视器，在仓库运行以下命令。脚本先打开监听，再提示手动短按ToF板RESET读取boot/config；不发送重置或刷写命令。

```powershell
Set-Location E:\linnan\linnan
& .\artifacts.local\hardware-bringup\venv\Scripts\python.exe .\research\active\hardware-bringup\host\real_labeled_capture.py --port COM5
# 中断后续录（第二会话加 --session 2）：
& .\artifacts.local\hardware-bringup\venv\Scripts\python.exe .\research\active\hardware-bringup\host\real_labeled_capture.py --port COM5 --resume
# 例如只重录第一会话的HEAD暗档：
& .\artifacts.local\hardware-bringup\venv\Scripts\python.exe .\research\active\hardware-bringup\host\real_labeled_capture.py --port COM5 --session 1 --redo head_rod-dark
```

其他参数见脚本`--help`与[回放收据](REAL_LABELED_CAPTURE_REPLAY_TEST_20261011.md)。默认从第一会话引导到第二会话，结束第一会话后提示拔插重启。Atom首末JPEG是可选本地网络快照，无Wi-Fi也能继续；如需启用，添加`--camera-url http://<Atom私有IP>:81/stream`。原始字节、解析帧、主机时间戳和metadata保存在`artifacts.local/work/real-labeled-capture-<date>/session-<n>/`，不上传。

标注以后基于实测几何、前后照片与人工复核，固件距离仅作诊断。开阔场景不自动等于clear，地板/支架/背景也须审核；没有逐query背景证据的部分为UNKNOWN。这批静态录制不提供动态接触及时性指标。
