# 第一次有标注 CNH：后处理标注方案

2026-10-11，EXPLORE，**只准备方案与离线骨架，本任务不执行实机标注**。输入未来录制的卷尺/水平仪实测量、前后照片、人工复核的完整场景几何；固件逐 zone 距离/status 只供信号诊断，不能生成真值。

## 几何合同与摆放建议

冻结 M3 的 query 来源是 [`cnh_cvr_projection.py`](../dtr-r0/nearfield/cnh_cvr_projection.py) 的 `query_masks`：X 向右、Y 向下、Z 向前；HEAD 为 `x=±0.30m,y=-0.20..0.42m,z=0.30..3.00m`，BODY 为 `x=±0.30m,y=0.42..0.90m,z=0.30..3.00m`。HEAD/BODY 是继承名称，不用名字推人体绝对高度。M3 训练的标签扩大到 x=±0.33m；实物评价仍用原 ±0.30m 接触线和 0–10cm 擦边壳，不据 M3 名称扩大物理接触标签。

传感器离地高度没有唯一全局冻结值；继承场景地面 `y=1.65m` 是 **H=1.65m** 的名义安装假设。操作优先 H=1.65m（1.60–1.70m）；家中支架可接受 1.20–1.70m，低于1.60m是偏离名义，必须记实测值并重算绝对高度/视场。高度范围为 HEAD `[H-0.42,H+0.20]`、BODY `[H-0.90,H-0.42]`，因此 H=1.65 时分别 1.23–1.85m / 0.75–1.23m。沿用 `rotation(-10,'x')` 的矩阵后光轴 Y 为正，物理上是 **向下俯 10°**；手机记录向下为 +10°，代码变换为 −10°。建议 8–12°，记录正负定义和实测量。来自 [`cnh_cvr_pilot.py`](../dtr-r0/nearfield/cnh_cvr_pilot.py) 与 [`cnh_aligned_boundary_dev.py`](../dtr-r0/nearfield/cnh_aligned_boundary_dev.py)，物理外参仍需以后标定。

平面前表面轴向 Z=0.64m / 1.50m，从 CNH 光学原点对应的投影量到目标面；不能把斜着量的光轴距离/径向距离混写成 Z。量距起点若只到外壳，记录偏移未知，并给至少 ±2cm 的位置误差。大平面宽≥1.5m、高≥1.4m，覆盖建议 HEAD 与 BODY 条带；可用家中同一墙面，移动设备实现两距离并保持同一反射材质/区域。记实际宽高与上下左右边界，做不到不把未覆盖部分写 clear。

细杆建议 **固定的竖直短段**，长 0.20m、直径 0.02–0.04m，前表面 Z=1.50m、中心 X=0：HEAD 中心高度 H−0.10m（名义 1.55m），BODY 中心高度 H−0.65m（名义 1.00m）。长度、直径和支架都实测；长竖杆可能跨两个 query，不能只按中心标签。外侧条件用同 HEAD 杆/距离，高度不变，中心 X=+0.50m；直径4cm时最近侧 X=.48m，离接触线18cm，明确超过10cm擦边壳。杆支架置于 query 外，并记录支架/桌子/地面等实际表面。

45°矩形名义 FOV 下，HEAD/BODY 中心的竖向相对光轴角约 −6.19° / +13.43°；HEAD 外侧杆中心水平约18.6°，均落在 ±22.5° 内。以上只说明名义中心可见，不能保证离散射线命中细杆或每个 query 完整可见。保持同一距离可减少杆被FoV裁切的混淆；实测偏角不同则重算描述，不用固件是否返回决定真值。

## 人工复核与标签

1. 给每个 segment 的 `metadata.json` 算 SHA256，前后照片关联到该 segment；照片带卷尺/目标尺寸和支架。复核设备/目标固定、画面无人，标出摆放或拍照引入的移动时刻；不明确的帧 UNKNOWN。
2. 把实测几何转到重力对齐、原点在传感器的走廊坐标：`y=H-h_above_floor`；Z 是轴向前距，圆柱前表面距须加半径得到圆柱中心 Z。X 正右；物体全尺寸用于几何，不能只用中心点。板面用有限厚度 box；竖直短杆用 closed cylinder，圆柱64面近似要保留在误差记录。
3. 每 query 算 **全部物体表面** 与该 query 盒的交集。POS 表示表面进入接触盒；不表示人实际接触。冻结 `cnh_sequence_observed_geometry.py` 的 `EPS=1e-8m` 用于收缩接触盒横向边界，零面积切触不算交集。无 POS、同 y/z 范围内表面落到横向±.40m扩展盒，标 pass。两个盒都没有已知表面，且人工确认完整 query/背景清单，才标 clear；否则 UNKNOWN。0.42m共边由闭区间同时覆盖，量具误差跨边界时 UNKNOWN。
4. 实测误差不能省略：骨架对位置误差的 −界/0/+界，以及杆半径/长度上下界作有限敏感性检查；跨 POS/pass/clear 边界标 UNKNOWN。可靠正表面可在背景未完整复核时给 POS；pass/clear 必须背景完整。倾斜平面、非竖直杆、任意复杂物体或姿态不明确超出本骨架，需人工 mesh/扩展几何实现，当前 UNKNOWN。
5. 场景固定时逐帧复用已审核几何标签，运动/重装期间 UNKNOWN。序号缺口与模型8帧历史、5分数平滑暖机另计，不能伪造缺帧或把缺帧当 clear。静止录制没有走近轨迹，`0.9m 首次报警及时性=NOT_EVALUABLE`；不能生成动态接触/停步指标。

继承评价口径见 [`cnh_sequence_observed_geometry.py`](../dtr-r0/nearfield/cnh_sequence_observed_geometry.py#L169)、[`cnh_margin_labels.py`](../dtr-r0/nearfield/cnh_margin_labels.py) 和 [`cnh_aligned_boundary_dev.py`](../dtr-r0/nearfield/cnh_aligned_boundary_dev.py)。旧三层口径“contact 需要报警，pass 通知可接受、单列，clear 的通知计成本”保留。本次可统计有人工几何 clear 支撑的静止回放触发率，场景/会话作为独立单位，不能写自然实机总体误报率；UNKNOWN 不进入负例分母。

## 六条件用途与数据隔离

| 条件 | 有资格分析的量 | 不能自动推导 |
|---|---|---|
| 开阔 >4.8m | 经人工审核 clear 区域的触发、固定残差、ambient/方差与bias | 条件名/无效距离不等于空场景，地板/侧墙仍须审核 |
| 同平面 .64m / 1.50m | 实测距离约束下幅值、峰宽/脉冲、跨bin扩展；bias与回波联合分离诊断 | 表面反射率未知，不能直接换成光子计数；有平面的bin不能当纯串扰 |
| HEAD/BODY短杆 | 测量支持的几何 POS、细目标可见性与静态触发 | 不产生接触及时性，不证明触发来自目标 |
| 同距外侧杆 | 与内侧杆匹配的 clear/pass 分解、擦边成本 | 不按中心偏移代替全部物体几何，不把支架忽略 |

两档 ambient 指两种可复现照明设置，实测 CNH ambient 与可选照度并存；“开灯”不保证传感器 NIR ambient 更高。A 会话拟合输入增益/floor/bias，B 会话只做冻结后的描述评价；本12分钟小集合不支持选模型/阈值再宣称独立确认。至少拔插一次记录开机和重新开始，若没有重装不声称跨安装泛化。

## 离线骨架与未来命令

[`host/real_labeled_annotation.py`](host/real_labeled_annotation.py) 不打开串口、不访问模型/保护数据、不联网。`template` 只创建默认全部待复核的 JSON；默认不能得到负标签。审核单必须绑定原采集 metadata SHA256；输出用独占创建，旧标签保留。

采集 `frames.jsonl` 保留 Decoder envelope：序号/设备毫秒时间取 `sensor.seq/sensor.ms`，主机时间取外层 `host_received_utc/host_received_monotonic_ns`。后处理输出保留这些字段；人工逐帧动作审核可在外层或 `sensor` 添加 `invalid/motion_during_capture/pose_changed`，任一为真则 UNKNOWN。这里的 invalid 是审核标记，不从固件 status 生成。

```powershell
# 未来完成录制后执行，当前任务不运行真实标注。
python research/active/hardware-bringup/host/real_labeled_annotation.py template --output artifacts.local/work/real-labeled-capture-20261011/session-1/segment-REPLACE/geometry-review.json
# 人工填照片、实测坐标/误差、objects、完整性审核与 metadata SHA256 后：
python research/active/hardware-bringup/host/real_labeled_annotation.py annotate --frames artifacts.local/work/real-labeled-capture-20261011/session-1/segment-REPLACE/frames.jsonl --metadata artifacts.local/work/real-labeled-capture-20261011/session-1/segment-REPLACE/metadata.json --review artifacts.local/work/real-labeled-capture-20261011/session-1/segment-REPLACE/geometry-review.json --output artifacts.local/work/real-labeled-capture-20261011/session-1/segment-REPLACE/labels-v1.jsonl
```

路径中的 segment 名应替换成采集汇总给出的实际目录。板面 object 格式为 `type=box,low_m=[x,y,z],high_m=[x,y,z],translation_uncertainty_m=[dx,dy,dz]`；杆为 `type=cylinder_y,center_m=[x,y,z],radius_interval_m=[r_min,r_max],length_interval_m=[l_min,l_max],translation_uncertainty_m=[dx,dy,dz]`。不确定的尺寸/姿态不要填名义值冒充测量。floor与固定家具也属于objects，完整性审核必须说明覆盖范围。

骨架未建立连续误差区间的数学证明；有限边界敏感性、圆柱离散化与重力/走廊名义方向是局限。准备阶段只查结构/语法与合成几何；未来录制后的真实标注、拟合和回放另行执行。
