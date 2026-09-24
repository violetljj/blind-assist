# City0 路缘石逐实例派生替换：已消费布局的反证

2026-09-24。结论：**路缘石 Nanite 轮廓假设失败，City 整体仍不准入。**
同一 2 个已消费 City 布局、16 帧 Development 工程诊断中，保留原车辆近场清零对照，
只把 City0 黄色路缘石的指定实例替换为原位、原材质、Nanite 关闭且强制 LOD0 的未保存派生网格。
原始 30/75 mm 和逐区 2% 规则不变。未采新布局、未访问正式测试集、未给旧回执补 PASS。

## 固定目标与实现

- 旧车辆清零诊断 `artifacts.local/evidence/cnh-city-vehicle-mask-audit-20260924-v1/diagnostic/result.json`
  将像素 (223,277) 的四次重复最大误差归给黄色路缘石导出网格；这只是**导出网格射线赢家**，不是原生可见实例 ID。
- 固定组件 `TemplateActor_UAID_D45D6454B40F8AF200_1551189997/...YELLOW__11` 的实例 3、surface index 15；原网格 `/Game/Megascans/3D_Assets/Modular_Curb_5_M_00/Modular_Curb_5_M_LOD0_vcflbc0dw_YELLOW`。启动前核对旧诊断四片段的同一射线和实例，拒绝改目标或候选搜索。
- [实现](cnh_city_instance_substitution.py)另有完整近场车辆加路缘石模式，但本次**未运行**该模式。每个替换实例保留原组件与索引，原变换临时置零；在原世界变换生成派生 StaticMeshActor，派生网格关闭 Nanite，组件 `disallow_nanite=True`、`forced_lod_model=1`，有效材质槽保持一致。退出前销毁派生 Actor、恢复原变换，核对源网格、远实例、custom data/种子/材质/剔除元数据，不保存地图或资产。

首个运行 `cnh-city-curb-substitution-20260924-v1` 在 UE 启动前因 Python 找不到项目本地 OpenEXR 失败，保留失败回执。
使用已有 `artifacts.local/work/cnh-route-comparison-20260924/runtime` 后，
`cnh-city-curb-substitution-20260924-v2` 完成 16/16 帧与七路传输；一件路缘石被替换、16 件近场车辆维持旧清零对照。
源项目和地图哈希不变；原组件恢复、派生 Actor 释放、UnrealEditor 退出均通过。
进程树回执的 11 个任务进程全部为 `EXITED`，临时目录已释放。
相关单测 16 项通过；真实 UE 回执是捕获与资源完整性检查，**不是几何准入**。

## 原阈值诊断

| 布局 | 背景径向 P95 旧→新 | 背景最大误差旧→新 | 结论 |
| --- | ---: | ---: | --- |
| City0 | 14.690→14.674 mm | 881.078→880.927 mm | 4 次同一射线仍超过 75 mm；FAIL |
| City1 | 6.202→6.202 mm | 28.341→28.341 mm | 此单项维持旧 PASS；不推及 City 整体 |

City0 像素 (223,277) 的原生轴深仍为 2.914830 m；派生路缘石的网格轴深为 3.699307 m。
同一物理射线在 centre、boundary、outside、removed 的末端帧重复四次，并非四个独立错例。
zone53 的四个首位姿射线 (349/351/353/355,251) 也完全未变：
原生径距 5.002319–5.014444 m，导出人行道实例网格径距 4.980926–4.994948 m，
误差 19.50–21.39 mm；`extra_fraction=2.2214%`，仍越过 2% 原门槛。
不能采用 5 m 缓冲带、轮廓剔除或重命名背景来追认。

## 只读归因及下一机制

在新捕获的相邻采样像素，(223,275)、(225,277)、(227,277) 的导出赢家均为
`Parking_Meter_00/ujpkaadfa_LOD0` **实例 2**，轴深与原生深度相差约 0.02–0.17 mm；
唯独轮廓像素 (223,277) 的原生深度延续约 2.915 m 前景，而导出射线跳到 3.699 m 路缘石。
原生 2.914830 m 的世界点位于该停车收费表实例的变换后 AABB 内，其 AABB 投影覆盖约
x=212–238、y=183–321，包含问题像素；材料能力回执显示该实例无编译 WPO/PDO/位移。
这将**停车收费表边缘的可见覆盖/采样与 LOD0 中心射线差异**列为下一机制假设，
不能据 AABB 或相邻导出赢家直接证明问题像素的原生归属。

下一项若另行执行，应先给原生 (223,277) 建立可见实例 ID 或隔离深度对照，再只改该收费表实例的
Nanite/LOD0 路径并复核同一冻结射线与 zone53；同时记录 SceneDepth 像素采样/时间抖动状态。
如果该前景仍无法归属，就保持 `UNKNOWN`，不要把导出路缘石赢家当作原生真值。
zone53 属于另一个人行道 5 m 边界问题，需独立解释；路缘石或收费表局部修复不自动清除它。

证据：`artifacts.local/evidence/cnh-city-curb-substitution-20260924-v2/capture/`、
`artifacts.local/evidence/cnh-city-curb-substitution-audit-20260924-v1/diagnostic/result.json`；
冻结配置 `artifacts.local/work/cnh-route-comparison-20260924/plan/city-curb-substitution-v1.json`。
