# 宿舍十条件实际 CNH 录制：首次头戴 pilot

日期：2026-10-11；EXPLORE。用户追加授权实际采集，并要求根据实时画面自动开始、声音引导。原准备阶段“不采集”边界由该授权替代。本轮没有继续刷写、改变冻结模型/阈值或访问保护480/test。

## 结果

十条件各30s已存，选用段共1507帧；8×8×16布局、逐段raw SHA256、原字节长度和解析行数核验一致。保留所有失败准备段、近杆第一次丢帧段及后续重录。选用段不是全部质量通过，且不具有逐query独立真值。

| 条件 | 选用帧 | 缺序号帧 | 保存状态 | 几何依据 |
|---|---:|---:|---|---|
| 1 阳台朝外 | 143 | 10 | needs_review | RGB见室外楼/天空，下方有栏杆；中央多UNKNOWN，不能证明全zone无表面 |
| 2 白墙近 | 153 | 0 | complete | 准备弱测距约0.692m；录制中央中位约0.623m |
| 3 同白墙远 | 153 | 0 | complete | 录制中央中位约1.452m |
| 4 明档空通道 | 153 | 0 | complete | 条件摆放；保留左墙/床梯等背景，不等于全FOV空 |
| 4b 暗档空通道 | 153 | 0 | complete | 拉帘关灯，回到近似原站位；头姿未独立测量 |
| 5 近竖杆＋旁边筒 | 153 | 0 | needs_review | 约1m为摆放估计，中央距离无法核对；不是孤立细杆 |
| 6 远竖杆＋筒 | 153 | 0 | complete | 约2m为摆放估计，目标距离未独立测出 |
| 7 外侧杆 | 153 | 0 | complete | 约1m、横移约0.5m为操作要求/估计；杆移出中央画面，实际偏移未测 |
| 8 额头高度横杆 | 153 | 0 | complete | RGB见固定横杆伸入中线；距离约1–1.5m、高度约额头均为估计 |
| 9 重启后明档空通道 | 140 | 13 | needs_review | 移开杆/筒，真实ToF重启后录制 |

近杆第一次30s仅45帧、缺104帧，原位重录后153帧、零缺序号，旧段标superseded并保留。选用段的23个缺序号来自阳台和重启空通道，不隐藏质量警告。

4/4b的ambient zone中位再按帧取中位分别约7.546875与2.9008925，约2.60倍；单位为CNH/scaler归一化值，不是lux或光子计数。仅证明录到了不同ambient，尚未拟合新噪声模型。

第9段真实boot_id由`733729a37df06f02`变为`7794afafc78557bf`，`startup-06/gate.json`自检PASS、约5.1016Hz；TCP重连不算物理重启。

## 实际安装、标签和流程偏离

用户头戴传感器，估高1.55m，有自然抖动，俯角未测。`operator-context.json`及正式段metadata记录head_worn、static_scene=false；环境尽量不动，观测仍有头动，不能将帧方差直接视为严格静态仪器噪声。显示保留用户接受的RGB顺时针90°、镜像开和已观察ToF方向，原始区序/直方图不变，非完整外参标定。

杆中央未稳定测到；用户说明旁边有筒，后续不再要求反复放书。程序兼容流程仍有`book-probe`目录名，但现场没有添加书，metadata记录no book、原始程序来源及实际场景；该探测不作为书本代理真值。杆段target_distance_m置UNKNOWN，另外记录摆放估计及不确定性，原固件距离仍保留。偶有1–3mm或背景读数不能充作杆的真实位置。几何边界不清的query应UNKNOWN；本轮没有运行逐query标注或冻结M3/原5格/S回放，不能报告实机检出率/误报率。

明暗段头姿不是严格固定，拉帘也改变回波背景；近/远杆有筒的混杂，外侧实际偏移未测。后续先用开阔/平面段检查固定模式、幅值和脉冲，再对稳定短窗估计噪声与ambient关系，避免把运动和目标变化拟合为仪器噪声。

## 现场修复与收据

采集台声音开始一声、结束两声，播报条件和实时平面距离；支持即时中文口播，代理可通过本地按钮引导靠近/退后。等待摆放时读取内存帧，解决TCP消费者停读导致的板端断开；EOF自动重连仍保留原transport/boot来源。

最后30s已完整写盘，等待结束确认时Wi-Fi重连超时，原流程把末段标failed。核验十段均有完整30s、raw哈希/字节长度和8×8×16帧匹配后，末段改为needs_review，会话记complete_with_quality_warnings；原失败metadata和manifest单独保留，原字节/帧没有修改。等待下一段/结束确认不再读取硬件，避免结束后断连将已保存数据判失败。完成清单加载后采集台显示十段完成，不自动开始新采集。

数据：`artifacts.local/work/real-labeled-capture-20261011/`，含`capture.json`、`inventory.tsv`、逐attempt原字节/帧/时间戳、`operator-context.json`、`installation-view.json`。

独立文件核验：`capture-integrity-final.json`，10段/1507帧PASS；脚本`finalize_after_ack_timeout.py`。失败留证：`capture.before-finalization.json`与第9段`finalization-timeout.metadata.json`。沿用CPU≤900 command-wall s预算，未运行GPU/模型；验证限实际采集、Python编译、文件完整性和结束异常修复，未扩大测试。采集线程/ToF连接已结束，本机页面与相机预览可保留给用户，点“关闭本地服务”释放。

结束确认针对检查`end-ack-check.txt`通过：硬件读取会报错的替身下，下一段/结束回答仍正常返回；完成页面实际加载显示十段完成，worker_active=false、error=null，截图`capture-complete.png`。
