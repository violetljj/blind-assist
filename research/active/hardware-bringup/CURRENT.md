# 实机接入：当前状态

更新：2026-10-13。本页已收拢旧过程；完整整理前正文保留在Git `cd227114`，旧结果与失败沿用各报告。总研究决定见[CURRENT_DECISION](../../../docs/CURRENT_DECISION.md)。

## 手机融合实验已安装

[10.19.0 工程接入](MOBILE_FUSION_INTEGRATION_20261012.md)已安装 SM-S9280：数据采集 → 打开融合实验 → 开始实验，独立显示 ToF 强档或 DAV2-S RGB，经 2/3 帧确认；可低头标定高度，静音且离页停止。模型数值/图像变换四项设备检查通过，默认辅助仍为基础 ToF。现场设备摆放触发姿态门控，佩戴后的完整 RGB 融合链路尚待验证。原离线整段姿态/最近帧协议保持，手机因果匹配不是其等价确认。新录制按[冻结执行入口](FROZEN_FUSION_EVAL_RUNBOOK.md)进行。

## 手机采集与姿态

手机SM-S9280当前版本10.19.0（本轮覆盖安装）。[手机操作](../../../docs/PHONE_DATA_CAPTURE.md)支持Atom原始RGB、H3完整8×8×16 CNH和BMI270，同时预览、显式起止、原始记录和ZIP导出；断流/后台保留中断段。H3仅一个TCP消费者。显示方向沿用已确定安装，原始数据不旋转。

Atom人工零参考与名义光轴仍未完成独立标定。Android接收匹配不是曝光同步，各设备时钟独立。能力与既有验证见[IMU报告](ATOM_IMU_PITCH_20261012.md)、[主机用法](ATOM_IMU_HOST.md)、[采集完整性](REAL_LABELED_CAPTURE_PILOT_20261011.md)。

## 冻结联合栈与新数据

[细结构开发](REAL_THIN_STRUCTURE_DEV_20261013.md)的深度脊定位17/22；[局部跟踪/几何](REAL_THIN_LOCAL_GEOMETRY_DEV_20261013.md)完成：按独立RGB观测链累计使持续目标9→13/22、侧杆0→3/6，干扰持续候选87→116/12张。保留跟踪机制；盒内稳定目标仍3/22，出现近域外估计与台阶误入BODY，几何仅调试，不接入报警。[距离可信度](REAL_THIN_CONFIDENCE_DEV_20261013.md)确认异常近距涉及锚点外推；ToF硬互证丢掉全部17张已定位目标，局部跨度也误挡真实杆。保留全部视觉候选，距离异常/未验证状态独立输出，不采用硬门槛。

[10月13日尺度/时序开发比较](REAL_FUSION_REFINEMENT_DEV_20261013.md)：已消费18段1677就绪帧，保留稳健ToF拟合成功时才替换尺度的开发候选。低头细杆any65→67/79、首报提前0.568秒；18段any计数未损，但桌/横杆BODY少15个query帧，未知是否正确。[第二轮锚点/运动/地面比较](REAL_FUSION_REFINEMENT_V2_DEV_20261013.md)未超过该候选：双mode增地面报警、运动搬运无any收益、地面回退增柱但损横杆，均不采用。冻结v1/v2及App保持；下一优先补固定安装位置/距离与去杆配对参考，独立头戴检验仍待录制。

当前实机研究保留[俯角/地面/噪声联合栈](REAL_POSE_FLOOR_NOISE_DR_DEV_20261012.md)，现已接入独立手机融合实验。训练噪声原口径偏小10倍；[独立消融](REAL_NOISE_ABLATION_READOUT_DEV_20261012.md)又确认校准场景覆盖同样关键。仅加训练噪声仍使真实空场强报；默认删bin1与单纯降低BODY强阈值均未采用。较安静S阈值1.499677777仅可选。

[新手机18段回放](REAL_PHONE_NEW_BATCH_DEV_20261012.md)已完成：6746 RGB、1818 CNH、36114 IMU，6872文件手机hash一致，冻结模型/阈值24姿态条件。低头大厅地面HEAD/BODY强档0/98，拟合高度/姿态下仍0、轻档0。椅/桌/坐人HEAD全强，走近墙/柱仍有明显强档。中央竖杆主IMU口径HEAD强4/48、0/79；低头杆拟合高度后BODY11/79。弱视觉标签、杆/底物混杂及安装高度未知限制解释。

新三低头段未复现旧右下zone约314mm强回波，bin1/bin0中位约0.012–0.014，旧0.754；[近距补偿探索](REAL_NEAR_RESIDUAL_DEV_20261012.md)两简单配方仅部分改善、增加模拟通知，均不采用。旧近杆轻档依赖远壳输入，不能直接认作近杆回波。

瓶颈：轻提醒可持续100%，可选阈值也未整体解决；近竖杆强档仍弱。下一优先按[固定安装配对采集单](REAL_NEAR_GEOMETRY_DEV_20261012.md)记录空/内杆/外杆及去杆差分，实测高度/位置并排查近距遮挡。本轮18段为新录制冻结Development描述，不是接触或安全确认。

[姿态门控＋RGB可行性](REAL_POSE_GATE_RGB_FEASIBILITY_DEV_20261012.md)：俯角>+10°或|横滚|>25°不出提醒（仅去掉两段抬头天花板的98/98轻档，障碍段不变，已接入live引擎）；UniDepth无内参即可清楚看到ToF漏掉的细竖杆，但尺度/视场不可靠（地面估计1.05–1.84m），先用屏幕棋盘格标定Atom相机再做RGB读出与融合。

[Atom标定＋RGB或ToF融合](REAL_RGB_TOF_FUSION_DEV_20261012.md)：屏幕棋盘格标定（RMS0.37px），ToF对齐得相机←ToF旋转（偏航10.5°/俯仰14.5°），UniDepth已知内参＋ToF尺度锚定；RGB在约1.5m看到ToF漏的正前细竖杆（BODY 72%/95%，像素宽6–12，非坐墩），空通道0；ToF强档或RGB融合首次让细杆、门柱、横杆都报。开发描述，待新录制冻结检验与手机端小模型。

[小深度模型对比](REAL_RGB_SMALL_MODEL_DEV_20261012.md)：手机候选为DAV2-Small相对（官方Apache-2.0，与DEPTHOR内嵌逐值一致）＋ToF/地面锚定518px：正前杆BODY67%/92%（UniDepth-L 72/95），空通道0–3%，笔记本56ms；横杆HEAD仅约11%需ToF补；DAV2-S公制Hypersim淘汰（0/18%），UniDepth-S 33/77%。手机实测（8 Gen 3）686×518 HTP 120ms/GPU 190ms，HTP输出与FP32相关0.99999；在线头高不可靠，改开机低头ToF标定。现已接入独立手机融合实验；下一步新录制冻结检验。

[冻结检验v1/v2](FROZEN_FUSION_EVAL_V1_PROTOCOL.md)：2026-10-12在新头戴录制前冻结（ToF强档或DAV2-S RGB，v2加2/3帧时序确认），明日录制后只跑脚本；蒸馏版DAV2-S使竖杆变差（67→33%），未采用。

[RGB读出门槛旁证](REAL_RGB_READOUT_CALIBRATION_DEV_20261012.md)：ARKitScenes真实帧（LiDAR仿真ToF锚定DAV2-S）上，事先门槛150px≈8.3e-4sr的eval检出92.2%/误报4.8%，cal 98.9/3.0%，AUC0.96–0.999；无细杆，不改冻结门槛。

[直方图锚定与ToF互证](REAL_HIST_ANCHOR_CONFIRM_DEV_20261012.md)：前向模型直方图锚定在开阔场景不可靠（空通道BODY误报22–47%，竖杆44/69%＜固件＋地面61/92%），不采用；ToF互证率墙柱75–80%、家具50–100%、细杆12–35%，可作可信度分级而非细杆升级。

## 默认、使用与历史入口

模拟Development默认仍由[候选指针](../dtr-r0/nearfield/CNH_SIM_DEFAULT_CANDIDATE.json)和[合同加载器](host/local_signed_bias_default.py)管理；[v4零证据修复](LOCAL_SIGNED_BIAS_CONTRACT_DEV_20261011.md)与原M3/5格及失败保持。实机候选不自动替换模拟或App。

[电脑M3演示](CNH_M3_LIVE_DEMO_20261010.md)用于场景变化模型响应；[无线协议](WIRELESS_PROTOCOL.md)、[硬件账本](README.md)、[H3固件](CNH_H3_WIFI_DEV_20261011.md)与[早期历史](CURRENT_HISTORY_20260924.md)保留工程身份与恢复入口。[运行记录](RUNS.md)保存本轮及旧轮成本、输出与结果。物理标定、真实漏报/误报对照和导航/听感验收仍待完成。
