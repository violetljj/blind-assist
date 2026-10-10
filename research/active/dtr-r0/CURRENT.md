# 前视障碍感知：当前状态

更新：2026-10-10。盲杖互补前视感知主线；原M3、5格局部融合、L2与A+LOCAL保留。

RGB [四官方主干横比](nearfield/RGB_DEPTH_BACKBONE_COMPARE_DEV_20261010.md)完成各496帧，冻结query/56train affine/三折与304cal不调阈值。当前冻结系统近带交ToF、RGB职责≥0.8m；中远带实验首选DAV2 Indoor Large原始米制，三ARK见证440/554、FREE25/199 vs原DepthPro affine354/554、27/199（救138损52），但far292/323 vs319/323、65中远FREE21/103 vs10/103，逐capture无成本替换未获支持。UniDepth affine见证最多468/554但FREE32/199。近带米制误差明显恢复，Metric3D/MoGe raw的全局cut>0.25可结构排除近见证；60唯一增益DAV raw6/11伴FREE1/133，65近POS0使强信号N/E，职责分工不作单目近带原理否定。原DepthPro affine/全部旧失败与stop保留；完整同步RGB/CNH仍NOT_EVALUABLE。

## 当前：信号与读出漏检拆解

头朝向＝身体朝向＝行进方向，HEAD/BODY共向query；方向估计/PDR暂不作前置。明显风险强提醒，接触/擦边难分可轻提醒；分级只用运行时证据，强轻/pass/clear实际通知成本分列。共向不替代外参/时钟核验。

[CNH_BASELINE_MISS_DECOMPOSITION_DEV_20261010](nearfield/CNH_BASELINE_MISS_DECOMPOSITION_DEV_20261010.md)完成已消费E2E hold与旧ideal validation解释性复用：原5格新HEAD未及时98/256，高SNR≥4为38/98、低<2为32/98、中间28；BODY140/256中21/48/71。旧HEAD86/384中73/0/13，BODY140/384中60/16/64。K聚类到物理scene，各新128/旧96scene；不是新确认。

新高SNR漏检最多为BODY sign_edge19/64、HEAD sign_edge14/64、protrusion11/64、horizontal10/64、vertical3/64。HEAD全部38仅1帧达到4，短促回波不证明同成本可检出。暗rho .19的1.7cm横/竖杆和BODY暗2.5cm边缘列为当前模拟弱信号边界，中间档保留。下一建议短促contact窗证据保留、局部目标/背景关联，pass/clear负监督与通知成本入损失；仅建议，未训练。

both/955的pass通知37→286来自3.5/7.5cm间隙，有通知clip37→193；所有pass本来都<10cm，无远pass对照。clear在13.5/18.5cm仍28→49通知，不能全以擦边解释。完整交叉表/逐事件ledger及一次独立复算PASS；CPU保守323/1500s、GPU期望50/600s，无新采噪/前向/阈值选择/保护集访问。

## 冻结结果与停止保留

[CNH_RGB_FROZEN_E2E_RESULTS_20261010](nearfield/CNH_RGB_FROZEN_E2E_RESULTS_20261010.md)保留：原5格HEAD158/BODY116各/256；同通知成本匹配主955为158/119、薄HEAD均2/32；原both202/154，但clear/pass49/286。固定配方不在hold调参，不升级App。细杆期望与四类读出负结果沿用[CNH_HEAD_THIN_SIGNAL_ORACLE_DEV_20261010](nearfield/CNH_HEAD_THIN_SIGNAL_ORACLE_DEV_20261010.md)及[CNH_SPARSE_RAY_TRACK_DEV_20261010](nearfield/CNH_SPARSE_RAY_TRACK_DEV_20261010.md)所链接原报告；失败/停止规则按各run保留，不否定其它机制。

方向/步行历史、旧扰动/T/T2、保护480/test及Nymeria异常按原记录保留；设备/City/新UE/硬件第二阶段不由本诊断重启。A+LOCAL的101/101与53ms不属M3成绩。完整同步RGB+CNH仍NOT_EVALUABLE。模拟oracle、有限AABB、相关K和消费复用不证明实机、安全或硬件极限。

更新前全文保留于Git `d0c52ce2` 同路径及本run `integration_before/`。路线日志见[RUNS](RUNS.md)。
ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED
