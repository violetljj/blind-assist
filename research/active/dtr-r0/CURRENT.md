ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-09。主线为盲杖互补前视感知，M3和A+LOCAL保留；不恢复历史动态研究。

## 当前：原生支持仍正，继续定位编码与池化响应

[完整输入](nearfield/CNH_VERTICAL_NATIVE_INPUT_DEV_20261009.md)：同受控hist重建6场景×K4×f6–13共192输入，+3°、ρ=.65长柱BOTH、4/10cm、同侧／镜像clear及1cm接触。128配对物理窗，近段64，每宽／关系16；无新采样、M3推理或训练。新CPU/GPU阶段各120秒内，投影阶段含失败11.451秒；1536旧指标bitwise，完整输入和修复／核验持久保留。

近段镜像BODY原生signed-log累计／当前差4cm+5.3033/+1.1942、10cm+7.0949/+1.2773，旧raw差−4.1428/−4.3418。整段预处理排序翻转未获支持，但10cm f10累计log差−.0715，不能说每窗都不翻。近段participation差两宽、两关系皆负，提供正支持集中线索；不是CNN偏好或网络因果。镜像左右布局差也不能当模型侧别偏置。

下一优先同192输入→冻结M3编码／公共BODY max与mean池化／输出头连续响应，先核对旧raw；无需再采样或投影，但新前向另列预算，NOT_RUN。不据此启动集中度门控或统一缩短last5；可补回量未知，候选须同配对同报警成本报事件救回／损失及联合clear格／段／clip。

[条件定位](nearfield/CNH_VERTICAL_RESPONSE_LOCALIZATION_DEV_20261009.md)两长柱BODY贡献gross前四窗负条件均值项77.84%，非净值或报警数比例。共同角度条件和未匹配项分开，缺对照非零效应；raw含past8，不能归为物理拖尾。[受控配对](nearfield/CNH_VERTICAL_CONTROLLED_PAIR_DEV_20261009.md)为人工Poisson耦合，非共享物理光子。主要远段86/116格成本仍未解决，压力排序不代替真实误差分布。

原M3/[5格候选](nearfield/CNH_BAR_FUSION_VERTICAL_DEV_20261009.md)不升级，ideal H543/B461各/688，clear46/4576。[类别排序](nearfield/CNH_BAR_CATEGORY_PRIORITY_DEV_20261009.md)暗4cm第二缺口BODY7/56，具体新机制另定。[同帧参照](nearfield/CNH_BAR_BACKGROUND_REFERENCE_DEV_20261009.md)停止，时间累积暂缓；有支持改善参照或增量的新依据可考虑诊断，不要求先证明。[旧门控](nearfield/CNH_BAR_BOUNDARY_CONTRAST_DEV_20261009.md)Dwin/Dmax不细化，裁剪／三query min不采用，不关闭不同机制。

## 停止与保留

[扰动训练](nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)原配方exact各损7超2，停止，不加seed/epoch/换分布；不同机制开放。L2、[480确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留；UNKNOWN/三态/query覆盖待验证。方位微调、稀疏空间/关联L3、方向/Nymeria位移接口、−19°及设备/City/保护test/新UE/硬件第二阶段暂缓；101/101、53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、固定有限背景、人工误差、loss筛选和相关K/帧下已消费Development，非实机、安全、确认。旧当前全文：Git b87b0010同路径。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。
