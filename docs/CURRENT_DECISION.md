# 当前研究决定

更新：2026-10-09。主线为盲杖互补前视感知，M3和A+LOCAL保留；不恢复历史动态研究。

## 当前：竖杆raw偏低与last5反转已分组，原M3不动

[条件定位](../research/active/dtr-r0/nearfield/CNH_VERTICAL_RESPONSE_LOCALIZATION_DEV_20261009.md)仅现有受控响应，CPU新60秒内完成：1024近段高度窗、5120lag、2048去重来源窗。无新采样/投影/推理/训练。+3°镜像1cm中，ρ=.65的4/10cm长柱BODY raw均值差已为−4.143/−4.342；两组贡献前四窗负条件均值项77.84%（gross负项，非净值或报警数比例）。19/128窗raw差正而last5差负仍保留，不是19个事件损失。

跨角度共同20条件，连接320对包含640旧高度窗；每角另12条件/192窗未匹配，合384。共同镜像1cm的last5差均值−3°+2.700、+3°+1.238；不能混用完整128窗均值，未匹配长柱BODY缺对照不当零效应。同contact镜像−同侧差等于两clear臂差，不是contact变化；角度对照仍有物理左右镜像差异。

+3°镜像1cm来源f6–9（2.09–1.61m）平均累计支持/raw更偏向clear，近段聚合raw差至f12转正。但两长柱BODY在f10–13当前/累计粗支持差各帧均值皆正、raw仍负；需查完整空间/高度布局，不能归为物理拖尾或认定M3忽略信号。raw包含past8，last5只是分数序列贡献。

下一优先两组长柱BODY的完整输入→raw响应，暂不从全组均值启动统一缩短last5。受控payload只存hist/粗支持/raw，完整voxel未保存；精确诊断需从同hist小范围重建投影，预算另列，不需新采样或默认重推M3，提案NOT_RUN。时间读出未被排除，可补回量未知；候选须同配对同报警成本报事件救回/损失及联合clear格/段/clip。

[受控配对](../research/active/dtr-r0/nearfield/CNH_VERTICAL_CONTROLLED_PAIR_DEV_20261009.md)48scene/K4共192新Poisson序列，保持原边际的人工耦合；八组缩放支持差均值/中位数正、内部正量份额负。不是M3增量证明；条件线性参考在FP16舍入前，背景遮挡变化保留。原M3/[5格候选](../research/active/dtr-r0/nearfield/CNH_BAR_FUSION_VERTICAL_DEV_20261009.md)不升级；ideal H543/B461各/688、clear46/4576格。主要远段86格问题仍未解决，竖杆优先级只限本批人工压力条件，真实误差分布未知。

[类别表](../research/active/dtr-r0/nearfield/CNH_BAR_CATEGORY_PRIORITY_DEV_20261009.md)暗4cm仍第二缺口BODY7/56，具体新机制另定；[单帧参照](../research/active/dtr-r0/nearfield/CNH_BAR_BACKGROUND_REFERENCE_DEV_20261009.md)停止、远峰容差null不能证明hist无远段信号，时间累积暂缓。有支持改善参照或增量的新依据可考虑累积诊断，不要求先证明。[旧门控](../research/active/dtr-r0/nearfield/CNH_BAR_BOUNDARY_CONTRAST_DEV_20261009.md)Dwin/Dmax不细化，裁剪/三query min不采用，不关闭不同机制。模拟AABB、固定有限背景、人工误差、loss筛选及相关K/高度/帧下已消费Development，非实机/安全/确认。

## 停止与保留

[扰动训练](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)原配方exact各损7超2，停止，不加seed/epoch/换分布；不同机制开放。L2、[480确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留；UNKNOWN/三态/query覆盖待验证。方位微调、稀疏空间/关联L3、方向/Nymeria位移接口、−19°及设备/City/保护test/新UE/硬件第二阶段暂缓；101/101、53ms属A+LOCAL，M3/CNH实机效果未建立。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。更新前全文：Git 5b750872同路径。
