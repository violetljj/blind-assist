# 冻结 RGB＋CNH 端到端评价：输入来源核查

2026-10-10；EXPLORE；输入身份检查，不产生算法成绩。

**当前可立即实施的是新受控 CNH 观测→读出→通知比较。完整 RGB＋CNH 在未用数据上的及时提醒主终点目前为 `NOT_EVALUABLE`：现有新 CNH 配方没有同步 RGB，现有真实 RGB-D 不含器件 CNH；旧成对来源已消费或缺空间、时钟、身体事件真值。** 不能把独立来源的 RGB 数字和 CNH 数字相加，也不能把 RGB-D 前向合成的直方图称为实机输入。

用户已授权冻结 both＋1帧通知合并、RGB affine 及 0.8m 以上候选职责，并与原 M3＋原5格比较。此文件只识别可用输入和缺口；具体校准、留出、成本工作点、提醒融合规则由本次执行计划冻结。0.8m 是候选职责边界，尚非已验证传感器交接距离。

## 已核查的来源

| 来源与身份 | RGB／CNH、参考实际具备什么 | 未用系统评价资格与缺口 |
| --- | --- | --- |
| [Counterfactual Development](CNH_COUNTERFACTUAL_DEV_20261009.md)，生成代码 [cnh_counterfactual_data_dev.py](cnh_counterfactual_data_dev.py)、基线代码 [cnh_counterfactual_baseline_dev.py](cnh_counterfactual_baseline_dev.py) | AABB／有限背景前向光子模拟；8×8×16 CNH、ambient、sensor、public_query；普通读出和 M3／5格。运行时为 normalized histories／transforms／length；geometry、category、labels 属作者／评价侧。代码没有 RGB 图像生成或相机帧接口。 | train、cal、validation 均已用于当前候选选择。新 photon seed 本身不建立场景独立；新场景需另记身份和几何差异。可继续生成不同场景的受控 ToF→通知输入，不能声称已完成 RGB＋ToF。 |
| [真实头动确认批](CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)，单位400000–400479 | HEADS-UP 一段真实轨迹驱动的模拟场景和 M3；不是真实 CNH 或成对 RGB 录制。 | 原确认身份保留，本次未读 payload 或 outcome。不能从“真实头动”改称真实同步传感器评价，亦不复用为本次未用数据。 |
| [3RScan／ARKit 冻结迁移](RGB_BODY_QUERY_TRANSFER_DEV_20261009.md)、[固定27子盒](RGB_BODY_QUERY_QUERY_LEVEL_DEV_20261009.md)、[TUM 输入](RGB_BODY_QUERY_INPUT_BASELINE_DEV_20261009.md) | 3RScan 原生 RGB／深度／K；ARKit raw Training/47333462 的同像素 RGB／LiDAR depth／逐帧 K，后续追加 capture；TUM registered RGB／Kinect depth，最近时间≤20ms。它们有真实米制 first-return 参考，缺值／遮挡保持 UNKNOWN。 | 这些 RGB 查询来源已经消费；不是身体绑定步行或 HEAD 细杆事件参考，不含 VL53L8CH／CX 的完整 CNH bin。以它们合成 CNH 只能得到明确假设下的受控观测，不能自动获得真实混合回波、遮挡后占用或独立实机真值。 |
| [巷道六场 Development](CNH_ALLEY_GENERATOR_20260924.md)、[原融合诊断](CNH_ALLEY_BOTTLENECK_DIAGNOSIS_20260925.md) | 三 train／三 dev 实际 UE 场地。旧欠曝 RGB 后按固定曝光修复；六份 rgb-tof-overlay.json 将新双目 RGB 与旧 observations.npz／targets.npz 同索引绑定。collection-overlay.json 记录960帧／1920张 RGB。 | 确有同场景 RGB＋模拟 ToF／参考入口，可适配缓存融合工程；全是已消费 Development，不能转为新场景确认。旧右目深度原件本地精简后未重新核验。需另检查该包 ToF 观测格式是否满足当前完整 bin 读出，不能仅凭“有 ToF”认定可直接用当前 both。 |
| [冻结164布局 test 准备度](CNH_FULL_TEST_READINESS_20260925.md) | 当前 plan 只是确定性空槽；实际场地、源 mesh、几何、采集 spec 均未完整编写。三张候选巷道 test 地图不是已采 test。 | `BLOCKED_REALISTIC_164_NOT_AUTHORED`。本次未访问保护 test 帧、标签、预测或结果，不把作者预分区／材质探针视为就绪数据。 |
| [本地并行录制20260921](../../hardware-bringup/LOCAL_PAIR_20260921.md)，[源码证据索引](../../hardware-bringup/pair-evidence-index.json) | Atom RGB＋XIAO 4×4×24 CNH，baseline／motion／motion-cued 三段。原始字节、JPEG、seq、状态、接收时间和 SHA 索引具备。补录204 CNH帧，188位于可配对窗口。 | 主机接收近邻中位15ms、最大47ms仅为记录配对；非曝光同步。相机-ToF外参、尺量距离、完整物体进出循环、身体走廊／HEAD细杆事件未验证。4×4×24 与当前8×8×16形状不同。已消费连接／响应检查，不能承担此次未用事件主终点。 |
| [20260927 H3 真实回放](../../hardware-bringup/CNH_H3_LEARNED_DEMO_20260928.md)、[录制校验](../../hardware-bringup/CNH_H3_CAPTURE_VALIDATION_20260928.md) | simple-20260927T164234Z-cc4d0e 的背景／物体／运动／恢复背景；8×8×16 CNH＋相机。历史回放146/146相机接收配对，P95差63ms。 | 无可用相机-ToF物理配准、逐query身体／HEAD细杆真值；背景经验归一化不等于 sim-floor。真实 replay 缺 bias、ambient、T_Q_tof 所需显式字段。旧模型已回放并观察结果，不能恢复为未用集。回放吞吐非事件端到端延迟。 |
| [SANPO-Synthetic 预检](CNH_PROPOSAL_SANPO_PREFLIGHT_20260929.md) | 3000帧／60session 原生 RGB-D、K、部分位姿；原预检支持3m内可见深度入口，未合成 CNH、未接视觉模型。 | 已消费 BA-NFO／looming Development；深度 optical-Z／径向及世界位姿方向契约尚需闭合，物理布局独立性 UNKNOWN。可提出同心、可见表面、直接漫反射的模拟路线，不能把旧test名字恢复为未用验证。 |

## 本次可执行路线与结果解释

1. **先完成新受控 ToF→通知比较。** 冻结现有 both（各训练 seed 单独报告，不能观察新结果后选 seed）、原 M3／原5格及同一通知器。预先声明新几何场景、cal／evaluation分区及身份排除范围，再生成新观测。按 cal 固定工作点，评价 HEAD 横向细障碍及时救／损、首次提醒提前／推迟、BODY保留，以及强／轻 pass、clear 通知成本。既有 cal 成本相同不保证 evaluation 等成本，实际不匹配须如实报告。
2. **RGB affine 保留为冻结模块，但不为新 CNH 世界捏造预测。** 当前纯 AABB CNH 不具备 RGB 输入，完整 RGB＋ToF 表中应写 `NOT_EVALUABLE`；RGB-alone 查询结果独立保留，不能替代系统提醒成绩。真实 first-return 只能评分有可观测参考的射线／子盒；整盒 FREE、身体事件和提前量缺口不能补成负样本。
3. **完整系统后续需要同一事件的 RGB、CNH、公开标定／时钟与评价侧真值。** 若采用 UE／RGB-D→CNH 模拟，需要新而可追溯的场景、同事件图像、明确视场／外参和 depth轴、生成假设及固定融合规则；以“受控模拟端到端”命名。若采用实机，需要同装置标定、CNH特征标定／缺值契约、可核对 HEAD细杆接触／擦边／空闲事件和参考距离；接收时间近邻与人工分段意图不能单独替代这些参考。

## 核查边界

本次读取已跟踪研究文档、来源索引及生成／基线代码，不扫描 artifact 全树，不读保护outcome，不下载、不推理、不重新采集。上述数字来自本次实际查看的历史记录，属于对应旧run口径，并非本次重新测量；没有对 payload hash 做新逐文件复验。旧欠曝结论已经由后续RGB修复记录覆盖，不能继续说“巷道没有可用RGB”。所有 payload 延用 canonical artifacts.local；本文件不修改源数据、共享 CURRENT／RUNS 或冻结旧协议。
