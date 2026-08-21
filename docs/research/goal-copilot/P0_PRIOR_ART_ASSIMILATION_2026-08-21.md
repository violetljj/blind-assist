# P0 Prior-Art Assimilation

状态：`COMPLETE / 40_SEARCH_RESULTS_REVIEWED / 4_WORKSTREAMS / TASK_POSITIONING_REVISED / SILVER_B_CONTINUES / NO_SCIENTIFIC_VERDICT`

日期：2026-08-21

## 结论先行

BlindAssist 的用户问题不是新问题，组件任务也不是空白：BLV last-few-meters、named-POI entrance
navigation、embodied referring expression、主动询问和视觉闭环控制都已有强相关工作。尤其是 2026 年的
BridgeNav 与 ABot-N1/ABotN-POIBench 已把“命名 POI → 物理入口 → 最后几米到达”推进到数据、模型和
closed-loop benchmark；因此不得再把“入口级 POI 导航”本身写成 BlindAssist 的首创。

当前仍可能成立、但尚未被本轮穷尽性证明的结合点更窄：

> 面向 BLV 用户，在未预扫描的开放街景中，把自然语言目标解析为一个可为 `UNIQUE / SET_VALUED /
> AMBIGUOUS` 的现实入口 referent 集；用实时 RGB 与不完整地图形成可审计证据，在不确定时拒绝强猜，
> 然后把已锁定实例交给最后几米持续引导。

这是 prospective research positioning，不是 novelty claim。要形成论文首创主张，仍需更广的系统综述与
可复核 benchmark 对照。

## 逐项吸收

| 工作 | 已建立什么 | BlindAssist 可直接借什么 | 不能直接迁移的边界 |
|---|---|---|---|
| [Closing the Gap](https://www.microsoft.com/en-us/research/publication/closing-the-gap-designing-for-the-last-few-meters-wayfinding-problem-for-people-with-visual-impairments/) | 22 人 formative study、13 人 Landmark AI design probe；明确 doorway、连续 storefront、定位粒度与 camera aiming 是 BLV last-few-meters 的真实问题 | 用户问题定义；landmark/sign/place 三类信息；把 O&M 技能、残余视力与情境纳入交互设计 | Landmark AI 是 design probe，place matching 含 Wizard-of-Oz；不是可部署 grounding/navigation baseline |
| [NaviNote](https://arxiv.org/abs/2603.08837) | 预扫描区域内 VPS 亚米级定位、语音/agentic 空间注释；16 人对照中成功到达 14 人，对照 6 人 | 精确定位后的 turn-by-turn/audio compass；空间注释与 BLV voice-first 设计；后续用户研究指标 | 目标和空间 anchor 预先存在，依赖 pre-scanned VPS；不解决开放世界入口发现或多 referent truth |
| [STEP Navigation](https://apps.apple.com/tw/app/step-navigation/id6448701994) / [SmartAIs](https://www.smartais.de/en) | 产品明确面向入口、站点和 last-meter guidance | 产品需求、hands-free/振动/骨传导反馈、地图请求与部署流程作为竞品约束 | 当前公开材料主要是开发者/厂商声明；STEP 依赖 virtual anchors/映射库且公开地图规模有限，未找到可与论文 benchmark 等价的独立结果 |
| [REVERIE](https://arxiv.org/abs/1904.10151) / [Layout-Aware Dreamer](https://arxiv.org/abs/2212.00171) | 高层自然语言 → 未见室内环境导航 → remote referred object grounding；布局/目的地先验帮助搜索 | task decomposition；navigation success 与 object grounding 分开；语言、布局、object pointer 联合建模 | Matterport3D 离散室内图与标注 object bbox；通常给定可判定 referent，不覆盖 BLV 街景、入口角色歧义或安全引导 |
| [To Ask or Not to Ask?](https://openaccess.thecvf.com/content/WACV2025/html/Abraham_To_Ask_or_Not_to_Ask_Detecting_Absence_of_Information_WACV_2025_paper.html) | 学习检测 VLN instruction 信息缺失，明确区分“何时问”与“问什么”；用 precision/recall balance 评估过度谨慎和过度自信 | 未来 clarification trigger 的定义；把 ask/guess 作为 calibration 问题 | 只解决 when-to-ask，不生成澄清问题；不是入口 grounding 或 BLV interaction 验证 |
| [DialFRED](https://arxiv.org/abs/2202.13330) / [ELBA](https://openaccess.thecvf.com/content/WACV2025/papers/Shen_ELBA_Learning_by_Asking_for_Embodied_Visual_Navigation_and_Task_WACV_2025_paper.pdf) | agent 可主动向 oracle 提问；DialFRED 有约 53K QA，ELBA 学 when/what to ask 并在 TEACh 上改进 task completion | questioner/performer 分层；澄清动作与物理动作分离；ask cost/utility 的实验框架 | 虚拟室内 manipulation/nav、oracle 可访问 GT；不能把 oracle 答案或 simulator success 直接当 BLV 开放街景证据 |
| [BridgeNav](https://arxiv.org/abs/2602.06427) | 定义 prior-free outdoor-to-indoor instruction navigation；55K street-view seed、生成轨迹、20K door-refined manual boxes；用 latent intention 与 optical-flow dynamic perception 做入口级 waypoint prediction | `far=target visibility / mid=signage / near=entrance` 的阶段化注意力；door refinement；SR@0.1/0.2/0.3m 与 trajectory deviation；真实机器人 qualitative check | 主要 benchmark 来自生成视频/轨迹；不是 BLV；短命令通常隐含唯一 POI，不处理 set-valued reference 或 clarification；论文将 BridgeNav 描述为 open-loop waypoint 评估 |
| [ABot-N1 / ABotN-POIBench](https://arxiv.org/abs/2607.10383) | slow reasoner 输出 pixel goal、fast expert 输出连续 waypoint；POIBench 有 11 个商业区域、163 POI、物理 entrance frame、3DGS closed-loop，指标 SR&lt;2m/SPL；论文报告 77.3/72.6 | 当前最直接 benchmark/architecture 对照；pixel-grounded slow/fast interface；entrance frame 而非单点；SR/SPL 与 failure attribution | 依赖高保真 3DGS、人工 walkability/entrance annotations 和预构建 benchmark；2026 preprint，尚缺独立复现；named POI/entrance truth 已给定，不覆盖含糊角色、多合法入口、BLV 手机交互 |
| [Project Guideline](https://github.com/google-research/project-guideline) | 开源 Android/C++：ARCore pose、perception→world map、多帧聚合、obstacle map、control、空间音频、tracking loss→STOP、simulator/replay | 目标锁定后的 runtime 架构与 fail-closed STOP；世界坐标聚合、控制/反馈解耦、仿真与日志 | 受控紫色 guideline、特定设备/佩戴方式且要求 sighted spotter；任务不是目标发现或 referential grounding |

## BlindAssist 模块映射

| BlindAssist 层 | 直接借鉴 | 当前保留的自有问题 |
|---|---|---|
| P0 goal/reference truth | REVERIE 的 remote grounding；To Ask/DialFRED/ELBA 的 ambiguity/ask 分层 | `UNIQUE / SET_VALUED / AMBIGUOUS` truth；开放街景、入口角色、证据不足时 abstain |
| P0 visual/relational grounding | BridgeNav 的阶段化视觉重点；ABot 的 named-POI entrance pixel goal；Closing the Gap 的 sign/place/landmark 信息 | 不预设 entrance bbox/3DGS；RGB proposal + POI/map/facade/entrance relation 的可审计 evidence |
| P1 target persistence | ABot slow pixel goal→fast controller interface；Project Guideline 世界坐标聚合 | 从 weak/open-world detection 锁定一个现实入口后，保持 identity 且发现误锁 |
| P2 approach/control | NaviNote audio compass；ABot SR/SPL；Project Guideline control/STOP/audio | BLV 手机/眼镜低延迟反馈、partial observability、动态障碍和 fail-closed guidance |
| Interaction | NaviNote voice-first；To Ask、DialFRED、ELBA | 未来再实现最小 clarification policy；当前只保留 ambiguity truth 和允许 abstain |

## 可借 task 与 metric

立即采用为设计参照，但不直接混入当前 Silver-B 性能结论：

- task 分层：`referent resolution → candidate availability → selection given available → approach/arrival`；
- grounding：命中任一 valid referent，而不是单一预设 bbox；Provider 与 Brain 分母分开；
- ambiguity：ask/abstain trigger 的 precision、recall 与不必要询问率；
- arrival：后续使用 entrance frame、SR@distance、SPL/路径效率、collision/STOP accounting；
- BLV interaction：到达成功之外记录 camera aiming burden、mental demand、frustration、perceived effectiveness；
- system：tracking loss、证据过期、目标冲突时显式 STOP/ABSTAIN，不让历史坐标静默继续控制。

当前不能采用：把 BridgeNav/ABot 的生成数据、3DGS entrance annotation、oracle dialogue 或 NaviNote VPS
直接当 BlindAssist truth；把 STEP/SmartAIs 产品文案当独立性能证据；把 Project Guideline 的受控路线安全性
外推到开放街景入口导航。

## 对当前路线的直接影响

1. Silver-B 继续，不新增模型、不暂停 materialization；它仍是 map+geometry weak candidate data，当前
   goal-reference truth 保持 `AMBIGUOUS`，不能生成 detector recall 或 exact Brain accuracy。
2. 不再把 P0 描述为自创的通用 “Goal Grounding”。论文定位应进入
   `BLV Last-Few-Meters + Embodied Referring Expression + Interactive VLN + POI-Goal Navigation`。
3. ABotN-POIBench 是今后最直接算法 benchmark；BridgeNav 是最直接入口视觉/waypoint baseline；NaviNote
   与 Closing the Gap 是最直接用户问题和交互依据；Project Guideline 是 P1/P2 工程参考。
4. 当前最小下一步仍是扩大 Silver-B Development coverage，但 episode 必须保留 referent resolution，且分析
   名称改为 `map-geometry-conditioned candidate yield / weak ranking mechanics`，不叫 recall/accuracy。
5. 在任何 novelty statement 前，必须把 ABot-N1/POIBench、BridgeNav 与 BLV last-few-meters 系统列为直接
   related work，并通过更广检索验证“未预扫描开放街景 + set-valued reference + BLV”这一组合缺口。

## Claim ceiling

本轮只建立 prior-art mapping 与研究定位修正。它不证明相关工作穷尽、不证明 BlindAssist 首创、不证明
Silver-B utility、模型性能、导航成功、用户有效性或安全。产品页面只作为能力/部署声明，不与同行评审实验
等权。2026 preprint 的指标按作者报告记录，未独立复现。
