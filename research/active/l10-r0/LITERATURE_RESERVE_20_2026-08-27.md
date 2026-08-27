# L10-R0 十米副驾：追加 20 篇论文知识储备（2026-08-27）

状态：`LITERATURE_RESERVE / NO_ALGORITHM_AUTHORIZATION`

## 结论先行

本轮用四个互补检索方向审阅了 447 个结果槽位：视频场景文字、长期跟踪与缺席重获、主动补观测与标识导航、终点/到达/辅助交互。结果按题名、DOI 和论文版本去重，并排除了此前已经精读的 40 篇，以及当前 L10 README 已引用的 GoMatching、QueryNLT、AECNav。经过第一轮机制过滤后保留 58 个候选，最终留下下面 20 篇。

这 20 篇共同指向一条比“继续调 matcher”更有信息量的链：

`可读性与不确定性 -> 词级几何 -> 缺陷特定的补观测/靠近 -> 新鲜语义身份 -> 非权威连续性载体 -> 目标相对位姿与可见性 -> 用户可确认的交接`

没有任何一篇论文单独证明了这条链，更没有论文证明 L10 已具备真实十米导航、功能入口、用户完成或安全能力。它们是组件假设和评测设计储备，不是新路线授权。

## A. 先改文字信息源与词级几何（1–6）

| # | 论文 | 带来的新信息 | 对 L10 的最小落点 | 证明边界 | 判定 |
|---:|---|---|---|---|---|
| 1 | [DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting](https://doi.org/10.1109/cvpr52729.2023.01854), Maoyuan Ye et al., CVPR 2023 | 用有序显式点同时解码字符序列、中心线和边界，保留任意形状文字的内部几何。 | 在固定 ArTVideo 帧上只替换文字几何源，比较词级转向 bearing 与 merged-line center；不改身份状态机。 | 静态图像证据；不是 RapidOCR/CTC 的即插即用替代，也不解决时间重获。 | `PRIORITY` |
| 2 | [Character Region Awareness for Text Detection](https://openaccess.thecvf.com/content_CVPR_2019/html/Baek_Character_Region_Awareness_for_Text_Detection_CVPR_2019_paper.html), Youngmin Baek et al., CVPR 2019 | CRAFT 预测字符区域与字符间 affinity，可把整行候选拆成字符组/词组。 | 把它作为纯几何 carrier source：先拆 merged line，再由现有 OCR 提供词义；不得让 affinity 生成身份。 | 只做检测/分组；相邻词仍可能错误粘连，也没有 lexical identity。 | `PRIORITY` |
| 3 | [Conformal Predictors for Efficient Video Text Spotting](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_1118/paper.pdf), Ben Tanfous, Subhra Mukherjee, Neil M. Robertson, BMVC 2025 | 对识别字符与框位置做 conformal uncertainty，并利用轨迹对齐传播高置信字符。 | 在 source-disjoint 校准集上产生“词集合/框区间/UNKNOWN”，以 coverage、错认和错误身份 admission 联合评估。 | 建立的是覆盖保证与 VTS 增益，不是 physical identity；校准若换域仍可能失效。 | `PRIORITY` |
| 4 | [TraRA: Trajectory-level Recognition Aggregation for Video Text Spotting in Urban Surveillance](https://arxiv.org/html/2606.07161), Duc Tri Tran et al., 2026 preprint | 先按视觉/时间一致性切分错误轨迹，再用跨帧碎片与语言上下文重建完整词。 | 对数字/词被遮挡或分帧出现的轨迹做一次冻结聚合；同时把“恢复正确词”和“语言先验编造词”作为对称端点。 | 新预印本且依赖上游轨迹；VLM 聚合可能把不可读证据补成看似合理的错误文本。 | `PRIORITY` |
| 5 | [Towards Accurate Video Text Spotting with Text-wise Semantic Reasoning](https://www.ijcai.org/proceedings/2023/206), Xinyan Zu et al., IJCAI 2023 | VLSpotter把文字专用超分、词内语言修正与同帧文字间语义推理组合起来。 | 优先只检验文字超分这个新像素源：冻结 OCR 与控制，比较精确 token 恢复和错误 lexical admission。 | 语言/场景语义可“纠正”掉关键房间号；不能直接作为身份权威。 | `PRIORITY` |
| 6 | [On Calibration of Scene-Text Recognition Models](https://doi.org/10.1007/978-3-031-25069-9_18), Ron Slossberg et al., ECCV Workshop 2022 / proceedings 2023 | 说明字符级校准不必然带来词级校准，识别 admission 应在完整序列层校准。 | 将 L10 的语义 admission 从平均字符分数改为可验证的整词校准问题；先做只读可靠性图与选择性风险。 | 主要验证 attention STR，不直接覆盖 RapidOCR CTC 或新场景分布漂移。 | `PRIORITY` |

## B. 把身份、连续性和缺席拆开（7–8）

| # | 论文 | 带来的新信息 | 对 L10 的最小落点 | 证明边界 | 判定 |
|---:|---|---|---|---|---|
| 7 | [Long-term Tracking in the Wild: a Benchmark](https://openaccess.thecvf.com/content_ECCV_2018/papers/Efstratios_Gavves_Long-term_Tracking_in_ECCV_2018_paper.pdf), Jack Valmadre et al., ECCV 2018 | OxUvA 把“目标是否存在”和“存在时位置是否正确”分开计分。 | 固定输出 `PRESENT / ABSENT / UNKNOWN`，位置与 bearing 只在 presence 有权威时计分；false-present continuity hop 单列。 | 这是评测合同，不是算法；首帧框也不提供公开语义身份。 | `PRIORITY` |
| 8 | [Find First, Track Next: Decoupling Identification and Propagation in Referring Video Object Segmentation](https://openaccess.thecvf.com/content/ICCV2025W/LSVOS/html/Cho_Find_First_Track_Next_Decoupling_Identification_and_Propagation_in_Referring_ICCVW_2025_paper.html), Suhwan Cho et al., ICCV Workshop 2025 | 先选择有可靠语义对齐的 reference frame，再由独立 propagation 模块维持连续性。 | 保持“新鲜 OCR 建立身份，视觉 carrier 只传播”；重获必须重新经历 identify，而不是 appearance-only handoff。 | 论文是离线整段视频选择，语言对齐也不是精确 OCR 身份；没有原生 absent contract。 | `PRIORITY` |

## C. 缺陷特定的补观测与标识理解（9–13）

| # | 论文 | 带来的新信息 | 对 L10 的最小落点 | 证明边界 | 判定 |
|---:|---|---|---|---|---|
| 9 | [Active text perception for mobile robots](https://eprints.qut.edu.au/57664), Martin Wyss, Peter Corke, ICRA manuscript 2012/2013 | 宽视野扫描发现稳定文字候选，再递归云台居中/变焦；报告了文字识别增益。 | 定义 `TEXT_TOO_SMALL -> CENTER / ZOOM` 的确定性补观测基线，替代无差别 expected-information-gain 扫描。 | 云台室内机器人、静态文字；没有目标语义绑定、身体靠近、到达或交接。 | `PRIORITY` |
| 10 | [Signage-Aware Exploration in Open World using Venue Maps](https://arxiv.org/html/2410.10143), Chang Chen et al., IEEE RA-L 2025 | 用场馆地图名称先验、文字实例检索、2D→3D 多视角融合，并显式靠近/调姿以提高标牌可读性。 | 最接近 L10 的 source successor：只有在“候选标牌存在但不可读”时请求靠近或调姿，身份仍由新鲜文字确认。 | 商场机器人与场馆地图设定；标牌识别和搜索效率不等于入口、metric arrival 或用户完成。 | `PRIORITY` |
| 11 | [Next Best View for Text Detection and Recognition in Port Monitoring Unmanned Aerial Vehicles](https://www.inf.uni-hamburg.de/en/inst/ab/cv/media/guelsoylu-etal-igd-grc2026.pdf), Emre Gülsoylu, Niklas Fiedler, Simone Frintrop, GRC 2026 | 用视角、距离、视线遮挡与运动代价构造文字 legibility utility。 | 把通用 EIG 改成可解释的 deficit-specific 观测策略：预测哪一个小动作最可能让目标文字变得可读。 | 无人机/集装箱 ID；legibility utility 未证明适用于手持相机、步行控制或自然室内标牌。 | `RESERVE` |
| 12 | [SignScene: Visual Sign Grounding for Mapless Navigation](https://arxiv.org/html/2602.12686), Nicky Zimmerman et al., 2026 preprint | 把复杂标识上的语义指令映射到当前局部 3D 场景元素与动作。 | 区分“写着目标名的终点标牌”和“指向远方目标的方向标牌”，防止读到文字就误报到达。 | 主要证明 sign grounding；不改善可读性，也不验证命名目的地的最终交接。 | `PRIORITY` |
| 13 | [Sign Language: Towards Sign Understanding for Robot Autonomy](https://arxiv.org/html/2506.02556), Ayush Kumar Agrawal et al., 2025 preprint | 提供真实公共空间标识集，并区分 directional cue 与 locational cue。 | 作为 L10 标牌 source/evaluator taxonomy：同样读到目标词，只有 locational cue 才可能支持近场终点判断。 | 静态图像理解；没有主动观测、控制、到达或用户研究。 | `RESERVE` |

## D. 从“看见目标”走到真实终点与交接（14–20）

| # | 论文 | 带来的新信息 | 对 L10 的最小落点 | 证明边界 | 判定 |
|---:|---|---|---|---|---|
| 14 | [TextInPlace: Indoor Visual Place Recognition in Repetitive Structures with Scene Text Spotting and Verification](https://arxiv.org/html/2503.06501), Huaqi Tao et al., 2025 preprint | 外观分支做粗检索，文字分支筛选 discriminative text 并重排，专门对付重复室内结构。 | 在相似走廊/相邻门牌中把 appearance 保留为 proposal，把辨识性文字保留为 verification；加入显式 NONE。 | VPR/place retrieval，不证明同一功能入口、沿路控制、到达或用户完成。 | `PRIORITY` |
| 15 | [TextSLAM: Visual SLAM With Semantic Planar Text Features](https://arxiv.org/html/2305.10029), Boying Li et al., IEEE TPAMI 2024 | 把文字建模为带语义的平面地标，并联合纹理、平面几何与在线语义更新形成 3D text map。 | 身份绑定之后，用文字平面估计相对姿态、朝向和 stand-off；比较它与大框高度 completion proxy 的误停。 | 是建图/定位工作；没有主动补观测、目标靠近、功能入口或用户完成证据。 | `PRIORITY` |
| 16 | [From Region Arrival to Instance-Level Grounding in Vision-and-Language Navigation](https://arxiv.org/html/2607.03792), Xiangyu Shi et al., 2026 preprint | 把 proximity precision、目标可见性、终帧 grounding 和短程 final approach 分开。 | L10 终点至少拆成“位置接近、目标仍可见且正确、朝向/终帧可用”，不再让 centered-large-box 代表全部完成。 | REVERIE 派生模拟 VLN；没有 metric entrance aperture、真机控制或用户交接。 | `PRIORITY` |
| 17 | [Aim My Robot: Precision Local Navigation to Any Object](https://arxiv.org/html/2411.14770), Xiangyun Meng et al., IEEE RA-L 2025 | 控制器显式到达目标相对位姿，包含距离、横向偏移与朝向，并有 sim-to-real 验证。 | 只在身份冻结后作为 `WHERE/WHEN` 控制参考：把目标相对姿态作为输出，不从 box size 暗推完成。 | 假设对象与参考位姿已经成立；不解决可读名称身份、入口 affordance 或用户完成。 | `PRIORITY` |
| 18 | [Closing the Gap: Designing for the Last-Few-Meters Wayfinding Problem for People with Visual Impairments](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/07/LandmarkAIASSETS.pdf), Manaswi Saha et al., ASSETS 2019 | 通过视障参与者研究明确“到达附近”与找到准确店面/入口之间的 last-few-meters 缺口。 | 用它定义产品端点和反馈：算法的“near”不能替代找到可行动入口，更不能替代用户确认。 | 这是需求与设计证据，不是自主 metric-arrival 算法或完成 verifier。 | `PRIORITY` |
| 19 | [Detect and Approach: Close-Range Navigation Support for People with Blindness and Low Vision](https://export.arxiv.org/pdf/2208.08477v1.pdf), Yu Hao et al., ECCV Workshop 2022 | 单目视频先做目标 3D 定位，再持续估计用户轨迹并修正靠近路径。 | 可作为 `find -> approach -> correct drift` 的技术上界，并要求把身份、路径修正和最终交接分别计分。 | 受控目标实验；没有可读标牌身份权威、入口 affordance 或用户完成研究。 | `PRIORITY` |
| 20 | [PathFinder: Designing a Map-less Navigation System for Blind People in Unfamiliar Buildings](https://wotipati.github.io/projects/other_papers/CHI2023_PathFinder/CHI2023_PathFinder_preprint.pdf), Masaki Kuribayashi et al., CHI 2023 | 机器人检测路口、识别标识并向盲人传达，在七名盲人参与者中评估地图外陌生建筑导航体验。 | 保留“标识与路口是观测载体、用户仍是任务主体”的交接设计，不把自动识别包装成独立安全能力。 | 原型/用户体验证据；没有精确入口定位、metric arrival 或独立 completion truth。 | `RESERVE` |

## 最值得先做的五个读后动作

1. 先做 `DeepSolo/CRAFT geometry-only` 对照：只换词级几何源，直接看 merged-line bearing 是否改善。
2. 再做 `whole-word calibration + conformal UNKNOWN`：把 coverage、错认和错误身份 admission 放在一张选择性风险表里。
3. 只在明确的 `TEXT_TOO_SMALL / OBLIQUE / OCCLUDED` 状态触发一个 Active Text / legibility action；不复活通用 EIG 策略。
4. 把 completion 指标改成 position、visibility、grounding、orientation、handoff 五个端点；任何一个缺失都不能叫完整完成。
5. 如果要引入几何，优先测试 `text plane -> relative pose/stand-off`，不要再用框高或框中心冒充距离、入口与完成。

## 不应从这 20 篇推出什么

- 视频文字 SOTA 不等于目标身份、物理实例或开放世界 `NONE` 已解决。
- tracking、SLAM、VPR 或 sign grounding 只能提供各自层的证据，不能替代功能入口、到达或用户完成。
- 超分、语言模型、轨迹聚合都可能生成更流畅但更错误的文字；必须把错误 lexical admission 作为一等指标。
- 主动观测只有在针对可观测缺陷且不越过身份边界时才有意义；“多看一眼”本身不是算法成功。
- 这份清单不重开已关闭的 matcher/threshold/fusion sweep，也不改变默认 App、真机、产品或安全权限。

## 检索与去重记录

- 检索日期：2026-08-27。
- 四个工作流：scene-text source/geometry、long-term state/absence、active sign acquisition、arrival/handoff。
- `sources_reviewed = 447`，包括检索工具返回的重复版本与一次解析重试；全文/摘要核验使用论文、出版社或官方项目页。
- 已按题名、DOI 和 arXiv/出版社版本合并；最终 20 篇与此前 40 篇精读池、GoMatching、QueryNLT、AECNav 无精确题名冲突。
- 入选标准：必须为当前链路带来新的信息源、状态表示、观测动作、几何/终点真值或辅助交互边界；只报告 leaderboard 增益、只换 backbone、只调 threshold 的论文不入选。
