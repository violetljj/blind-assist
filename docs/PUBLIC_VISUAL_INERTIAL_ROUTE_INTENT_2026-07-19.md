# 公开视频可行动性与视觉惯性路线意图诊断（2026-07-19）

## 结论先行

当前训练瓶颈已经从“数据集是否太小”进一步定位为“监督合同与输入信息不完备”：

1. 旧标签把最终安全绕行误写成早期不应告警，存在因果语义冲突；改为 `context_attention -> intervention_needed -> route_clear` 后，该冲突已消除。
2. 事件级纯视觉输入能够识别环境和施工标志，却不能在使用者实际转向前判断其将选择哪条路线。事件均值、帧级风险轮廓、生命周期 readout、waypoint ridge 和多种 head 优化均未闭合跨来源门禁。
3. 手机具备高频陀螺仪和 rotation-vector 传感器，但手机自身转动不等于行走路线转向。ADVIO-15 的未来意图和当前转向确认三个确定性 probe 均失败，因此不能把未对齐的被动 IMU 直接当作路线输入。
4. 下一版架构应把“显式路线意图”作为独立输入：导航规划/用户指令给出将走哪一侧；只有建立可靠的 device-to-world/route 姿态变换后，IMU/视觉里程计才可辅助确认转弯。视觉风险轮廓判断指定路线上的障碍，生命周期头负责 open/clear。像素分割和距离场继续只作辅助监督。

这不是生产升级结论。Android 默认模型、风险状态机、校准、blind 和商用训练均保持不变。

## 因果标签审计

| 轮次 | 证据 | 结果 |
| --- | --- | --- |
| r7.85 | 三个冻结事件的 current/past-only 状态重放 | Bangkok 原“安全负例”实际在 307 秒需干预、311 秒路线清空；最终安全不能反写早期 no-alert。报告 SHA256 `0120009d...07e5` |
| r7.86 | 12 个事件、10 个来源统一重标 | `context_only=9`、`intervention_then_clear=2`、`persistent=1`；旧 route-role 与因果 actionability 不一致 `3/12=25%`。报告 `e616cec4...651b` |
| r7.87 | Cardiff 两个视觉上像“接近障碍”的窗口 | past-only trace 均不足两秒，二者都是 `context_only`，没有用视觉印象冒充干预证据。报告 `9cb5be2a...55e` |
| r7.88 | Ulm 路线转弯候选 | c01 在 17 秒进入并持续干预，补足第三个独立 intervention 来源；c03 仍为 context-only。报告 `eb2b6a09...3047` |
| r7.89 | source-isolated actionability manifest | 16 个事件、11 个来源；4 个干预事件来自 Bangkok/Edmonton/Ulm，12 个 context 事件来自 9 个来源；无旧标签字段。报告 `a38d0064...e3df` |

## 纯视觉确定性诊断

| 轮次 | 唯一变化 | 结果 |
| --- | --- | --- |
| r7.90 | 132 维事件均值，固定 DINO + 当前 RGB + 过去 1/2/3 秒 flow | AUROC `.0795`、balanced `.3182`、干预召回 `0`；事件均值破坏了时间结构。报告 `bfd961b9...610f` |
| r7.91 | 相同输入/切分/ridge，仅改为帧级 risk profile + 固定两秒生命周期 | AUROC `.3409`、balanced `.5227`、干预召回 `.5`、context 召回 `.5455`；比事件均值好，但跨来源仍失败。报告 `e280c307...855e` |

r7.91 说明“风险轮廓 + 生命周期头”的任务拆分方向正确，但当前视觉因果输入不包含尚未发生的路线选择。继续 prototype/bootstrap、距离场权重、SAM/ASAM 或阈值搜索都不能补出缺失信息。

## ADVIO 视觉惯性辅助路线

ADVIO 是公开的行人视觉惯性里程计数据，手机视频、IMU、ARKit 与 ground-truth pose 共享同步时间轴。这里只选择 Office 03 sequence 15 验证管线，来源登记为 `noncommercial_research_only_turn_intent_auxiliary`。

### r7.92–r7.93 完整性与同步审计

- 官方压缩包实际大小 `54,845,329` 字节。
- MD5 `f5febcd087acd90531aea98efff71c7c`，与官方 Zenodo 记录完全一致；SHA256 `15127eca...111f`。
- 共享有效时长 `51.434256s`；gyro `99.9001Hz`、frame timestamps `59.9952Hz`、ground-truth pose `99.9001Hz`。
- 位姿四元数最大单位范数误差 `6.99e-08`；轨迹低方差轴为 y，诊断使用 x-z 水平面。
- r7.93 全部完整性、单调时间戳、采样率、同步和四元数门通过；报告 SHA256 `b12ce571...22ce`。

来源许可为 CC BY-NC 4.0，因此严格禁止商用、生产训练、默认模型替换、校准或 blind 使用。它也不提供障碍风险或 actionability 真值。

### r7.94–r7.95 因果 IMU probe

冻结 target 为：最近 0.5 秒 x-z 路径与未来 0.25–1.0 秒路径的绝对夹角达到 20 度。未来 ground-truth 只生成辅助标签，从不进入输入。输入只看过去 0.5 秒 gyro/accelerometer，采用五个连续时间块留出并在训练侧留 1 秒 guard。

| Probe | 样本与输入 | OOF AUROC | Balanced | 结论 |
| --- | --- | ---: | ---: | --- |
| r7.94 | 400 样本；36 维有符号 device-axis 统计 | `.4770` | `.4701` | 失败；报告 `9611b8f1...183d` |
| r7.95 | 完全相同样本/标签/切分/head，仅换 22 维旋转不变 magnitude/derivative | `.4746` | `.4647` | 失败且无改善；报告 `68a860d7...e87b` |

两个 probe 的类别近乎平衡（201 turn / 199 straight），因此失败不能归咎于类别不平衡。它表明被动 IMU 在转向发生前并不包含可靠的自主决策信息。

### r7.96 因果转向确认负控

为验证“IMU 至少能确认已经发生的转向”，r7.96 把 target 改为完全截至当前时刻的两段路径夹角，保持 r7.95 的 22 维旋转不变输入、ridge、连续块留出、guard 和阈值。390 个样本中有 131 turn / 259 straight；结果 AUROC `.3465`、balanced `.3878`、turn recall `.3664`、straight recall `.4093`，报告 SHA256 `a8cc4427...3058`。

这项负控说明手机姿态变化与人体在世界坐标中的路线变化并不等价。当前证据不授权 Android benchmark-only 传感器接口。后续若重启 IMU 路线，必须先提供可验证的 device-to-world/route 姿态对齐；否则 IMU 既不能承担意图预测，也不能直接承担路径转向确认。

## r7.97a：显式路线输入的直接上限通过

在不训练新 head 的情况下，把冻结 future-route teacher 仅作为外部导航/用户路线的 oracle 代理，重放 r7.89 全部 16 个事件和 11 个来源。固定规则为路线点进入扩张 marker 的比例至少 `1/3`，连续两秒才升级。结果 intervention recall `1.0`、context recall `.8333`、balanced accuracy `.9167`，最低有效帧覆盖 `1.0`。勘误版额外复验 12 份 feature report 和 13 个本地视频文件 SHA，报告 SHA256 `ae019fa9...1e85`。

这项结果验证了架构接口，而不是未来视频模型：runtime/eval 仍禁止 future teacher。正式解法收敛为外部 route field 条件化视觉风险轮廓，并以确定性交叠和生命周期输出 actionability。详见 [EXPLICIT_ROUTE_INTENT_MODEL_CONTRACT_2026-07-19.md](EXPLICIT_ROUTE_INTENT_MODEL_CONTRACT_2026-07-19.md)。

## 推荐的模型接口

```text
显式路线意图（导航规划 / 用户选择，可缺省）
                    │
                    ▼
视觉风险轮廓 ──> route-relative risk ──> 生命周期 open / clear
                    ▲
                    │
       已对齐的姿态 / VIO（可选；未对齐 IMU 禁止直接使用）
```

无显式路线意图时，系统只能输出无方向的 `context_attention`，不能从“障碍在画面中放大”直接升级为左/右绕行指令。若 past-only route evidence 已连续满足冻结条件，可以输出保守的 `intervention_needed`，但方向仍需独立可通行侧证据。

## 下一阶段门禁

1. 优先从导航路线或用户选择构造 route field；无该输入的 episode 明确标为 unknown，不让模型猜测未来自主选择。r7.97a 已验证该接口的 oracle 上限。
2. 若研究 IMU/VIO，先独立证明 device-to-world/route 对齐和当前转向确认；r7.96 未通过前不新增 Android 传感器接口。
3. 至少两个独立设备/场景验证姿态对齐后的延迟与误触发；ADVIO 仅作非商用研究参照，不计生产数据信用。
4. 再把 route-intent token 与 r7.91 风险轮廓做一次严格 OFAT；通过跨来源 event/lifecycle 门后，才讨论 prototype/bootstrap 五组短跑。
5. 距离场仍为辅助监督；SAM/ASAM 保持第三顺位，不能早于输入合同闭合。

## r7.99–r8.07：类别化路线选择的覆盖审计

- r7.99 的三条固定路线模板在既有 16 事件上得到 balanced `1.0`，但 r7.99a 显示 4 个 intervention 全为 `STRAIGHT`；LEFT/RIGHT intervention 均为 0，因此满分只是一条 straight-choice 诊断。
- r8.00 将 r7.61 future-anchor mean-x 当作方向搜索后得到 5 个候选；r8.02 全拒绝。关键勘误是 r7.61 从未把该 x 定义成转向类别，它会被独立运动和光流失配污染。
- r8.03 用上半画面稳健背景流独立定义转向，得到 30 个跨来源候选；r8.05 固定模板连续相交后只剩 3 个。r8.07 复核表明 Kampala 两段均是开放通道旁的施工边界，Bramwell 是静态建筑误检，故 LEFT/RIGHT intervention 仍为 `0/0`。
- 结论：显式完整 route field 的架构上限继续成立；简化为 `LEFT/STRAIGHT/RIGHT` 的真实 provider 验证尚未成立。下一轮只检索“所选转向分支被同一真实临时障碍连续阻断并引发停步/改道”的公开视频，不再调整模板扩张或生命周期阈值。

## r8.08–r8.09 定向来源获取结论

- 查询和准入语义在获取前冻结；Commons 的 Windows TLS 恢复只重复原始三条查询，未改词或分页。
- Commons 唯一命中为警方执法视频，按元数据语义在下载前拒绝。
- Vimeo 的 Burwell 候选虽有条目级 CC BY 3.0 许可，但全片是多段剪辑的工程新闻材料，并非连续行人 POV，视觉复核拒绝。
- 新增 LEFT/RIGHT intervention 为 `0/0`。这轮是有效的来源排除证据，不是模型失败，也不改变 r7.97a 对完整 route-field 接口的支持。

r8.10–r8.11 的 Internet Archive 冻结查询只返回 3 项 1930–1945 年历史道路/测绘/军事影片；它们是全文关键词拼接误命中，均在下载前拒绝。通用档案全文检索不再作为当前事件扩源的优先渠道。

## r8.12–r8.18 路线条件距离风险场

- 精确 route-to-risk 交叠交给同一 LOSO 线性 head 可达 `1.0`，而冻结 DINO 二值风险图即使沿路线采样也只有 `.6249`；根因在风险表示而非 head 优化。
- 每个父来源同时加入 barricade/sand pile 后二值 teacher 仍为 `.6332`，来源偏置没有被障碍族数量解决。
- 单独改成 bbox 距离场辅助监督后，路线条件 balanced `.9156`，最差来源 `.89`，三个方向均至少 `.88`；全局 readout `.5446`，直接证明路线条件交互不可省略。
- 固定两帧 open 生命周期 balanced `.9429`；五组 80-step bootstrap 稳定但不过门，故不转向 optimizer/SAM/ASAM。
- 该结论仍为 train-only synthetic representation diagnosis；下一步必须在真实 provisional 事件和独立非未来 route provider 上复验。
