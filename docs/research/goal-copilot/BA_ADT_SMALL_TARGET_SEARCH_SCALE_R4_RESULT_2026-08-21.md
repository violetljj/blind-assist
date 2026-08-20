# BA-ADT small-target search scale R4 result

状态：`ADT1_SMALL_TARGET_SEARCH_SCALE_R4_NOT_SUPPORTED_ON_FIXED_WINDOWS / GLOBAL_GAIN_WITH_IDENTITY_REGRESSION / CONSUMED_DEVELOPMENT_ONLY / DEFAULT_APP_UNCHANGED`

## 结论

在 R1 锁定的五个 `NO_CANDIDATE` 窗口中，R4 的 detectability proxy 排除两个证据不足窗口，保留
W2/W3/W4 共 3 个窗口、97 个 eligible LOST-search frames。S0、S1、S2 在这 97 帧上都没有产生
IoU >= 0.10 的 target proposal：candidate recall 均为 `0/97`，有正确 proposal 的窗口均为 `0/3`，
确认重捕获均为 `0/3`。因此，同一 YOLO11n detector 的 1280 全图搜索和 2x2 tiled 640 搜索都没有
打穿固定的小目标失败窗口，R4 不支持“只扩大搜索尺度即可解决当前五窗瓶颈”。

全序列指标确有改善，但不属于固定窗口机制成功，而且伴随身份与 false-visible 回归。S1 出现 1 次
wrong-instance，S2 出现 4 次；S2 每个 LOST search frame 还需 4 张 tile inference。因此不能凭全局 recall
或 longest dropout 的改善保留 S1/S2，也不能把结果解释为端侧候选生成突破。

## 冻结比较

三个 arm 共用 checkpoint `yolo11n.pt`，SHA-256 为
`0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1`。正常 detector/tracker 仍以
640 运行；只改变 LOST proposal search：

| arm | LOST candidate search | 其余组件 |
|---|---|---|
| S0 | 既有 full-frame 640 | 冻结 |
| S1 | full-frame 1280 | 冻结 |
| S2 | 2x2 tile、20% overlap、每 tile 640；global box 去重 | 冻结 |

TargetMemory、appearance verifier、弱时空先验、2-of-3 confirmation、memory quarantine、flow tracker、
正式 evaluator 与 Goal Copilot 均未改变。S2 没有使用 GT ROI；GT 只进入隔离的 R4 evaluator。

R4 的 Development-only detectability proxy 为 visibility >= 0.50、source bbox 最短边 >= 4 px，且每窗
至少 3 个 eligible frames。它只是固定比较的分母，不是产品可检测性真值。W0/W1 被保留为 insufficient
evidence，不进入要求成功的分母。

## 固定窗口 primary result

| 指标 | S0 | S1 1280 | S2 2x2 |
|---|---:|---:|---:|
| eligible windows | 3 | 3 | 3 |
| eligible LOST-search frames | 97 | 97 | 97 |
| candidate recall | 0/97 | 0/97 | 0/97 |
| windows with correct proposal | 0/3 | 0/3 | 0/3 |
| windows confirmed reacquired | 0/3 | 0/3 | 0/3 |
| first-valid proposal latency | 未观测 | 未观测 | 未观测 |
| confirmation overhead | 未进入 | 未进入 | 未进入 |

四段 duration 按窗口记录。W2/W3/W4 的 `T_invisible` 为 `75/167/15` 帧，`T_subdetectable` 为
`24/0/3` 帧。由于三个 arm 都未产生正确 proposal，`T_system_miss` 分别在固定窗口末尾右删失，已观测
下界为 `36/52/20` 帧；`T_confirmation` 没有起点，不能计为 confirmation failure。

## 全序列 secondary diagnostics

| 指标 | S0 | S1 1280 | S2 2x2 |
|---|---:|---:|---:|
| localization recall | 0.6203 | 0.6720 | 0.6852 |
| mean IoU | 0.4743 | 0.4999 | 0.5106 |
| longest visible dropout | 159 | 149 | 148 |
| @30 / @90 / @180 reacquisition | .4/.5/.5 | .5/.5/.5 | .5/.5/.5 |
| correct / wrong / unresolved redetection | 13/0/0 | 16/1/0 | 14/4/0 |
| false-visible | 0.0073 | 0.0128 | 0.0182 |
| LOST search frames / inference images | legacy counter unavailable | 2666/2666 | 2615/10460 |

这些 secondary 数值说明搜索尺度改变了固定五窗之外的全局行为，但 identity regression 已使两个 arm
都不满足 R1 的 `wrong-instance = 0` 约束。它们不能覆盖 primary fixed-window negative。

## 工程决定与 claim ceiling

R4 到此封存，不在同一已消费序列上继续 post-hoc 扫 resolution、tile 数、overlap 或阈值；不接
DINOv2 verifier、Sky、Goal Copilot 或默认 App。现有证据关闭的是已测试的同-detector scale arms，
不是“所有多尺度搜索永远无效”。

唯一合理 successor 是另立 bounded、teacher-only 的
`ADT1_SMALL_TARGET_VISUAL_UPPER_BOUND_R5`：只在这组已消费 Development 窗口判断 materially stronger
visual-query/proposal teacher 能否产生正确 proposal，仍由现有 TargetMemory 与 confirmation 接收，且
禁止部署/效率/产品主张。若 teacher 仍为 `0/3`，应停止纯 RGB detector 堆叠并转向信息/空间记忆边界；
若 teacher 能稳定恢复，再单独讨论 teacher-to-edge，而不是现在启动 Sky。

## Evidence identity

- R4 evaluator：`artifacts.local/evidence/ba_adt_search_scale_r4/evaluation_final.json`，SHA-256
  `83F87B9CB8FA528AAAA8CCAD84380D2285755FBFAEE41383E8BDC169154ECD76`；
- S1 observations/evaluation SHA-256：
  `A04C02F4CEA1487C03195DA3E7806C2C0553D47097078C24C06D882AC23815DB` /
  `91989EF04EC3A305803350669EE0842B4CC2A5CF30D7AF9FE5FCCA6F04280F86`；
- S2 observations/evaluation SHA-256：
  `9B0D45BC728B6070492BFDD2CD9F06634FC99451492C1D9BEE3B0534050ABE54` /
  `78E60626EEA8BB51F318B8848CC491309EDBB8F5069543061656A28CD6916214`。

全部结果来自同一已消费 `clean_seq136 / Carrot_A` Development episode，只支持候选生成机制诊断；
不支持 held-out、真实用户、交互式导航、安全、端侧性能或默认 App 结论。
