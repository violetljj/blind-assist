# 动态出行风险候选精读索引

状态：`snapshot / pending candidates / not an active route`

本目录保存 2026-08-24 对动态出行风险 30 项论文、算法和开源项目的一手材料精读。它用于以后复核候选机制、证据强度和不可迁移边界，不建立算法 successor、实验执行、Android、默认 App、生产或安全 authority。

## 入口

- [30 项逐项精读](DYNAMIC_TRAVEL_RISK_30_DEEP_READING_2026-08-24.md)：每项记录研究问题、核心机制、核心证据、核心价值、读后感与取舍。
- [根目录待决候选池](../../../idea.md)：保留短候选卡片和当前 `PENDING_CANDIDATE_POOL` 状态。

## 使用边界

- 汽车、机器人、合成数据、研究者模拟和通用机器学习指标只保留原证据域，不升级为 BLV 功能证据。
- 数据集只支持其 source-native labels；没有 wearer route、target future 或事件真值时，不派生虚构的碰撞 authority。
- 任何一项缺输入健康、时间/位姿或身体路径证据时，都不能把“未告警”解释成“前方安全”。
- 如以后明确启动研究，必须从对应研究分类 current 登记一个新的、唯一的 successor；本 snapshot 不自动恢复旧 USTRF 或其他已关闭路线。
