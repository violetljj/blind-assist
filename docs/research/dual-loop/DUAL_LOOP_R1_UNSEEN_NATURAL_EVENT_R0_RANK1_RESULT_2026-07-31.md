# 双环 R1 未见自然来源事件评价 R0：rank-1 结果

日期：2026-07-31（Asia/Hong_Kong）  
阶段：`DEVELOPMENT`  
终点：`FIRST_UNSEEN_SOURCE_NOT_EVALUABLE`  
执行有效性：`VALID`  
baseline：`UNOPENED`  
candidate `039757b`：`UNOPENED / NOT_RETUNED`

## 结论

rank-1 上海夜间步行 session 无法回答 R1 是否产生事件级收益。两个相互不可见、
绑定同一 RGB manifest 与同一 prompt 的 AI 审阅均得到 `0` 个高置信正例；预冻结
最低要求为 3 个。因此按协议在 baseline adequacy 之前停止，候选算法没有运行，
本终点既不是算法失败，也不是算法有效。

正式复核同时冻结 6 个双方一致的负窗，覆盖：

- `NORMAL_WALKING_SHAKE`：2；
- `TURN_OR_NEAR_IN_PLACE_ROTATION`：2；
- `SAFE_OPPOSING_FLOW_WITH_CLEAR_SPLIT`：2。

这些窗口只作为来源特征与回归资产，不能被重新包装成未见效果证据。

## 身份与隔离

- 来源：Wikimedia Commons，page id `153983964`，CC BY 3.0；
- 480p payload SHA-256：
  `58971199576c01d675080f8592b0ca69870054108418fefe5bdff73e199e0f49`；
- review manifest：566 个 1 Hz RGB frame，SHA-256
  `97204f6cb914ab5080db8c6c8dbf1888b7a6c426d04dc44222677ec438ddbd0a`；
- canonical prompt SHA-256：
  `ed24afa88aa3bd455a00bf0f789c10137059f549ed976a0a9ad22537b1f1564c`；
- formal review A/B SHA-256：
  `0cf995178df0280aa94af5371b45abd499ce0cfcdb0ac03aab442f637c2c5294` /
  `fe6fcc6aeb47e6680acc1e05c0a28ea81a44d3ffc0adee7127b873ab9584b785`；
- truth freeze receipt SHA-256：
  `24049a8203ab82a5ba5803f5730c4a2b81caa764ca007021518210678a821c54`。

复核只看 RGB，未读取 detector、risk、planner、feedback、baseline 或 candidate
输出。来源是 exact payload / capture-session unseen；但作者与旧上海视频同属
`AmbienceX`，所以不能声称 creator-family independent。真值是 model-reviewed
research evidence，不是客观传感器事实或真实用户效果。

## 分歧与裁决

早期非权威诊断出现 `0` 对 `7` 的正例分歧。第三个新上下文逐一复核争议区间后，
将它们全部判为安全分流、同向安全跟随、转头或近静态观察。随后用预先保存且哈希
绑定的统一 prompt 重跑两次正式独立复核；两路均得到 `0` 正例，因此正式结果采用
`model_consensus`，第三路只保留为分歧诊断，不替代正式两路 receipt。

## 失败学习

观察：长自然步行视频可以包含大量运动与人流，却仍没有满足“持续闭合＋路径占用＋
提醒价值”的正例。  
支持的推断：只以“自然、连续、有人流”筛来源，不足以保证事件评价机会。  
替代解释：1 Hz 接触表可能漏掉极短事件；但正式两路均检查完整时序范围，且争议段
经更细复核仍不满足路径占用。  
被挑战的约束：不降低正例定义，也不把普通迎面人流强行标成危险；应改进来源的
outcome-blind opportunity proxy，而不是降低真值门。  
下一可证伪假设：已冻结排序中的 rank-2 市集步行视频能提供至少 3 个路径占用正例，
同时保留至少 6 个多类负窗。  
信息增益：证明了 truth-first 门能在任何算法输出打开前阻止一个无正例来源消耗
baseline/candidate，并暴露“自然步行”metadata 对任务机会的弱代理性。

## 下一步权限

本 rank-1 终点公开后，可按原 registry 的预冻结 Unicode title 顺序，把 rank-2
`Iran Shiraz City Tour 2021 -5` 作为新的 evidence instance。不得回改 rank-1
窗口、阈值或正例定义，也不得利用 rank-1 的算法输出筛选后继来源。

