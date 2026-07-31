# TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0 独立设计复核结果

状态：`PASS / CONTRACT_FROZEN / PROPOSAL_ONLY / EXECUTION_NOT_AUTHORIZED`

日期：2026-07-31（Asia/Hong_Kong）

复核对象：[TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0 设计合同](TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0_DESIGN_CONTRACT.md)

合同 SHA-256（本复核绑定版本）：
`cd80680aef5306588f1e972308e2c3be76c5d6d9341f25371713e2fe043db816`

## 结论

独立只读复核通过。合同现在足以作为一个不访问候选输出的 F1 设计接口：研究问题、
B/C1/C2 数据角色、production-selected target 的因果输入、严格相邻帧约定、背景环
候选集、similarity/LK 质量门、residual 与 raw 对照、事件分母、确定性选择和失败
终点均有可复算定义。

该 PASS 只确认设计合同可以冻结，不授权 B 实现、B 输出访问、C1/C2 回放、任何
Android/shadow/active 接线或产品/安全主张。

## 本轮修订并复核的关键项

- 补齐半开 bbox、像素坐标、原生 frame shape、严格相邻 frame index、无隐式 resize 和
  dynamic-mask detection 输入边界；
- 要求 C1/C2 在候选输出访问前分别冻结 metadata-only identity/ancestry manifest，
  不足时以 `NOT_EVALUABLE` 关闭；
- 固定 luma/LK/similarity 的实现身份、RANSAC 采样/阈值/seed/tie-break、reflection
  禁止规则和 condition number 定义；
- 固定四角 warp 后轴对齐 bbox、越界不裁剪、raw 无效条件和事件 truth state 映射；
- 明确 `truth-eligible pair`、event coverage、paired gain count/fraction、弃权事件
  分母及 `wrong-signed` 计数；
- 将实际有效环面积作为最后 tie-break，并把 B/C1/C2 的“非单事件/目标”改为至少
  两个事件、两个 target、单项贡献不超过 50% 的可执行门；
- 冻结输出 provenance、frame/detection manifest hash、shape、mask/点统计、quality
  与 `abstention_reason` 枚举和失败优先级；
- 将 `SIMILARITY_CANARY_NOT_SUPPORTED` 与 `NO_DEVELOPMENT_INCREMENT` 的分流写成
  事件最低可评价要求与 Development 晋级门的确定性关系。

## 证据与边界检查

- 未读取 CrowdBot、Matoaka、Shiraz 或任何候选 output；未执行 producer、evaluator、
  replay、Android、CameraX、QNN 或设备操作。
- 未继承 R1/D0 的数据、truth、formal identity、one-shot authority 或 runtime 权限；
  R1 failure decomposition 仍保持 `POLICY_GRANULARITY_MISMATCH_SUPPORTED`，scene-scale
  active 路线仍关闭。
- 合同仍明确 `scientific_status=NOT_RUN`、`claim_eligibility=CLAIM_NOT_SIGNABLE`、
  `execution_authority=NOT_AUTHORIZED`；C2 即使未来通过也只允许
  `CROSS_SESSION_SIGNAL_REPLICATION`，不产生 active、产品或安全权限。
- 所有数值门仍是图像信号/质量诊断门；其依据、敏感性和版本化修改规则已写明，不能
  因 B 或 C1/C2 结果原地调参。

## 后继授权边界

下一步只能由独立明确授权开启 B Development。若获得授权，必须在隔离 offline module
中按本合同生成实现 lock、identity/ancestry manifest、truth-blind producer、独立
validator/evaluator 和专项 fixtures；本复核不预授权这些动作。
