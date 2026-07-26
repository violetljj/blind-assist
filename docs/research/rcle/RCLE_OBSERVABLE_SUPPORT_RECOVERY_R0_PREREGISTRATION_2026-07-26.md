# RCLE Observable Support Recovery R0 设计预注册

状态：`DESIGN_FROZEN / DESIGN_REVIEW_PASS / EXECUTION_NOT_AUTHORIZED`

冻结时间：`2026-07-26T12:43:48+08:00`

机器设计锁：
[`RCLE_OBSERVABLE_SUPPORT_RECOVERY_R0_DESIGN_LOCK_2026-07-26.json`](RCLE_OBSERVABLE_SUPPORT_RECOVERY_R0_DESIGN_LOCK_2026-07-26.json)

设计锁 SHA-256：
`3fcc21e28ba84e18d10b1c236a9a0df167d2a6464ea5ebefcb52ce4395152bac`

纸面修订：`R2`。首版设计锁
`5cb297ebbb4167778fa75dd62968134771aacfc933cb3bc9e23bee2785c08207`
在独立审查中因 point-class acceptance、prior survivor、patch boundary 和
field-exit 语义不唯一而失败；第二版
`2cf39ed78936c7f8992495f4558c45e8f4ccb5242e293420ca83c9ad74bc4f80`
在复审中因跨文档 baseline-photometric 句义冲突和 development receipt
锁定时序冲突而失败。本修订只消除这些纸面自由度，没有使用代码、fixture、
formal seed 或结果。

## 冻结结论

本边界只预注册一个新的观测模型假设，不是 R1 的第二次 coverage
revision，也不授权算法实现、正式 trial、真实数据抓取或 Phase B：

> 在不读取 oracle occlusion mask、不降低任何既有门的前提下，一个只使用
> 连续三帧真实可观测证据的 support manager，能否为
> partial-occlusion pitch 恢复足够而且仍可验证的空间支持？

唯一候选为 `OBSERVABLE_THREE_FRAME_SUPPORT_MANAGER_R0`。它只管理进入既有
局部仿射估计的可观测 correspondence；不改变 expansion 定义，不跨帧平均
expansion，不生成伪 flow，不把不可评价项回填为零。

最终机器设计锁已通过
[独立设计审查](RCLE_OBSERVABLE_SUPPORT_RECOVERY_R0_DESIGN_REVIEW_RESULT_2026-07-26.md)。
当前动作仍到此为止；审查 PASS 只允许以后另行讨论是否授权实现，本身不
产生实现或实验权限。

## 旧证据永久降级

R0/R1 的 seeds `1000–1019`、2520 个 trial IDs、全部 metrics、失败 cell、
失败原因和 receipts 永久标为 `DISCOVERY_EVIDENCE`。它们只支持问题定位：
R1 剩余失败集中在 partial-occlusion pitch，且链路为
`support/hull → common 5/9 → pair coverage`；它们不得再用于：

- 选择或微调 support manager；
- 选择 photometric、lifecycle、补点、RANSAC、feature 或 LK 参数；
- 充当开发集、确认集或独立验证分母；
- 将 R1 的 `REVISE` 改写为待补跑的近似 `PASS`。

R0 协议和 R0/R1 receipt 均保持原哈希，旧产物不覆盖、不重写。

## 新数据角色冻结

| 角色 | Seeds | Matrix | 使用规则 |
| --- | --- | --- | --- |
| discovery | `1000–1019` | 既有 R0/R1 2520 trials | 永久只读发现证据 |
| development | `2000–2019` | 原完整 2520-trial matrix，仅替换 seed | 候选实现和测试锁定后最多运行一次；任一原门失败即关闭 |
| sealed validation | `3000–3019` | 原完整 2520-trial matrix，仅替换 seed | 候选代码、环境和开发 receipt 锁定前不得生成或运行；之后独立上下文一次性运行 |

“sealed”是结果与执行时序隔离，不是隐藏 seed ID：公开 seed 清单用于证明
没有事后换样；在候选锁定前禁止物化验证帧、运行验证候选或读取验证结果。
失败 seed 一律不得替换。开发或验证矩阵都不得只跑已知的四个失败 cell。

## 唯一候选

### 三帧角色

对目标 pair `t → t+1` 使用 `t-1, t, t+1`：

- `t-1 → t` 只提供 raw observed track lifecycle 与遮挡证据；
- expansion 仍只由 `t → t+1` 的实际 `dt` 和原局部仿射公式估计；
- 第一个没有历史帧的 pair 原样运行 R1 baseline，并保留在原 pair 分母中；
- raw 与 rotation-compensated 两条路径继续共享同一初始 support pool，
  之后分别接受真实跟踪、分别应用原门。

### 可观测遮挡

不得读取 `SyntheticSequence.occlusion_masks`、generator 的遮挡位置、矩形、
方向或任何等价 metadata。允许的信息只有图像、时间戳、图像边界和
source-known rotation warp 可计算的有效边界。

“在 `t-1 → t` 存活”精确定义为 raw observed track 同时满足：forward
有限、forward-backward `≤1.0 px`，并且两帧完整有效 `7×7` patch 的
photometric error `≤20.0`；不要求它已经成为 prior cell affine consensus
inlier。

一条 prior survivor 若不是可观测的视野/warp 边界退出，并在 `t → t+1`
出现以下任一失败，则记为 observable occlusion：

- forward-backward error `>1.0 px` 或往返跟踪无有限解；
- 8-bit 灰度、median-centered `7×7` patch 的 mean absolute error
  `>20.0` intensity。

新生于 `t` 的点若失败，只记为普通 track failure，不能反推为遮挡。
observable occlusion 的排除中心固定为 prior survivor 在 `t` 的端点，只
用于排除其 `10 px` 邻域中的补点候选，不能产生正 support、flow 或
expansion。

Photometric patch 以亚像素 track center 为中心，在 `[-3,+3]²` 的 49 个
整数 offset 上双线性采样；两个 patch 分别减去各自中位数，再计算 8-bit
intensity 的 mean absolute error。49 个采样点必须全部有效，禁止裁剪、
padding 或 partial patch。raw path 的有效域只有图像边界；
rotation-compensated path 的有效域是图像边界与 source-known rotation warp
有效域的交集；两者都不得读取 generator occlusion mask/metadata。

Field exit 优先于 occlusion。若 forward center 有限，以它检查完整 patch；
若 forward 不可用，则用已观测 `t-1 → t` 位移按
`dt_current/dt_prior` 缩放得到 constant-velocity prediction。预测点不能在
对应路径有效域容纳完整 `7×7` patch 时记为 `GEOMETRIC_FIELD_EXIT`，不再
判 photometric failure 或 occlusion。其余 prior-survivor 当前腿失败才可
判 observable occlusion。分类优先级固定为：
`GEOMETRIC_FIELD_EXIT → OBSERVABLE_OCCLUSION → ORDINARY_NEW_TRACK_FAILURE`。

### 确定性空间补点

先完整运行 R1 baseline。只有某个原 3×3 cell 在 raw 或 compensated
路径中出现 support `<12` 或 hull `<0.10`，该 cell 才能启动 support
manager；已充足 cell 不改动。

候选池固定为：

1. `t` 的 baseline features；
2. `t-1 → t` 可观测存活 track 在 `t` 的端点；
3. 仍不足 cell 内的 deterministic spatial supplements。

补点把目标 cell 临时分为 `4×4` 选点区，只用于选点，不改变原 3×3
evaluation grid。按子区 row-major 顺序，每区最多选择一个最高
Shi–Tomasi response；并列按 `y, x` 排序。与已有点小于 `5 px`、位于
observable-occlusion `10 px` 邻域或使 cell 总候选超过原 `80` 上限的点
丢弃。

三类点的准入逐类固定：

| Point class | `t-1 → t` | `t → t+1` | 对 support/hull 的贡献 |
| --- | --- | --- | --- |
| baseline feature at `t` | 不要求历史 | 完全沿用 R1 path-specific acceptance，不新增 photometric 门 | 它们单独定义 baseline 是否触发 manager；沿用 R1 计数 |
| carried survivor endpoint | 必须满足上述 raw prior-survivor 定义 | raw/comp 各自要求 forward 有限、FB `≤1 px`、完整 path-valid patch error `≤20` | 只在该路径自身通过时计入 |
| spatial supplement | 不要求历史，失败不得叫遮挡 | raw/comp 各自要求 forward 有限、FB `≤1 px`、完整 path-valid patch error `≤20` | 只在该路径自身通过时计入 |

合并后的每条路径仍执行完全不变的 R1 consensus 和全部 local-affine 门。
baseline 点不增加 photometric 准入门，避免把“恢复不足 cell”的候选偷换成
对既有 baseline 的第二套筛选；carry/supplement 则必须通过新增的真实可观测
检查。没有合格新增 correspondence 就是没有恢复。禁止插值、复制邻 cell
flow、虚构 lifetime 或把遮挡 mask 反向当作答案。

## 完全不变的估计与门

下列内容逐项绑定 R0 protocol SHA-256
`d20e77f3ea5f7ac55376006f1d14feb0ffb5daffd10a42792912fb89cdb1b502`：

- `3×3` evaluation grid；
- `v(p)=A(p-c)+b` 与 `expansion=0.5×trace(A)`；
- consensus support `≥12`；
- track hull fraction `≥0.10`；
- design condition number `≤1000`；
- median residual `≤0.75 px/frame`；
- raw/comp common evaluable cells `≥5/9`；
- evaluable pair fraction `≥0.80`；
- clean、FPS、noise、blur、partial-occlusion 的全部误差和 coverage 门；
- trial 统计单位、seed-cluster bootstrap、完整分母和
  `NOT_EVALUABLE` 语义。

本候选不得通过更多候选点绕过 consensus support：新增 carry/supplement
只有在对应路径实际通过 forward、FB 和 photometric 检查且进入同一 R1
consensus 后才计入 support 与 hull；baseline 点始终只按原 R1 acceptance
与 consensus 计数，明确不新增 photometric 门。

## 若以后获实现授权，执行顺序

1. 独立审查本设计锁；审查失败只能在纸面另立版本，不得写代码或跑结果。
2. 只实现上述一个候选；允许 unit fixtures 和确定性/负回归测试，不允许
   在任何 formal seed 上搜索多个版本。
3. 锁定代码 SHA-256、设计锁 SHA-256、环境、tests 和输出位置。
4. 在 development `2000–2019` 上完整运行一次原 2520-trial matrix。
5. 先检查全部原 clean、FPS、error、stress 与 coverage 门；任何一项失败，
   终态为 `CLOSE_OBSERVABLE_THREE_FRAME_SUPPORT_MANAGER_R0`。
6. 只有 development 全门 `PASS`，才锁定其 receipt 并允许物化 sealed
   validation。
7. 在独立上下文对 `3000–3019` 完整运行一次；禁止候选作者根据验证输出
   修改代码、阈值或样本后重跑。
8. validation 任一原门失败即永久关闭该候选；真正全门 `PASS` 也只产生
   `SYNTHETIC_OBSERVABLE_SUPPORT_RECOVERY_EVIDENCE_ONLY`。
9. validation `PASS` 不自动开放 Phase B；是否重开必须成为单独决策。

## 明确禁止

- 降低 support、hull、residual、common-cell、pair 或 coverage 门；
- 读取 generator 真实 occlusion mask 或等价 metadata；
- 扩大 synthetic canvas 以消除视野退出；
- 在旧 2520 trials 上继续调 RANSAC、feature 数量或 LK；
- 同时实现多个 support-manager 版本或 dense-flow 版本后选最好结果；
- 抓取 Bonn/真实数据、运行 Replay、进入 Phase B；
- 连接 Android、Risk Field、告警、人体、安全或生产路径。

## 独立审查清单

审查必须对设计锁的精确哈希作答，并逐项给出 `PASS/FAIL`：

1. 旧 R0/R1 是否被永久限制为 discovery，而非确认；
2. development 与 sealed validation 是否无 seed 重叠且都保持完整分母；
3. 是否只有一个候选和一套冻结参数；
4. 是否完全阻断 oracle occlusion mask/metadata；
5. 所有新增 support 是否来自真实可观测 correspondence；
6. expansion、3×3、support、hull、condition、residual、common 5/9、
   pair 0.80 和全部 Kill Gate A 是否原样；
7. 首 pair、视野退出、普通 track failure 与 observable occlusion 是否
   fail-closed；
8. 开发失败、验证失败和验证通过是否都有不自动扩权的终态；
9. 是否没有代码、formal trial、新结果或真实数据动作。

任一项 `FAIL` 时，当前设计不得进入实现。

## 解释风险

这是一项可证伪的设计假设，不是效果判断。固定 `7×7/20 intensity` 的
photometric 规则可能把非遮挡光照或纹理变化误判为遮挡；三帧 lifecycle
也可能产生 survivor bias；空间补点增加了候选覆盖，但不保证增加独立、
几何良好的 consensus。原 residual、hull、common-cell、误差和 coverage
门继续承担反虚假恢复约束。

本 synthetic 设计仍使用 source-known rotation；即使未来独立验证通过，
也不证明真实 pose、rolling shutter、真实遮挡、真实相机或人体有效性。
