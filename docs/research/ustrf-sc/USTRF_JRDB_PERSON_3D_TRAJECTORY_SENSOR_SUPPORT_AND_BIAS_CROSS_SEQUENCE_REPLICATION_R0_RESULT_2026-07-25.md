# JRDB person 3D trajectory sensor support and bias cross-sequence replication R0 result

状态：`CROSS_SEQUENCE_PROFILE_AVAILABLE_WITH_PARTIAL_REPLICATION / VALID`

权限上限：`DIAGNOSTIC`

## 结论

单 sequence 的数量级结论大体复现，但不能把 Meyer Green 的 `81.85%` 支持率和 `0.195/0.481m` residual median/P95 当成跨场景常数。

3 个仅按 source metadata 冻结的新 sequence 合计：

- object-frame：`8,118/9,771 = 83.08%` sensor-supported；
- motion pair：`7,822/9,679 = 80.81%` sensor-supported；
- centroid residual median/P95：`0.168/0.446m`。

pooled 值接近 seen baseline 的 `81.85% / 78.14% / 0.195/0.481m`，但 sequence 间差异很大：

| Sequence | Object support | Pair support | Residual median / P95 |
| --- | ---: | ---: | ---: |
| `gates-basement-elevators-2019-01-17_1` | `1,568/1,774 = 88.39%` | `1,528/1,756 = 87.02%` | `0.113 / 0.669m` |
| `stlc-111-2019-04-19_0` | `2,543/2,558 = 99.41%` | `2,515/2,536 = 99.17%` | `0.190 / 0.311m` |
| `clark-center-2019-02-28_0` | `4,007/5,439 = 73.67%` | `3,779/5,387 = 70.15%` | `0.166 / 0.410m` |

worst-sequence 不是同一条：最低 object / pair support 都是 Clark Center（`73.67% / 70.15%`），最大 residual median 是 STLC（`0.190m`），最大 residual P95 是 Gates Basement（`0.669m`）。因此 sequence composition 是当前主要限制，aggregate 会掩盖 worst-sequence 和 tail residual。

## 分层复现

### 远距

冻结的 `40-plus` 对比 pooled `0-20` 只在 Clark Center 可评：

- `40-plus`：`1/43 = 2.33%` supported；
- support delta：`-93.61pp`；
- residual median delta：`+0.474m`。

Gates 与 STLC 的 120 帧窗口没有 `40-plus` 分母，故跨 sequence 状态为 `NOT_EVALUABLE`，不能把 Meyer Green 的远距退化升级为三序列一致复现。

距离退化在可观测范围内仍明显：Gates 从 `0-10` 的 `99.51%` 降到 `20-40` 的 `40.83%`；Clark 从 `0-10` 的 `98.93%` 降到 `20-40` 的 `53.58%`，再到 `40-plus` 的 `2.33%`。

### 3D-only

冻结的复合方向门为 `DIRECTION_REPLICATED`：3 条可评 sequence 的 3D-only residual median 都高于各自 3D-and-2D 参照。

- Gates：support `73.68%` vs `88.55%`，residual median 增加 `0.180m`，`n=19`；
- STLC：support `100%` vs `99.41%`，但 residual median 增加 `0.126m`，3D-only 只有 `n=2`；
- Clark：support `9.76%` vs `76.18%`，residual median 增加 `0.115m`，`n=205`。

所以“3D-only residual 更差”方向复现；“3D-only support 一定更低”只在 `2/3` sequence 成立，STLC 的 `n=2` 不能支持反向泛化。

### 遮挡与稀疏点云

遮挡退化随 sequence 变化，但最差组都明显弱于 fully-visible：

- Gates fully-visible / fully-occluded：`99.60% / 41.97%`；
- STLC：`100% / 90.00%`；
- Clark：`98.78% / 30.67%`。

冻结点门继续按最小单元守恒：3 条 sequence 合计 `830` 个 `1..2` 点 object-frame 全部局部 `abstained`，`823` 个零点 object-frame 全部 `annotation-only`；没有改成 supported，也没有关闭整条 sequence。supported `3..9` 点组的 residual tail 高于 `10+` 点组，尤其 Gates P95 `0.761m` vs `0.641m`、Clark `0.484m` vs `0.328m`。

## 输入冻结与算法不变性

在读取任何新 sequence PCD payload、label payload、support 或 residual 前，一次性冻结：

1. `gates-basement-elevators-2019-01-17_1`
2. `stlc-111-2019-04-19_0`
3. `clark-center-2019-02-28_0`

每条均为 logical positions `0..119` / stems `000000..000119`。26 个 metadata-eligible sequence 按 `sha256("jrdb-cross-sequence-r0|" + sequence_id)` 排序，排除 Meyer Green 后取前 3；冻结后没有替换。

支持计算精确复用并 hash-check 原 R0 PCD LZF / field-major 解码、upper/lower 分离审计、logical-rgb360 oriented-box、`>=3` 点门、四类 object/pair ledger、quantile、motion、pose sensitivity 和分层函数。三条 input 全部物化后才整体写入 input manifest；pooled 结果由 primitive rows 拼接后复算，没有平均 sequence 百分比或 quantile。

## 输入限制

新 bag 暴露了不影响本轮 frozen support kernel、但必须单列的 transport/packet 差异：

- 部分 bag 没有同 bag `tf_static` 五边；使用 baseline config 间接 hash-bound 的 JRDB dataset-wide calibration required edges，并在每条 packet/receipt 中记录 fallback provenance。
- 部分 IMU orientation 为零四元数或不能覆盖窗口端点；IMU 在本轮 object/pair support-bias kernel 中完全不消费，相应 packet 字段标为 `NULL_NOT_EVALUABLE`，不生成 IMU authority claim。
- STLC 有 RGB-PCD source time delta 超出旧 `0.05s` availability 门；本轮只消费 frame-stem、2D/3D labels、PCD 与 pointcloud time，RGB-PCD simultaneity 明确标为 non-consumed。

这些处理不改变 PCD、oriented-box、点门、ledger 或 pointcloud-time pair 计算，但本结果不能用于 IMU、RGB simultaneity 或同 bag static-TF availability 结论。

## 验证与权限

- focused tests：`5/5 OK`；
- Python compile：通过；
- independent validator：`16/16 VALID`；
- config SHA：`c999be7569c4fc1d14305ba5635f0f46ab9c399e3ac6ef64eb1a3a092842733b`；
- sequence-freeze SHA：`e470b188d5e1e715b91fe65bb0f951e8f6ff2842b3782b36908295a9b35a4071`；
- input-manifest SHA：`288d7f2766b1047b970c75bfd62f210cb6793f6d1e69551632fac870db4291c6`；
- ledger SHA：`0b968e48f3e6ef68af185386019c5c972292a48e5033c8834a820601a7bbf27e`；
- receipt SHA：`2f3bb2385791ef2f3263b08ec2a29597d8f6c515aa5adf7cd472a1c5c0d25015`；
- validation SHA：`87880769457488c1521566272d0f4442729e7383035fc6fb897d54e84cb0bd8d`。

本结果仍是 annotation-conditioned、source-annotation-derived 的 diagnostic evidence，不是 independent person-center/trajectory truth，不开放 candidate selection、route/event、alert、Android、人体/独立行走或 production authority。

唯一建议边界：本轮已经回答“单 seen sequence 是否是主要限制”。如继续，不应先做复杂 centroid；应另立覆盖真实 `40-plus` 分母的 metadata-stratified sequence replication，或取得 independent person trajectory truth。二者都不得由本轮自动启动。
