# JRDB person 3D trajectory far-range denominator adequacy metadata-blind replication R0 result

状态：`FAR_RANGE_SUPPORT_DECLINE_REPLICATED / VALID`

权限上限：`DIAGNOSTIC`

## 结论

远距支持率下降在预注册分母门通过的 `4/4` 条 sequence 上同向复现，补上了上一轮只有 Clark `n=43` 可评的主要统计缺口。

| Sequence | `0-20m` support | `40m+` support | Delta |
| --- | ---: | ---: | ---: |
| `bytes-cafe-2019-02-07_0` | `7,597/8,393 = 90.52%` | `1/230 = 0.43%` | `-90.08pp` |
| `clark-center-2019-02-28_1` | `9,708/10,719 = 90.57%` | `42/101 = 41.58%` | `-48.98pp` |
| `clark-center-intersection-2019-02-28_0` | `5,553/6,046 = 91.85%` | `34/546 = 6.23%` | `-85.62pp` |
| `hewlett-packard-intersection-2019-01-24_0` | `2,983/3,264 = 91.39%` | `229/1,809 = 12.66%` | `-78.73pp` |

四条 pooled `40m+` 为 `306/2,686 = 11.39%`，pooled `0-20m` 为 `25,841/28,422 = 90.92%`。终态由逐 sequence `4/4` 方向决定，不由 pooled object-frame 冒充 4 个独立复制。

这支持“冻结的 annotation-conditioned LiDAR point-in-box 支持在 40m+ 显著下降”，不证明质心是真实人体中心，也不授予更复杂 centroid 算法的比较或选择权。

## metadata-blind freeze 与分母门

任何候选 label/PCD payload 前，固定 hash 同时冻结 8 条未见 sequence × 360 连续帧：

1. `gates-ai-lab-2019-02-08_0`：positions `100..459`
2. `packard-poster-session-2019-03-20_0`：`218..577`
3. `tressider-2019-04-26_2`：`1021..1380`
4. `bytes-cafe-2019-02-07_0`：`442..801`
5. `clark-center-2019-02-28_1`：`302..661`
6. `clark-center-intersection-2019-02-28_0`：`304..663`
7. `svl-meeting-gates-2-2019-04-08_1`：`106..465`
8. `hewlett-packard-intersection-2019-01-24_0`：`1312..1671`

label-only audit 的 `40m+` 分母依次为 `0 / 0 / 0 / 230 / 101 / 546 / 0 / 1,809`；因此 4 条达到预注册 `>=100` 门。未充分的 4 条没有运行 PCD support，且没有被替换或移动窗口。

首次 metadata freeze 的文字把 distance 写成 horizontal norm，但 hash-frozen parent kernel 一直使用 3D norm。任何候选 payload 前只修正文案为 `sqrt(x²+y²+z²)`，sequence/window 与 `n=100 / >=3 sequence` 门完全不变；初始与 amended freeze identity 由 validator 精确比对相同。

## 同步分母与退化画像

4 条充分 sequence 合计 `45,455` 个 valid-3D object-frame：

- `sensor-supported / annotation-only / abstained / invalid = 33,606 / 5,704 / 6,145 / 0`；
- 3D-and-2D：`32,741/43,575 = 75.14%` supported，residual median/P95 `0.174/0.475m`；
- 3D-only：`865/1,880 = 46.01%` supported，residual median/P95 `0.186/0.492m`。

3D-only support 在 `4/4` 条都低于各自 3D-and-2D。residual median 的 adverse direction 在 Bytes、Clark Center 和 Clark Intersection 成立，但 Hewlett-Packard 为 `0.169m` vs `0.184m`，所以旧“3D-only residual 更差”方向在本轮是 `3/4`，不能升级为普遍规律。

遮挡 pooled support：

- fully visible：`18,235/20,143 = 90.53%`；
- mostly visible：`6,601/7,895 = 83.61%`；
- severely occluded：`6,142/9,481 = 64.78%`；
- fully occluded：`1,763/6,056 = 29.11%`。

点云分母严格保留局部四类语义：

- 零点 `5,704` 个，全部 `annotation-only`；
- 1–2 点 `6,145` 个，全部 `abstained`；
- 3–9 点 `8,637` 个，residual median/P95 `0.252/0.626m`；
- 10+ 点 `24,969` 个，residual median/P95 `0.158/0.375m`。

因此稀疏但过门的 3–9 点组有更重 residual tail；这仍是 descriptive bias profile，不是改点门或换 centroid 的授权。

## 执行、验证与限制

4 个大 bag 按 sequence 串行 Range 获取；每条 packet/eligibility 验证后删除临时 bag，只保留可重取的 hash-bound receipt。旧 120 帧物化器暴露两个非研究失败：固定 `362` member 检查和 Windows 长路径；前者只参数化为冻结的 `3×360+2`，后者只缩短 ignored artifact 路径，失败配置均保留，未改成员、窗口、分母或 support kernel。

- focused tests：`5/5 OK`；
- independent validator：`22/22 VALID`，并绑定 freeze/audit/runner/validator/tests 的最终实现 SHA；
- config SHA：`0587a9f89d2b4a442d6a294539943e5e94f79bd0fce563c4f45bc3aa842d9f0c`；
- amended freeze SHA：`5a67305c7d34bb9b24938720abbeef578196add44a5bef934bcc08cf286b880c`；
- denominator ledger SHA：`4d49c90d9950747c847f8aeead70cd236cb76f307339411ef7b10d47db5726f5`；
- support ledger SHA：`3747bead96d28d7a0047721a7b1a456f900d8a7e8a7cb7c017a19f24c41cf7a4`；
- support receipt SHA：`fcf5d75ea1c9550e7d5840b24ec6bf403b25abd41a35535a7ff9b018bccd689b`；
- validation SHA：`c2643c21d5896245ab727ed79401f405bb154ab8bad5eeb133bdff4a324cf3b6`。

两条 Clark sequence 同日且场景相关，4 条 sequence 不是 4 个完全独立环境；这限制 effect-size 泛化，但不取消预注册的 sequence-level direction 复制。

本边界完成后，下一条真正有价值的路线是取得 independent person trajectory truth。没有独立真值前，不比较或选择更复杂 centroid，不做 deskew、route/event、Android 或生产推断。
