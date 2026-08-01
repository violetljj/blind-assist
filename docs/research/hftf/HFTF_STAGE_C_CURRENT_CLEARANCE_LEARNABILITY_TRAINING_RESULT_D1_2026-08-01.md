# HFTF Stage C G0-D1 Development training result

日期：2026-08-01

终态：`G0_D1_SIX_FINAL_CHECKPOINTS_FROZEN`

## 结果

Development corpus 已通过独立验证：9 个来源固定为 6 train + 3
model-selection，共 225 个 current-only records。validator 重新绑定真实
manifest/RGB/depth/mask/pose/authority 并重推全部 labels；UNKNOWN→SAFE 违规为 0。
student samples SHA-256 为
`d707613109878ed11e573429e39124b819264b3939a7989e3f22189379c7372f`，
corpus validation SHA-256 为
`d20b6afa10625ef5edbfb7823be2aaa32a0ef1847ce43ae9e3531c0071f8eb0b`。

两个 arms × 三 seeds 均完成 Phase A 与 Phase B 的 30 epochs，共 12 个 runs。
独立 training validator 重算 Phase A selection、核对 Phase B 固定 epoch、比较相同
seed 的跨 arm/phase initial states 与 loss 参数，并 strict-load/finite-check 全部
checkpoints。最终六个 Phase B checkpoints 为：

| Seed | Direct epoch / SHA | Clearance epoch / SHA |
|---|---|---|
| 17 | 24 / `c6256d5d…63cf3` | 13 / `b5e9dbe4…4eed2` |
| 29 | 22 / `73514643…0f560` | 11 / `248b9a32…2e415` |
| 43 | 21 / `ce65905d…b6323` | 20 / `d252f96f…320a` |

training validation SHA-256 为
`b1ed88a7f7a889035b2e47b5e4d0f38349505b1349ab16d6bdf3b44f52e62156`。

## 权限边界

该终态只说明六个 Development checkpoints 和 prediction parents 已冻结，允许下一步
另立 one-shot fresh execution contract。它不证明 signed-clearance 比 direct-risk
更好，也没有产生 fresh effect evidence。

三条 fixed fresh sessions 尚未获取/打开，三条 reserved official-test 继续关闭。
future/temporal、研究主线、默认 App、Android、生产与安全权限均不改变。
