# P0-D3 targeted set-valued one-shot closure result

状态：`ONE_SHOT_COMPLETE / CURRENT_PUBLIC_DATA_SOURCE_INSUFFICIENT_FOR_CALIBRATION_DESIGN / P0_D2_DATA_FRONTDOOR_INSUFFICIENT / SET_VALUED_3_LT_4 / FIT_NOT_AUTHORIZED / NO_SECOND_BATCH / NO_MODEL_RUN / NO_SCIENTIFIC_VERDICT`

日期：2026-08-21

## 结论先行

P0-D3 已按固定 6-parent batch 一次性结束，不再补采。outcome-blind Overture roster 在任何新 Mapillary RGB
下载前冻结；Mapillary radius acquisition 得到 5/6 parents、20 frames，`UGent Campus Aula` 为 0 eligible
frame 并按协议保留失败、不替换。score-blind full-frame review 得到：

| resolution | 新 venue parents |
|---|---:|
| `UNIQUE` | 1 |
| `SET_VALUED` | 1 |
| `AMBIGUOUS` | 3 |
| `NOT_OBSERVED` | 1 |

合并全部 consumed Development 后为 92 episodes / 34 venue parents：`UNIQUE=12 / SET_VALUED=3 /
AMBIGUOUS=26`，resolvable union `13`。唯一失败门是 `SET_VALUED 3<4`；因此 Logistic/Conformal 均
`FIT_NOT_AUTHORIZED`。终态按预注册止损线固定为：

```text
CURRENT_PUBLIC_DATA_SOURCE_INSUFFICIENT_FOR_CALIBRATION_DESIGN
```

禁止第 2 批、禁止补第 7 个 parent，也不允许替换无覆盖或 review 后为 `AMBIGUOUS` 的 parent。

## 固定 batch 与 review

| parent | acquisition | score-blind resolution | 依据 |
|---|---|---|---|
| The Mint Brussels | 4 frames | `AMBIGUOUS` | 多租户玻璃商业立面；不能把各 tenant 门升级为 The Mint venue 的等价公共入口 |
| Galerie Bortier | 4 frames | `UNIQUE` | 名称位于单一公共拱廊入口上方；两侧为玻璃窗 |
| Groot Auditorium | 4 frames | `AMBIGUOUS` | 多栋相连建筑和多扇门，缺少 auditorium-to-door 绑定 |
| Rechtbank eerste aanleg Leuven | 4 frames | `SET_VALUED` | 同一公共台阶上三扇同形制正门，无员工/服务/紧急出口标识 |
| Erfgoedcel Leuven | 4 frames | `AMBIGUOUS` | courtyard 内多个机构与门，不能确定 Erfgoedcel 入口归属 |
| UGent Campus Aula | 0 frames | `NOT_OBSERVED` | radius query 无 eligible frame；未替换、未赋 visual truth |

roster 只使用 Overture taxonomy、place metadata 与 building footprint geometry；未读取 OSM `entrance=*`、
Mapillary pixels、proposal、Brain、score 或历史 resolution。采集后只读取完整原始 frame；本轮未运行
Grounding DINO、YOLO、Terra、V3、Logistic、Conformal、Sky 或任何拟合。

## Evidence

- roster SHA-256：`dc1f7849d04f5d767f366587d1679f798a5bc7ad746b75eb561d17de69c31e79`
- acquisition SHA-256：`86731a13c81a83c7b148806cf6b895c67da5d87c49a2fb7ce2a7f3f84a47be90`
- reviewed cohort SHA-256：`82c451dafa7383e0a08194dd35bd91f752adbaf815091092c3b7aef9283e0081`
- frontdoor SHA-256：`9379ccc0024ea8ee684bdf9a9a2b1239bf5b305f68608e9f5ea92b8b374c9656`
- review audit SHA-256：`c9152fb095ba68e22e4c080a7b0bc46551b2cac53076fe53736d60af7b5be53b`
- roster：`artifacts.local/evidence/p0-s0/2026-08-21-p0-d3-one-shot-roster-v1/roster.json`
- acquisition：`artifacts.local/evidence/p0-s0/2026-08-21-p0-d3-one-shot-acquisition-v1/acquisition.json`
- terminal review/frontdoor：`artifacts.local/evidence/p0-s0/2026-08-21-p0-d3-one-shot-review-v1/`

## 后续边界

关闭的是 P0-D2 V1 这套需要 `SET_VALUED>=4` 的 Logistic/Conformal claim，不是 P0 算法发现。现有 consumed
Development 可以继续支持低成本、低 claim 的 rule-based commitment baseline、Terra/V3 runtime feature usefulness、
candidate ranking、ambiguity detection、teacher pseudo-label 或 cheap Sky discovery；任何 winner 都必须在另立的有限
科学确认阶段验证，不能从本结果推出 calibration/model 性能。

Claim ceiling：`CONSUMED_DEVELOPMENT_ONE_SHOT_DATA_CLOSURE_ONLY_NO_MODEL_OR_CALIBRATION_PERFORMANCE_CLAIM`。
