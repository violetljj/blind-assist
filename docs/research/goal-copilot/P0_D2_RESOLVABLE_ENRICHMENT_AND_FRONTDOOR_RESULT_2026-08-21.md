# P0-D2 resolvable enrichment and calibration frontdoor result

状态：`ENRICHMENT_COMPLETE / DATA_FRONTDOOR_INSUFFICIENT / FIT_NOT_AUTHORIZED / NO_V3_PROMPT / NO_SKY / NO_SCIENTIFIC_VERDICT`

日期：2026-08-21

## 结论先行

本轮没有训练 calibrator。按 outcome-blind metadata roster 采集并用冻结 Grounding DINO/materializer 处理 60 个
Brussels acquisition frames 后，score-blind review 最终形成 16 episodes / 8 个与旧 cohort 地点名、frame 均零重叠的 venue
parents：`UNIQUE=4 / SET_VALUED=0 / AMBIGUOUS=12`。4 个 UNIQUE 来自两个新 parents：`9 et Voisins` 3 帧、
`Azra` 1 帧，均为可见品牌/招牌与单一物理门直接绑定。

合并旧 47-goal cohort、D1 24-goal confirmation 与本轮 enrichment 后，consumed Development 共 87 episodes /
29 venue parents：

| resolution | episode-independent venue parents |
|---|---:|
| UNIQUE | 11 |
| SET_VALUED | 2 |
| AMBIGUOUS | 23 |
| resolvable union | 11 |

冻结前门要求 `UNIQUE>=8 / SET_VALUED>=4 / AMBIGUOUS>=12 / resolvable union>=12`。当前只失败两项：
`SET_VALUED 2<4` 与 `resolvable 11<12`。因此 learned selective predictor 和 conformal referent set 均
`FIT_NOT_AUTHORIZED`；这里停止比在 87 frames 上拟合一个机器学习版 V2 更诚实。

## Outcome-blind acquisition 与独立性修正

roster 只读取 Overture/OSM source metadata，要求单 place-to-building、1–2 个合法 entrance anchors、place confidence
至少 0.85，并按类别 family 限额；不读取 RGB、proposal、Brain 或 review outcome。40 parents / 70 anchors 被平衡为
`14/13/13` parents、`24/23/23` anchors 的三个 shard，各取 20 frames。

第一次把全部 anchors 交给现有 runner 时，在下载前触发 Mapillary 整 bbox 请求并返回 HTTP 500；空 run 目录已删除，
未产生 image/model outcome。按 anchor 数机械分 shard 后，60/60 frames 全部取得 proposals，三个 materializer 均可
确定性 replay。

review 后又执行两层零重叠检查：

- `Jimmy Fairly` 虽 building UUID 不同，但 venue name 与 D1 重复，整 parent 排除；
- `1062362652395915`、`934153035312803` 被新 anchors 重新取得但与 D1 frame 重复，相关 episodes 整体排除；
- 最终 target-name overlap `0`，frame overlap `0`。

这些修正发生在 calibrator fitting 和任何 Terra run 前；本轮没有调用 Terra、没有测试新 prompt、没有调整 detector
threshold 或 evaluator。

## Evidence

- outcome-blind roster SHA-256：`5006a2056f8e8b53bb544c6ae6c70b87c3b45e2f27b4b6d7baf2f3642a134cdd`
- reviewed enrichment cohort SHA-256：`8d42381c31b8037ae4666751882e66e3039940701eafd8e4bb4b9b0f6cdffb32`
- enrichment audit SHA-256：`3e5593df1003ae81816634a20a74959604724137567425e8e225f3e921e66470`
- data frontdoor audit SHA-256：`691d2f884214a9bc07e1e8efa87c595488e4f30a3f8f214c00736d1ec94d8431`
- roster：`artifacts.local/evidence/p0-s0/2026-08-21-p0-d2-resolvable-enrichment-roster-v1/roster.json`
- cohort：`artifacts.local/evidence/p0-s0/2026-08-21-p0-d2-resolvable-enrichment-cohort-v1/brain-cohort.json`
- frontdoor：`artifacts.local/evidence/p0-s0/2026-08-21-p0-d2-data-frontdoor-v1/frontdoor-audit.json`

## 下一动作

下一批 Development 数据只补当前缺口：至少两个新的、真实存在多个合法入口的 `SET_VALUED` venue parents，外加至少
一个新的 resolvable parent。必须继续保留 score-blind review 和 name/frame disjoint；不能把同一建筑多帧或多个
candidate boxes 当作多个 parents/referents。数据前门通过后，先物化 runtime feature ledger 并过 leakage/completeness
检查，再拟合协议中的 logistic 与 conformal arms。

Claim ceiling：`CONSUMED_DEVELOPMENT_DATA_ENRICHMENT_AND_SUFFICIENCY_ONLY_NO_CALIBRATION_OR_MODEL_PERFORMANCE_CLAIM`。
