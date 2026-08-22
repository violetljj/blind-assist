# P1-PA3 automated public-dataset cohort result

状态：`NO_USER_CAPTURE_OR_LABELING / S0V4_SOURCE_LABEL_CONTRACT_MISMATCH_NOT_EVALUABLE / S0V5_GENERIC_DOOR_PARTIAL_BOUNDED_AVAILABILITY / ENTRANCE_CONFIRMATION_NOT_EVALUABLE / IDENTITY_NOT_AUTHORIZED`

## 结论

本轮纠正了“prospective 必须由用户拿设备现场采集”的过窄解释。PA3 所需 precedence 是 public Goal Contract
先于本项目 pixel/truth access；它可以由网络公开数据自动完成，不要求用户选图、标帧、跑命令或判断中间结果。

自动化链条先冻结 24 个同构产品目标：`帮我找入口 -> NAMED_BUILDING_ENTRANCE -> "building entrance"`，
`reference_mode=SET_VALUED`。随后才查询公开数据文件树，并以 relative image path 的 SHA-256 排序固定 roster；
RGB capture manifest 在 private label/mask access 前封存。provider 只接收当前 RGB 与该 pre-truth Goal Contract。

## S0v4：DeepDoors2 fail closed

DeepDoors2 README 声明 semantic mask 的 door/frame RGB 为 `(192,224,192)`。按该声明预冻结 exact-color truth
规则后，自动下载的 24 个固定 mask 实际只出现 `(0,0,0)`、`(128,0,0)` 与 `(0,128,0)`。因此本 cohort 终止为
`SOURCE_LABEL_CONTRACT_MISMATCH / NOT_EVALUABLE`；没有事后改颜色、换样本或运行模型。

## S0v5：DoorDetect mechanics

DoorDetect 官方仓库明确规定 `class 0 = door` 且使用 normalized YOLO bbox。C0、source lock 与 Git tree roster
全部先于 selected RGB 与 label bytes；1213 个 image/label pair 中按预冻结 path-hash 规则取 24 个。private label
materialization 得到 `8 VISIBLE / 16 NOT_VISIBLE`，达到 `>=5 visible episodes AND >=8 visible frames`，由机械授权门
生成唯一一次 YOLOE semantic-only execution receipt。

固定 provider 为 `YOLOE-26n-seg / Ultralytics 8.4.52 / imgsz=640 / conf=0.001 / max_det=100 / K=10`，
canonical prompt 仅为 `building entrance`。24 次 dispatch 全部完成，无 retry/replay：

| Primary IoU >= 0.30 | Result |
|---|---:|
| evaluable visible cases | 8 |
| Recall@1 | 4/8 |
| Recall@3 | 4/8 |
| Recall@5 | 4/8 |
| Recall@10 | 5/8 |
| terminal | `P1_PA3_PARTIAL_BOUNDED_SEMANTIC_TARGET_AVAILABILITY_ON_COHORT` |

这建立了一个窄但真实的结果：合法 goal-semantic prompt 能把部分真实 door region 放进 bounded pool，proposal
availability 不是普遍为零；`@10 > @1` 也符合 proposal 层“提高候选覆盖而非一次完成 identity”的职责。

## Claim ceiling

DoorDetect 的 class 0 官方定义为 room door，不是 building entrance truth。因此 S0v5 只能支持
`AUTOMATED_PUBLIC_GENERIC_DOOR_GOAL_SEMANTIC_PROPOSAL_MECHANICS`，不能升级为室外最后十米入口 confirmation、
identity verifier 授权、App 集成、产品或安全结论。入口专用公开候选源（例如 Mapillary entrance pipeline）的模型
预测也不能冒充 evaluator truth。后续审计确认 LSAA 的 door bbox 独立人工 truth provenance 不成立，因此没有使用；
全自动 FacadeElements successor 与 relation verifier 结果见
[`facade + relation result`](P1_PA3_FACADE_ENTRANCE_AND_GOAL_RELATION_RESULT_2026-08-22.md)。

## Evidence identity

- C0 receipt body SHA-256：`6be3589662b9f4e9e6c5d779e9ea4e500a362da5951b5a4ac2ef422436fdfc10`
- S0v5 roster body SHA-256：`bccc9476967e6b539c1eb2ea8401a9e0bce0469cd568192ef287ce73eb134785`
- public input SHA-256：`6f1a653d05f4556aa0674a1ce052cab49b00023315a5c9ed73dd5fec72900542`
- private input SHA-256：`88099073defbd43be5d18e70bf724bdd50a895f731e5fb15903d3d9a223e29b8`
- prediction SHA-256：`22d584940b31650fb1e6f388201c97cfe831a8ba9de3366755964c78edce8dcc`
- evaluation SHA-256：`d038dad5f690177981045760817ac7c535d46640f99632e5f59e094f1863e7ee`

Ignored evidence roots:

- `artifacts.local/evidence/p1_pa3_s0v4_automated_public_dataset_v1/`
- `artifacts.local/evidence/p1_pa3_s0v5_automated_doordetect_v1/`
