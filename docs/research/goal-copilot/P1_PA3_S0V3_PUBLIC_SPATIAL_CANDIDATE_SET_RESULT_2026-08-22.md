# P1-PA3-S0v3 public spatial candidate-set cohort result

状态：`NOT_EVALUABLE_FOR_PA3_INFERENCE / 6_VISIBLE_EPISODES / 7_VISIBLE_FRAMES / PROVIDER_CALLS=0`

## 结论

S0v3 已把 12 个新的 museum product goals、C0 Goal Contract、OSM place/parent、bounded public entrance set，
以及无 entrance tag 时最多 4 个明确标为非 truth 的 building-frontage candidates，全部冻结在 Mapillary metadata、
project pixel、private truth 和 provider 之前。一次机械 geocoder query amendment 也发生在上述边界之前；没有改变
episode roster、place semantics 或选择规则。

统一 geometry-only acquisition 在 12 个 frozen goals 中物化 8 个 episode、22 帧。pre-provider 私有标注得到
`6` 个 visible episodes、`7` 个 visible frames。预注册的 PA3 authorization gate 同时要求至少 `5` 个 visible
episodes 和 `8` 个 visible frames；因此 frame denominator 差 1，PA3 inference 不授权，YOLOE、Grounding DINO、
MobileCLIP 与 identity verifier 的调用数均为 `0`。这不是 proposal provider 的负结果。

## 合同与证据

- goal receipt body SHA-256：`4ad91f997f2207ff69dd77e81df2e226bb13eb92fdab01cf01c21e25bc20ac36`
- acquisition body SHA-256：`ef85d94a3201be1e050744b654081c639d5f9c5b15c8df653a43cec9541dfcde`
- public spatial contract body SHA-256：`1b655ae9ced571d8a1fa59c989601994b6eff336196a3cc894da7ff230e7e08b`
- materialized public input SHA-256：`3845035be3f85f6de8ba09f9f4fbb867c5c1063699b9a53e5474be2ff2869378`
- materialized private input SHA-256：`61eea856a4e46b11fc6700ecd3d15993bdc22777a8bf33fc242bf4f0dfbb5cba`
- terminal：`P1_S0V3_NOT_EVALUABLE_FOR_PA3_INFERENCE_VISIBLE_FRAME_DENOMINATOR`

ignored evidence root：
`artifacts.local/evidence/p1_s0v3_spatial_candidate_set_confirmation_v1/`。其中 private truth 覆盖全部 22 帧；
materializer 成功复核所有 C0、spatial-contract、capture、truth 与 precedence binding，并机械输出
`pa3_inference_authorized=false`。

## 唯一 successor

停止 retrospective Mapillary resampling。下一 cohort 必须 prospectively 先记录 public product goal，再采集明确面向
目标入口的第一视角 observation，最后私有标 truth；在达到新的预冻结 denominator 前仍不得运行 PA3。即使未来通过
proposal gate，`SET_VALUED` 入口任务也只授权 candidate-to-legal-goal verifier，不自动授权 UNIQUE instance identity、
AMRM、App、产品或安全主张。

