# Semantic Anchor Graph and Referent Belief V2 Result

状态：`DEVELOPMENT_STANDARD / SYNTHETIC_OCR_STAGE / SUBSTRING_FSM_CORRECT_3_TO_SAGE_R_12 / WRONG_LOCK_6_TO_0 / NONE_0_TO_3 / UNKNOWN_0_OF_3_TO_3_OF_3 / NATURAL_OCR_NOT_EVALUATED / DEFAULT_APP_UNCHANGED`

## 结论

`SEMANTIC_ANCHOR_GRAPH_AND_BELIEF_V2` 已把 V1 的“唯一 substring 命中 + 两帧防抖”升级为可运行的 SAGE-R
算法原型。在相同 synthetic OCR token、polygon、candidate geometry 与质量输入上，V2 同时利用：

- target-token 与 observed-token 的 partial lexical compatibility；
- same-line / token-order layout；
- sign 与 physical candidate 的 `ABOVE_CANDIDATE` 几何关联；
- scene-adaptive semantic distinctiveness，使重复出现的 `ROOM` 权重低于 `302`；
- candidate-set + `NONE` posterior；
- 低 observability 时不更新 posterior 的 `UNKNOWN`；
- viewpoint/token/quality source signature，抑制同一 burst 重复计票。

这条机制在预期的 hard semantic failure 上产生了明确受控增益，但证据停留在 OCR-output stage，尚未证明真实 OCR、
CameraX、自然门牌分布或开放世界 calibration。

| 指标（23 synthetic frames） | substring + 2-frame FSM | SAGE-R V2 | delta |
|---|---:|---:|---:|
| correct terminal frames | 3 | **12** | **+9** |
| target correct locks（14 target-present） | 3 | **9** | **+6** |
| wrong locks | 6 | **0** | **-6** |
| correct `NONE`（9 absent frames） | 0 | **3** | **+3** |
| low-quality `UNKNOWN` preserved | 0/3 | **3/3** | **+3** |

`NONE=3/9` 不是缺陷隐藏：6 个 remaining absent frame 是同一 directory-board OCR burst。V2 有意保持
`UNCERTAIN`，因为重复观测不增加独立信息；第 7 个 fresh wide view 才转成 `NONE`。Baseline 在同一 burst 第二帧开始
连续误锁 directory 所靠近的 A 门。另一个 directory-binding episode 中，baseline 把 exact `ROOM 302` directory text
误锁到 A；V2 先 `UNCERTAIN`，随后用 `ROOM 30?` 与 door B 的关系支持锁定 B。

## Cohort 与边界

固定 seed `302` 自动生成 7 个 mechanism-targeted episode：

1. 相邻 `ROOM 301 / 302 / 320`；
2. directory board 中出现 `ROOM 302`，实体门牌只有 partial `30?`；
3. `ROOM 302A` substring hard negative；
4. 高质量 target-absent；
5. 远距、模糊、斜视的 unreadable observation；
6. 同一 directory OCR 连续六次的 correlated burst，再给一个 fresh wide view；
7. `TARGET -> UNKNOWN occlusion -> fresh candidate reacquisition`。

这些 token 与 geometry 是确定性 synthetic Development evidence；没有运行 RapidOCR/PaddleOCR，也没有从相机测得
confidence、blur、perspective 或 token height。Cohort 在设计时已知目标 failure mechanisms，threshold 也是
Development constant，因此不能包装为 natural-distribution benchmark、独立 Confirmation 或已校准的 open-set 概率。
聚焦测试另外固定算法与阈值，仅改变 deterministic geometry jitter seed `302..311`；10/10 seeds 均保持 V2 correct
terminal 高于 baseline 且 wrong lock 为 0。这是局部生成扰动稳健性，不扩展 claim ceiling。

## 复现

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.test_semantic_anchor_graph_and_belief_v2

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.semantic_anchor_graph_and_belief_v2 `
  --run-dir artifacts.local/evidence/semantic-anchor-graph-belief-v2/run-20260824T191500+0800
```

Evidence：`artifacts.local/evidence/semantic-anchor-graph-belief-v2/run-20260824T191500+0800/`。

| 文件 | SHA-256 |
|---|---|
| `raw-decisions.json` | `afa9c1b21fa272a24328803b9b8dfe17bd98e55118f0d56aa9c381d8b632b6c3` |
| `final-report.json` | `c2c9f1886fffcff107df7de458f5ff22a71c0009f49013aadf15ecece02d35d7` |
| `result.html` | `f986ae2bda6fa9abf79a5fde130db6693c1c4b9dd5f78941ef89d50360f0d658` |

## 下一动作

保持 V4 active information gain 不启动。下一轮只把同一 V2 scorer/belief 接到现有 RapidOCR polygon 输出，并自动生成或
收集自然门牌式 `301/302/320/302A + directory board + target absent` OCR-stage Development cohort，检验 synthetic
关系增益能否跨到真实文字检测误差。若增益消失，优先定位 token grouping、candidate association 或 observability，而不是
回到 appearance model zoo 或用 tracker 创建 identity。

Claim ceiling：

`SYNTHETIC_OCR_STAGE_DEVELOPMENT_RELATIONAL_IDENTITY_AND_OPEN_SET_BELIEF_NO_NATURAL_OCR_CAMERA_ANDROID_NAVIGATION_SAFETY_OR_PRODUCT_CLAIM`
