# BA-ADT-REAL-EVIDENCE ADT-0 sample result

状态：`VALID / ADT0_SAMPLE_EPISODES_MINED / PARTIAL_EVENT_COVERAGE / FULL_SEQUENCE_SELECTION_NEXT`

## 结论

官方 10 秒 ADT sample 足以验证下载、身份校验、RGB/GT 权限分离和 GT-only Goal Episode Miner，
也自然包含持续可见、首次发现、候选丢失/重捕获和接近事件。但它没有提供单一目标的完整
`search → acquire → track → lost → reacquire → approach` 六阶段 episode，因此不能直接成为第一版
完整 demo 的目标片段。

这不是 ADT 数据路线失败。它把下一步收窄为：保持事件定义不变，从 Dataset Explorer 选择少量
完整 sequence，先下载 GT 做 ADT-0 mining，再只为入选 episode 下载对应真实 RGB。

## 输入与防火墙

- sequence：`Apartment_release_golden_skeleton_seq100_10s_sample_M1292`；
- RGB system-input side：官方 preview MP4，7,587,010 bytes，SHA-256
  `a8382faa5cd2b5b7ef73c0a57d06483a2e8d4b5c1913678b3e0f9229790f726e`；
- evaluator/mining side：main-groundtruth ZIP，6,673,432 bytes，SHA-256
  `e06fc93fbff2844631765d80eab5939304191bf12cc63eb6b69e3e2787190c23`；
- acquisition total：14,260,442 bytes；下载后 SHA-1 与官方 manifest 一致；
- miner 读取 RGB/VRS 次数：`0`；GT 不得进入未来 RGB estimator。

## ADT-0 结果

| 项目 | 结果 |
|---|---:|
| RGB GT frame timestamps | 300 |
| 带 bbox 的目标 | 106 |
| 至少 12 连续可见帧的候选 | 102 |
| `SEARCH` 候选 | 18 |
| `ACQUIRE` 候选 | 102 |
| `TRACK` 候选 | 102 |
| `LOST` / `REACQUIRE` 候选 | 35 / 35 |
| `APPROACH` 候选 | 2 |

`GlassCabinet` 是 sample 中事件覆盖最完整的目标：`SEARCH / ACQUIRE / TRACK / LOST /
REACQUIRE`，但距离下降未达到 approach 门。`BlackCeramicMug` 和 `WhiteVase` 分别产生约
`0.34 m` 和 `0.27 m` 的 approach 候选，但在这段 sample 中全程可见，不含 search/reacquisition。

`LOST/REACQUIRE` 在本阶段只是基于连续 GT visibility 的 episode 候选；没有人工确认其语义是遮挡、
出视野还是 visibility threshold 抖动。不得把 35 个候选写成 35 次已确认 tracker failure。

## 可复现入口

命令见 [BA-ADT Real Evidence module](../../../scripts/research/ba_adt_real_evidence/README.md)。本地结果：

```text
artifacts.local/evidence/ba_adt_real_evidence/sample/acquisition.json
artifacts.local/evidence/ba_adt_real_evidence/sample/episodes.json
```

`episodes.json` SHA-256：
`1c90df9cbc4e5e253fe9b16fa701fe1b36f33d0db3c67358da9c2c4254b0d27f`。

## Claim ceiling 与下一步

当前只建立 `ADT sample GT-only episode suitability`，未运行 RGB detector/tracker，未产生 bearing、
nearness、approach 或 observation-quality prediction，也未接 Goal Copilot policy。不能声称真实视觉能力、
离线副驾有效性、闭环导航、真实用户安全或默认 App 权限。

唯一下一步是 `ADT0_FULL_SEQUENCE_SELECTION`：复用固定 miner 和门槛，对少量完整 sequence 的 GT
做 outcome-transparent Development mining；选择自然覆盖更多阶段的对象后，再为其下载 RGB 并进入
ADT-1。Sky 继续关闭。
