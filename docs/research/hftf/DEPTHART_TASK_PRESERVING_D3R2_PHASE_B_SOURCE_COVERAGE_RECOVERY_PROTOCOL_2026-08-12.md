# DepthART-S D3R2 Phase-B source-coverage recovery

D3R2 是 D3R1 r0 停机后的独立恢复版本，不是 resume、repair 或同版本重跑。它只复用已经
SHA 冻结的 exact-32 身份顺序、exact-9,600 frame stems、exact-64 URL/HEAD 事实，以及既有
许可审阅事实；D3R1 r0 的 source bodies、局部 truth/support、operator probe 和 body activation
均不继承。

ARKitScenes 官方 RAW 格式说明明确写明，同步的 60FPS `lowres_wide / lowres_depth /
confidence` 并不保证每个 timestamp 都存在全部资产。因此恢复的第一步不是换邻帧或降低门，
而是对原 fixed-exact-300 计划做完整的 source-member coverage census。

## 本次冻结的门

- 身份与顺序保持 `32/32`，每身份仍为原 `300` stems，总计 `9,600`；
- 未来 census 仍是 exact `64` 个全新 GET，总声明体积 `5,580,879,686 bytes`；
- 不复用 D3R1 r0 已下载的四个 bodies，不发 Range，不跟 redirect；
- 只读取并独立复算 ZIP central directory/member names；不打开、解压或读取 member payload，
  不运行 `testzip`，`zip_crc_verified=false`；
- 不解码 depth/confidence，不派生 source truth，不评价 support，不做 Phase-B selection；
- 64 个 asset checkpoint 必须构成连续前缀；完成全部 64 后才能发布 census；
- 任一 asset 的 terminal / 三次重试耗尽会封存 failure receipt 与临时标记，使该 attempt 永久不可 resume；
- census PASS 仍然 `scientific_terminal=null`，下一门只是 missing-source policy registration。

推荐但尚未激活的后续语义是 `FIXED_EXACT_300_WITH_SOURCE_UNAVAILABLE_UNKNOWN`：原 300 stems
一根不删不换，只有 depth 与 confidence 同 stem 都存在的帧才具备 source observability；缺任一
modality 的帧记为独立的 `SOURCE_UNAVAILABLE_UNKNOWN`，不能计为 clear、occupied、negative 或
support。这是 availability mask，不是 variable-N 新 roster。coverage evaluability gate 必须等完整
census 后另行冻结。nearest-frame substitution 继续禁止。

## 当前权限

本次用户“授权”只覆盖另立 D3R2、登记 scope、冻结协议/实现/validator/synthetic tests 和一个仍
不存在的新 root 路径。没有创建正式 attempt root，也没有发 HEAD/GET/Range，没有读取旧 r0
body 或任何 ZIP directory/member，更没有打开 truth、RGB、模型、角色、训练、Development、R2、
性能、默认 App、production 或 safety。

唯一下一门：

`EXPLICIT_D3R2_PHASE_B_EXACT64_COVERAGE_ONLY_CENSUS_ACTIVATION`
