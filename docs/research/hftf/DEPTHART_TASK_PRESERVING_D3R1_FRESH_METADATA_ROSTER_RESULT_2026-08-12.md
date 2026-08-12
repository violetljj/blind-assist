# DepthART task-preserving D3R1 fresh metadata roster result

终态：`D3R1_FRESH_METADATA_POOL_127_LOCKED_MEDIA_UNOPENED`

D3R1 已从 pinned Apple ARKitScenes Training split 锁定 127 个完全新 identity。生成时扫描
published pre-recovery commit `8d17a053dc6d345a688035cd298c49c70d36288f` 中
`docs/research` 的 1,516 个 JSON/Markdown 文件，命中 490 个官方 visit/session token；另把尚未
出现在文档扫描中的 TARO R10 32-parent fresh pool 作为 SHA-frozen firewall，追加排除 64 个
token。两组排除项 overlap 为 0，effective exclusion 为 554。

127 个 visit 与 127 个 session 各自唯一；对 workspace snapshot、TARO R10、旧 D3 exact-48、
D1、D2 和 sealed R2 的 overlap 均为 0。独立 replay 与 write-once artifact SHA 校验通过。

127 只是保守资源规划值，不是质量证据或通过保证；D3R1 仍沿用旧 D3 的 300-frame、0.5 秒
adjacent gap、0.25 秒 pose bracket 和 portrait `[1,3]` 门。本步没有发 HEAD、没有读取 body、
truth 或模型输出，也没有分配 TRAIN/DEVELOPMENT。

当前唯一 successor 为
`EXPLICIT_D3R1_SOURCE_SCOPE_REGISTRATION_FOR_EXACT_127_METADATA_ROSTER`。该门只允许登记精确
127 身份的 future Phase-A intrinsics/trajectory source-use scope；仍不自动授权 HEAD 或 GET。
