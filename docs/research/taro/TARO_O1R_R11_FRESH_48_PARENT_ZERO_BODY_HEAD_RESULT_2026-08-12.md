# TARO O1R R11 fresh 48-parent zero-body HEAD result

状态：`TARO_O1R_R11_FRESH_POOL_ASSET_HEADERS_AVAILABLE_MEDIA_UNOPENED / ONE_SHOT_CONSUMED / SCIENTIFIC_NOT_RUN`。

exact 48 个 Training parents 的 144 个冻结 URL 全部在第一次 HEAD 请求返回 `200 + positive
Content-Length`；redirect 与 transport error 均为 0，ETag/Last-Modified 为 `144/144`。总声明正文大小为
`2,960,390,828 bytes`，其中 upsampling `2,907,505,248`、intrinsics `48,933,412`、trajectory
`3,952,168`，低于 12 GiB ceiling。

执行只发送 HEAD，response/media body 为 0；没有 GET、下载、解码、DepthART、FARO、truth scoring 或
训练。exclusive root 已创建并消费，不得覆盖、替换、修复原位或重跑。head receipt、result、start receipt
和 manifest 已按 size/SHA 重算，144 行 attempt→final 与 request identity 全部一致。

该结果只证明 exact source assets 可达及其声明压缩字节数，不证明正文完整性、算法效果、部署、产品或
安全。唯一 successor 是 `TARO_O1R_R11_FRESH_48_PARENT_BOUNDED_SOURCE_DOWNLOAD_IMPLEMENTATION_LOCK`；
必须以本结果的真实 `2,960,390,828 bytes` 冻结 no-redirect 下载、HEAD/GET validator 绑定、checkpoint 与
exclusive source root，执行锁提交前不得发送 GET。
