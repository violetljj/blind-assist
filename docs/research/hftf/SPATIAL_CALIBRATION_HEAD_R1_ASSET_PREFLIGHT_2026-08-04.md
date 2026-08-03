# Spatial Calibration Head R1 asset preflight

终态：`SPATIAL_CALIBRATION_HEAD_R1_ASSET_HEADERS_AVAILABLE`

绑定 protocol `4A2083C96387817DA6E10EAEB09DFB573DCAE6F8EEEE88BE6B6E7EA9F23F198C`
与 24-parent roster。固定 HTTP HEAD 预检只读取对象元数据，不读取媒体 body：

- 20 个 development videos × 5 assets：100 个对象；
- 4 个 sealed videos × `lowres_wide.zip` identity-only：4 个对象；
- 总计 104/104 返回 HTTP 200 和 Content-Length；
- 总压缩长度 `10,087,539,019` bytes；
- sealed depth/confidence/trajectory 均未请求，sealed body bytes 为 0。

最终 result SHA-256：
`B61E33DC2318AC1B030214288CB566819D6050BA453BF955E62D1A14E490BAC3`。

首个 100-object preflight 有 6 个并发 SSL EOF。该结果原样保留；它没有 404，却被首版
错误归类为 cohort failure。控制面修复只对同一 URL 固定最多 3 次 transport retry，并将
持续 transport error 改为 `PREFLIGHT_INCOMPLETE`；没有换 video、换 asset 或读取 body。
修复后 104 个最终对象全部在第一次或固定 retry 内通过。

E: 在本次预检时仅约 12.31 GiB 可用，不能同时保留全部 archive 和完整解压内容。因此
下载实现必须逐 video 处理，只抽取冻结的 150 个 member，记录 archive SHA-256/ZIP CRC
后删除项目 `artifacts.local` 内的临时 archive；不得删 roster 外路径，也不得改抽样或替换
视频。媒体 GET 仍由显式许可 receipt 阻止，当前没有运行。
