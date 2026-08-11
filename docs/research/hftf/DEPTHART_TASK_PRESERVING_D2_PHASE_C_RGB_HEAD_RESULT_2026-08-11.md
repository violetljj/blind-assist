# DepthART D2 Phase-C RGB HEAD result

状态：`PASS / 8_OF_8_RGB_HEADERS_AVAILABLE / BODY_UNOPENED`

对 D2R1 锁定的 4 TRAIN + 4 sealed DEVELOPMENT identity 执行了精确 8 个
`lowres_wide.zip` HEAD 请求。8/8 均返回可用 body 长度，RGB 总量为
`3,718,339,716` bytes，其中 TRAIN 为 `2,004,642,898` bytes，sealed DEVELOPMENT
为 `1,713,696,818` bytes。

本门没有读取 RGB body、重新请求传感器 body、运行模型、训练或打开 Development/R2。
结合既有 D2R1 HEAD，若要为 exact 300-frame window 机械物化 RGB/intrinsics/depth/confidence，
需要 32 个 ZIP body、总计 `5,281,655,713` bytes；本结果不授权该 GET。

唯一下一步是显式授权 exact-eight Phase-C body materialization only。该步只能逐 ZIP 下载并
按已锁 frame stems 提取源文件；不解码图像、不派生 truth、不运行模型、不训练。
