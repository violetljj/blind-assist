# DepthART task-preserving D2R1 source-support result

状态：`PASS / 16_OF_16_SUPPORT_QUALIFIED / 4_TRAIN_4_DEVELOPMENT_SEALED / NO_MODEL_OUTPUT`

D2R1 按 Phase-A 固定的同一 16 个身份重新请求 intrinsics/trajectory/depth/confidence，
在全部 pose-derived continuous portrait runs 中按时间顺序扫描 300-frame window。HEAD 的
64/64 个资产可用，body 总量为 `2,847,223,483` bytes；执行共读取 19,341 个 source-support
frame 并在每身份第一个全门通过窗口处停止，合计测试 13,959 个窗口。16/16 个身份都在
未修改的 known/clear/occupied/grid/clearance-support 门下通过。

按冻结 Phase-A 顺序，前四个合格身份锁为 D2 TRAIN：`482610/47895972`、
`467109/47333324`、`421013/42444703`、`422855/42897480`。接下来的四个锁为 sealed
D2 DEVELOPMENT：`469596/47430877`、`434691/42898635`、`469646/47430802`、
`468094/47334356`。其余八个只保留 source-support-qualified 身份，不分配训练或评价角色。

执行没有读取 RGB、运行 DepthART/task head、保存 per-frame truth 或访问 R2。v1 checkpoint
sidecar 暴露了 Windows LF→CRLF 字节翻译问题；repair 没有改写 receipt，而是证明每份实际
CRLF 文件解析后与 manifest 完全一致，且规范化 LF bytes/SHA 与旧 sidecar 完全一致，再对
实际文件重新封条。该修复没有重算 truth、改变门限或角色。

本结果只关闭 D2 source-support 与身份角色门。唯一下一步是先获得 Phase-C exact-eight
`lowres_wide.zip` 的 HEAD-only 范围授权；在知道 RGB body 大小前，不授权下载、模型输出、
训练或 Development outcome。R2、性能、默认 App、production 与 safety 继续不授权。
