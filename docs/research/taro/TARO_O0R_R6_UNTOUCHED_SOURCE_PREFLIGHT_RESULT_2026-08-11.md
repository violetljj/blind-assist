# TARO O0R R6 untouched source preflight result

状态：`DATA_USE_LOCK_VALID / HEAD_24_OF_24_PASS / DOWNLOAD_INTEGRITY_PASS / EXACT_120_FRAME_INVENTORY_PASS / MODEL_AND_TRUTH_NOT_RUN`

## Exact cohort

按冻结的 repository exclusion snapshot 和 salted SHA-256 规则，从 ARKitScenes Training 选出 8 个从未进入
R4/R5/R6 formation 的 visit-disjoint parents：

| visit_id | video_id | exact pose-bounded frames |
|---|---|---:|
| 467175 | 47333514 | 16 |
| 467312 | 45261569 | 14 |
| 435329 | 42899445 | 8 |
| 423306 | 42897745 | 13 |
| 466652 | 45261100 | 11 |
| 469650 | 47333562 | 24 |
| 470439 | 47115427 | 5 |
| 469830 | 47334055 | 29 |

合计 120 frames。8 个 parents 均在官方 upsampling/raw/3DOD split 中一一存在；selection 未读取媒体、模型或
truth outcome。

## HEAD 与 source integrity

24 个 exact Apple 官方 URL 全部返回 200 和正 Content-Length；只有 24 次 HEAD attempt，response body 为 0。
总压缩字节 `318,241,411`（约 303.5 MiB），其中 upsampling `312,586,545`、intrinsics `5,233,774`、
trajectory `421,092` bytes。

随后 one-shot GET 下载 24/24 assets。每个文件均与 HEAD 的 URL、Content-Length、ETag/Last-Modified 绑定，
并重算 SHA-256/CRC；总字节仍精确为 `318,241,411`。没有覆盖或 rerun。

container inventory 对全部 ZIP 执行安全路径、compression、CRC 和 member timestamp 检查；uncompressed/source
materialized 计量为 `390,499,454` bytes。只解析 member identity、`.pincam` stem 和 trajectory 时间轴，没有把
PNG 解码成像素数组，没有读取 FARO/AppleDepth 数值，没有模型输出或 task metric。

关键 evidence：

- HEAD receipt SHA-256：`C5F47DE63B590E18126E31E65F8CB16DCC3B58D02CF5C2310A8DA865D3CE77F8`
- download receipts SHA-256：`E1698F10ADFB7A667EF95D0A7ADFEAC90BE585309D5D2F24BE680552A8F86C64`
- download result SHA-256：`E3E79DAB37AFF85D4CF2317F3E887D20CB6B003CDBAB2DA3AA34E7E89520E90E`
- exact frame plan SHA-256：`69352D6A940111E738488AA25CFAB8A924658B8C5720D9CE7A50AC612558D6A8`
- inventory sealed content SHA-256：`09A1AB17CF9921432B7D484F794C9D26F23729D8B934DB5108D3C1B5F4498D1C`

## 唯一后继

`TARO_O0R_R6_UNTOUCHED_CONFIRMATION_EXECUTOR_IMPLEMENTATION_LOCK`

只允许把已冻结的 R5 candidate/source/truth mechanics 泛化到上述 exact 8 parents / 120 frames，并绑定 R6
factor compositor。实现必须保留两阶段 firewall：全部 candidate + source-only ownership seal 完成后，才可首次
读取 FARO/task metrics；不得改变 roster、frame tokens、DepthART identity、factor ownership、8-parent floor 或 gates。

本结果不是 confirmation PASS，不证明 formal O0R、external generalization、deployment、product 或 safety。
