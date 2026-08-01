# HFTF Stage C D2 official-train 元数据资格筛选结果

## 结论

唯一一次 metadata-only 扫描成功锁定 6 条全新 official-train Development parents：

`D2_OFFICIAL_TRAIN_METADATA_COHORT_QUALIFIED`

这只证明存在满足冻结 source-shape 合同的 D2 cohort，不是 media、pose、geometry
teacher、student、mechanics、effect 或 safety evidence。当前只允许冻结下一份独立的
media/mechanics implementation contract，尚不允许执行。

## 锁定 cohort

| Rank | Session ID | Source fps | 5 Hz normalized source frames |
|---:|---|---:|---|
| 1 | `170afcede1bdc6eb955c432d15196db82f93b3c3ec136137ecf070c58eb61135` | 5 | `0..12` |
| 2 | `171469174b78aba85b3f52bca1547d3fe72b43140c550032e6a710640f943a00` | 5 | `0..12` |
| 3 | `17395cf62ccff67fd40fb885460e964beb971bb91d9d6311201f30813a66bd5c` | 5 | `0..12` |
| 4 | `17397b0026c4b8fb0cb271404f0b90865c47a7c2f2b887f7a5a3bfd272aeb6ee` | 20 | `0,4,...,48` |
| 5 | `1757cef62fc33b8571307c4873a0f82091cb6f047af8c3f21176348921f1f4da` | 5 | `0..12` |
| 6 | `17a9ed0a63fb28f97009e5ccc50547a188f1ee03c359cf9b05409854d8e7be7e` | 20 | `0,4,...,48` |

6 条 ID 严格升序、互不重复，并与冻结的 78-parent exclusion union 完全不相交。
它们从现在起锁定为 D2 one-shot Development mechanics cohort，不得 outcome 后追加或换源。

## 执行与证据

- exact execution contract 由 commit `335eb2630b3debac07cea9c38448f0b1cb3a8f3d`
  推送，并在执行前确认 `HEAD == origin/master`。
- `attempt.json` 在首个网络请求前落盘，SHA-256 为
  `b3547bc02c2f1a8e4633596681200ccc652a8cef0fe872ad4f0f8b5cafac0dc7`。
- `qualification.json` SHA-256 为
  `63a217c3e658bbe4fee9e351c5c9abf68379ec2ccb89a6c3449f1581e385ee47`，
  共 725476 bytes。
- 外层工具在 124 秒超时，但原 child process 未被终止。只监控原 PID，没有启动第二次
  CLI；原进程随后写出 durable qualification 并自然退出。因此这仍是一条 scan
  execution，不是重跑。

official-train split 共 1560 条。scan ledger 到第 149 个升序条目时凑齐 6 条：

- 71 条为冻结 exclusion union 中已遇到的 parents；
- 69 条 candidate metadata 请求在 3 次内部 retry 后为 404；
- 2 条为 candidate metadata invalid argument；
- 1 条 fps 不等于 5 或 20；
- 6 条合格。

尚未扫描到的另 7 条冻结 exclusion parents 不影响完整 78-parent exclusion union；
选择在第 6 个合格 parent 出现时按合同立即停止。

## 离线核验

主审离线重算并全部通过：

- terminal、6/6 counts、ID 升序/唯一/排除集不相交；
- ledger 严格升序并恰好选择前 6 个 eligible parents；
- 6×3 个 modality receipt-list hashes；
- 每种 modality 的 source frames `0..49`、generation、正 size 与 MD5；
- 5/20 fps timeline、有限正 intrinsics、description/pose object receipts；
- attempt/qualification bindings、contract hash 与全部 firewall。

独立只读审计另行重算 13 项 bindings、900 个媒体对象 receipts、18 个 canonical
modality receipt hashes、完整选择序列与权限防火墙，结论为 `CLEAR`，无 blocker。

## 权限边界

本次只读取 official split、description JSON、pose object receipt 与 media object
listings。RGB、panoptic mask、metric depth、pose 内容、teacher/student outcome 均未读；
reserved official-test 未开。

不允许重扫、补充或替换 cohort。成功结果只允许冻结独立的 D2 media/mechanics 合同；
媒体采集、pose 内容、geometry teacher、student、D2 effect、研究主线、App、Android、
生产与安全权限继续关闭。
