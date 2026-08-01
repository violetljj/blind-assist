# HFTF Stage C D2 official-train 元数据资格筛选执行合同

## 冻结结论

本合同只授权一次 **metadata-only** 扫描：在 SANPO official-train split 中，按
`session_id` 升序，排除已经 burned、consumed、关闭的 G0-D1 cohort 以及保留的
official-test parents 后，锁定 6 条全新 Development parent session。

这不是 D1 路径修补，也不是 D2 媒体采集、geometry teacher、student 或 mechanics
执行授权。成功仅允许继续冻结下一份独立合同。

## 输入与实现绑定

- D2 design SHA-256：
  `06a8ff9cbe9c4c9b98cceeb7a36c69ba098f6f7d53ab980adb747b987a1728d9`
- T0 result SHA-256：
  `82c5bed9dc9210dadd615c36176174ad1043ed4860c54b04941f78083075ac7b`
- official-train split generation：
  `1692794964120907`
- official-train split text SHA-256：
  `f9c5dc4c289fa87342abc0d2cc49f112fcc78c7e02e0b6b081e296a99344173c`
- exclusion union：5 个互斥类别，共 78 个 parent。
- planner SHA-256：
  `4d8b206c887352d92c15cb3fe375d357551861c5e0a6113073a7426f332da58a`
- planner test SHA-256：
  `c09c2d34a5b521eaf20b1da7d43151e1b5ec975aa08bb0df7ca3b5d833346140`

CLI 在首个网络请求前必须证明：合同与 planner 都是当前 `HEAD` 中 tracked、
unstaged/staged 均无漂移的文件，并且 `HEAD == origin/master`。合同绑定的全部父证据、
split receipt、实现 hash、canonical output 与权限防火墙任一不匹配，都 fail closed。

## 唯一允许读取的内容

扫描可读取：

1. generation/SHA 绑定的 official-train split 文本；
2. candidate 的 `description.json` 对象 receipt 与 JSON 内容；
3. `camera_poses.csv` 的对象 receipt，但不得读取其内容；
4. RGB、panoptic mask、metric depth 的对象 listing receipts，但不得读取媒体 bytes。

合格 candidate 必须是 `camera_chest/left` synthetic session，fps 精确为 5 或 20，
相机内参有限且为正，三种媒体 listing 都覆盖 source frames `0..49`。description、
pose 和每个要求的媒体对象都必须有 generation、正 size 与 MD5。

## 固定选择与失败规则

选择只使用按 `session_id` 升序的 official-train split，不使用 split 原始顺序。
candidate 在 helper 内部 3 次 retry 后出现 404、metadata 不可用或不合法，记入
scan ledger 为 ineligible，并继续固定顺序；这属于预先冻结的资格规则，不是 outcome
后的换源。

CLI 在首个网络请求前先创建不可覆盖的 `attempt.json`；即使 split 请求失败或进程
中断，canonical root 也会阻止再次扫描。扫描一旦执行就不重跑、不追加、不替换。
完整 split 扫描后不足 6 条，终态为
`STOP_NO_ELIGIBLE_NEW_DEVELOPMENT_COHORT`。split receipt/SHA 漂移或 split 级请求失败，
终态为 `D2_METADATA_QUALIFICATION_NOT_EVALUABLE_NO_RETRY`。成功终态为
`D2_OFFICIAL_TRAIN_METADATA_COHORT_QUALIFIED`。

## 唯一执行命令

只有在本合同、planner 与测试提交推送，并再次确认 `HEAD == origin/master` 后，才可运行：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/plan_stage_c_d2_official_train_metadata.py `
  --execution-contract docs/research/hftf/HFTF_STAGE_C_D2_OFFICIAL_TRAIN_METADATA_QUALIFICATION_EXECUTION_CONTRACT_2026-08-02.json `
  --retries 3 `
  --output artifacts.local/evidence/hftf/stage-c-d2-official-train-metadata-qualification-20260802/qualification.json
```

## 权限边界

结果不是模型或 safety evidence。无论终态如何，当前节点都不授权媒体下载、pose 内容读取、
geometry teacher、student、D2 mechanics、reserved official-test、研究主线、默认 App、
Android、生产或安全声明。只有 exact 6 parents 合格，才允许冻结另一个一次性
Development media/mechanics 合同。
