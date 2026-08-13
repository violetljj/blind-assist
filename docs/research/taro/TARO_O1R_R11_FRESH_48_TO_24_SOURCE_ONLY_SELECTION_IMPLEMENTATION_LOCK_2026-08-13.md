# TARO O1R R11 fresh 48-to-24 source-only selection implementation lock

状态：`TARO_O1R_R11_SOURCE_ONLY_TOP24_IMPLEMENTATION_LOCK_PASS / LOCKED_NON_EXECUTING / FORMAL_SELECTION_NOT_RUN / FARO_NOT_READ`

## 结论

R11 source-only 48→24 selection runner 与不导入 producer 的独立 validator 已实现并通过聚焦验证。本锁只冻结
实现，不执行真实评分，不创建正式 evidence root，也不读取 highres/FARO/truth/label/outcome。正式选择必须另立、
提交并推送 one-shot execution lock 后，才可按冻结 argv 执行一次。

## 冻结问题与算法

- 输入固定为 Phase A 已封存的 48 parents、1,043 source records、9,387 query features；
- 复用冻结 R9 selector `TARO_R9_SOURCE_ONLY_CLEAR_ENRICHMENT_GRID_SEARCH_V1`，selector content SHA
  `67FD8430418E23E4C974EBA4D7F49DCBD4DE66164A16491DE76F05AC974796CC`，rule
  `02CE016D6B0011F0`；
- 每 parent 的分数为通过冻结 rule 的 eligible query 数；按 eligible count 降序，再按
  `canonical_sha256([visit_id, video_id])` 升序消除并列，取前 24；
- `UNKNOWN` 只作为 selector 的候选状态，不作 negative；本阶段不输出 `CLEAR`，不计算任务效果；
- 必须先重验并封存全部 48 个 parent scores 与 top-24 identity，之后 selected-only FARO Phase B 才可获得读取权。

## 冻结实现与边界

- producer：`scripts/research/taro_o1r_r11_abstention_runtime/run_top24_selection.py`
  （53,987 bytes，SHA-256 `3ED2CF8712C98290B0D40A75FAF47081B7113F009A3740731F107E8D61C0B0B0`）；
- producer tests：`scripts/research/taro_o1r_r11_abstention_runtime/test_run_top24_selection.py`
  （18,025 bytes，SHA-256 `6B20B3144018674228629A303C331FCE047890979B955B0276E929977378A0A9`）；
- independent validator：`scripts/research/taro_o1r_r11_abstention_runtime/validate_top24_selection.py`
  （37,244 bytes，SHA-256 `E34029C9F7B09D15745FA3CE3C6F832F7A4E096A2B2F73CE5E2940D8363646A9`）；
- validator tests：`scripts/research/taro_o1r_r11_abstention_runtime/test_validate_top24_selection.py`
  （16,808 bytes，SHA-256 `F3F056DBDFF2C3E6B48CA30B99F23E90672506C1AFD53FFE0B2E317F47FEC5BA`）。

execution lock 必须绑定上述四文件、本锁、Phase A repaired PASS、inventory、R9 selector，以及 producer/validator
使用的 source adapter、materializer、candidate-scale、reducer、R7/R10/R11 与 evidence-writer 代码闭包。除本地
artifact bindings 外，每一 binding 的当前 bytes 必须与 `implementation_commit` 中 `git show` 的 bytes 完全相同；
implementation commit 必须已经位于 `origin/master`。

正式 argv 固定为：

```text
-m scripts.research.taro_o1r_r11_abstention_runtime.run_top24_selection --execution-lock docs/research/taro/TARO_O1R_R11_FRESH_48_TO_24_SOURCE_ONLY_SELECTION_ONE_SHOT_EXECUTION_LOCK_2026-08-13.json
```

正式 root 固定为 `artifacts.local/evidence/taro/o1r-r11-fresh-pool-top24-selection-r0`，创建即消费；成功时精确
四文件：`execution-receipt.json`、`parent-scores.json`、`selection.json`、`terminal.json`。terminal-last、4 MiB
实体 reserve、failure terminal、2 h wall、8 GiB peak RSS、16 MiB evidence ceiling 均 fail closed；network、model、
training、source ZIP member、highres、FARO、truth、label、outcome read 全为 0。

## 验证

聚焦 producer + independent-validator tests：`20/20 PASS`。它们直接覆盖 48→24 deterministic ranking/tie、
重新签名后的 score/selection/ledger 篡改、source-only public API、R7→R11 ordered subset、Phase A repaired PASS、
implementation-commit byte binding、exact four-file root、peak resource/final wall 与原子 failure terminal。
`git diff --check`：PASS。正式 top-24 root 与 execution lock 在本锁创建时均不存在。

## 唯一 successor

`TARO_O1R_R11_FRESH_48_TO_24_SOURCE_ONLY_SELECTION_ONE_SHOT_EXECUTION_LOCK`。

该 execution lock 必须在本 implementation commit 正常推送 `origin/master` 后单独创建并推送。完成唯一正式选择及
独立复核前，不得读取 FARO；选择封存并复核 PASS 后，唯一后继为 selected-top-24-only FARO Phase B implementation。
