# TARO O1R R11 selected-top24 FARO Phase B implementation lock

状态：`TARO_O1R_R11_SELECTED_TOP24_FARO_PHASE_B_IMPLEMENTATION_LOCK_PASS / LOCKED_NON_EXECUTING / FARO_NOT_READ / SCIENTIFIC_NOT_RUN`

## 结论

R11 selected-only FARO Phase B 的 label runner、完整 fixed-gate reducer 与不导入 producer/reducer 的独立 validator
已实现，并通过 19 个聚焦测试。本锁不读取 FARO，不创建正式 Phase B root，不形成科学终态；唯一正式执行必须在
implementation commit 推送 `origin/master` 后另立 one-shot execution lock。

## 冻结输入与 cohort

- top24 selection content SHA：`629ECF7069EE5942EAEF7946059CAD03D20D0F66CBD4DAF95E06A5315211A7B7`；
- exact selected cohort：24 parents / 674 physical frames / 6,066 queries；
- 每 parent frame count 与 identity 顺序取自 sealed selection，不允许替换或重排；
- 正式读取唯一 payload role 为 `highres_depth`，每 selected frame 恰好一次，共 674 次；unselected FARO read=0；
- Phase A 的 sealed source、R7 baseline 与 R11 candidate 必须在首次 FARO read 前全部按 selection lineage 重载验证。

## 冻结 reducer

truth `UNKNOWN` 永不进入 precision、recall、query-specificity 或 frame-specificity 分母。R11 `UNKNOWN` 对 definite
occupied 计 recall FN，对 definite clear 计“不输出 OCCUPIED”的 specificity success；这不把 truth UNKNOWN 当 negative。
candidate `OCCUPIED` 必须逐 query 是 R7 `OCCUPIED` 子集。

Evaluability 固定为：evaluable parents≥16、occupied parents≥12、definite occupied queries≥200、clear parents≥4、
clear physical frames≥12、definite clear queries≥20。只有通过 evaluability 才检查固定确认门：occupied precision≥.90、
其单侧 95% Wilson lower≥.80、micro 与 parent-macro occupied recall 均≥.90、相对 R7 两种 recall loss 均≤.01、
candidate FP≤R7 FP、query clear specificity≥.90、clear-frame specificity≥.90、其 Wilson lower≥.80、parent-macro
clear-frame specificity≥.90、clear outputs=0。terminal precedence 固定为
`EXECUTION_INVALID → NOT_EVALUABLE_DUAL_CLASS_COVERAGE → FAIL_FIXED_CONFIRMATION_GATE → WILD_LAB_RESEARCH_FACTOR_CONFIRMATION_PASS`。

abstention effect 单列：至少 2 个 rescued definite-clear frames 且覆盖至少 2 parents 才标为 evaluable；不足不影响绝对
candidate metrics 与确认资格。它不允许输出 CLEAR 或扩展产品/安全 claim。

## 冻结实现

- `phase_b_metrics.py`：19,064 bytes，SHA-256 `4D22C2CD18E129EF12D63F0D52B51F9A6A3F2A901DD9D2FA40F6E26A9F77300A`；
- `test_phase_b_metrics.py`：5,792 bytes，SHA-256 `34980BE7FA36F14CA7CDEFE0B3FBFC3CFD74831E1A633A4939AB705ED1E4FF17`；
- `run_selected_phase_b.py`：39,234 bytes，SHA-256 `E07E2A2CB2FF98B01AC99E82D07EE7AE07B4492219527B6738074C8C4233382D`；
- `test_run_selected_phase_b.py`：10,032 bytes，SHA-256 `91C30E0F198A26D6736FB328DD7BC218B8E6EB06CE98DD9157B40559FF05C7FA`；
- `validate_selected_phase_b.py`：27,151 bytes，SHA-256 `50BC7CA16D6BD4CACFB5C3E7EBCFC6CEEEDB011D22FFC58C1A81D56A09300040`；
- `test_validate_selected_phase_b.py`：3,275 bytes，SHA-256 `A40C21B7E5B4262B18BF23E25BBC3A566CA1AC65F4B070C8DE19882FC1E7A214`。

producer 复用同一 physical frame 的 FARO geometry 处理全部 9 queries，避免旧接口每 query 重建 1440×1920 geometry
的高开销，但 label 语义保持 `r7_canary` 的冻结 FARO truth 定义。正式成功 root 采用 sibling partial root 整体原子发布：
`execution-receipt.json + 674 labels + label-completion.json + result.json + terminal.json`，即 pre-terminal 677、最终
678 files；partial root 创建即消费。failure 也只可原子发布 `EXECUTION_INVALID` terminal。

## 验证

producer/reducer/independent-validator focused tests：`19/19 PASS`；覆盖 24/674/6066 cohort、payload role 在 ZIP lookup
前拒绝、selected read ledger、单-frame geometry reuse、UNKNOWN 分母、recall-loss、clear frame/Wilson/macro、FP、effect、
terminal precedence方向、implementation-commit byte binding，以及 validator 不导入 producer/reducer。`git diff --check`：PASS。

## 唯一 successor

`TARO_O1R_R11_SELECTED_TOP24_FARO_PHASE_B_ONE_SHOT_EXECUTION_LOCK`。

execution lock 必须绑定本 implementation commit、上述六文件、本锁、protocol/authorization/inventory、Phase A repaired
PASS、top24 exact four-file PASS 与全部代码闭包；正式 root 必须不存在。执行与独立复核完成后才可记录 R11 scientific
terminal。无论 terminal 为 PASS、FAIL 或 NOT_EVALUABLE，都不得重选 parent、改 candidate/threshold、重跑或升级 default App。
