# TARO O1R R11 fresh 48-parent source-only Phase A implementation lock

状态：`TARO_O1R_R11_FRESH_48_PARENT_SOURCE_ONLY_PHASE_A_IMPLEMENTATION_LOCK_PASS / LOCKED_NON_EXECUTING / SCIENTIFIC_NOT_RUN / FORMAL_PHASE_A_NOT_RUN`

## 结论

R11 all-48 source-only Phase A runner 已独立实现并通过聚焦验证。它只为 exact `48 parents / 1,043
physical frames / 9,387 queries` 物化冻结 DepthART candidate、source receipt、prospective bundle、R6
reducer、R7 source feature、正式 R7 positive factor 与 R11 abstention factor。R7 与 R11 均只允许
`OCCUPIED_OBSERVED / UNKNOWN`；`UNKNOWN` 不作 negative，R11 positive 必须逐 query 为 R7 positive 子集。

本锁只签实现和 synthetic/metadata-only tests，不激活正式 Phase A，不创建 evidence root，不运行模型，
不读取 source ZIP member payload，也不执行 R9 parent scoring、top-24 selection 或 FARO Phase B。

## 冻结实现

- runner：`scripts/research/taro_o1r_r11_abstention_runtime/run_pool_phase_a.py`
- tests：`scripts/research/taro_o1r_r11_abstention_runtime/test_run_pool_phase_a.py`
- independent validator：`scripts/research/taro_o1r_r11_abstention_runtime/validate_pool_phase_a.py`
- validator tests：`scripts/research/taro_o1r_r11_abstention_runtime/test_validate_pool_phase_a.py`
- future module argv：`-m scripts.research.taro_o1r_r11_abstention_runtime.run_pool_phase_a`
- inventory：content SHA `35156C2901A4CBEEDB6D611A56ABE3D711CEB68EF932480C21428BA4FF741600`
- exact counts：48 parents、1,043 frames、9,387 queries；原子 terminal 前固定 `5F+3 = 5,218` files，最终
  root 固定 `5,219` files。
- future exclusive root：`artifacts.local/evidence/taro/o1r-r11-fresh-pool-phase-a-r0`
- resource ceilings：16 h wall、16 GiB RSS、12 GiB CUDA allocated、2 GiB evidence；network/training=0。

future execution lock 必须精确绑定：R11 protocol/authorization/pool/inventory formal PASS；DepthART source
commit、checkpoint bytes/SHA、model/preprocess/postprocess、seed 0、CUDA 与 float32 output；正式 Python/Torch/
CUDA/OpenCV/NumPy/timm/device identity；candidate input、prospective、public reducer、locked uncertainty
artifact/receipt、R7 source/R7 positive/R11 abstention、evidence writer，以及 runner 传递导入闭包内全部 repository
Python modules/package initializers；下一阶段只作 parent ranking 的 R9 selector content SHA
`67FD8430418E23E4C974EBA4D7F49DCBD4DE66164A16491DE76F05AC974796CC` 与 rule
`02CE016D6B0011F0`。Phase A 自身不得调用 selector 或形成排名。

## Capability-scoped source reader

正式 runner 不导入 R10/R1 orchestration，也不导入 `r6_confirmation_io` 或任何 FARO reader。专用
`PhaseAFrameRef` 的 payload capability 集严格等于：

- candidate phase：`color`、`intrinsics`；
- source-feature phase：`lowres_depth`、`confidence`。

inventory 中的 highres member metadata 必须验证存在且与 sealed member-index SHA 一致，随后在创建 frame
object 前丢弃；`PhaseAFrameRef` 无 highres member。reader 在调用 ZIP `getinfo/open/read` 前先验证 phase、role
与 capability，并分别记录 attempts、completed 和 bytes。正式 success ledger 必须为四个允许 role 各
`1,043` attempts/completed、合计 `4,172` ZIP member reads、48 trajectory reads、1,043 DepthART inferences、
1,043 candidate blob reloads，而 `highres_depth attempts/completed`、FARO value、truth、label、outcome、network
与 training 全为 0。

模型上采样数组一律命名为 `candidate_depth_highres_m`，不得与 source highres/FARO payload 混用。

## Barriers、seals 与 failure semantics

正式执行的 preflight 只验证 lock、小型 predecessor evidence、runtime/checkpoint 与 output-root absence；exclusive
root、sealed execution receipt 与 4 MiB 实体 terminal capacity reserve 创建后，才允许重新 hash 144 个 source
containers、读取 96 个 ZIP central directories 与 48 个 trajectory payload。trajectory 每 parent 只读一次，同一
bytes 同时用于 SHA/size 验证和 exact-ns parse，不得把 hash pass 与 parse pass 重复记为一次。之后严格执行：

1. 先完成并封存全部 1,043 个 candidate input/native blob/candidate record 与 candidate completion；
2. 再读取 Apple lowres/confidence，逐帧封存 source receipt 与一个 sealed gzip lineage；lineage 内含 prospective、
   reducer、R7 source、R7 base factor 与 R11 candidate factor；
3. 每个 candidate/source/lineage 在 completion 前从 evidence root 重载，重验 blob、content seal、nested runtime
   validator、9-query order、R11 subset 与 abstention 恒等式；
4. Phase-A completion content-sealed；最后一个写入动作是单一 content-sealed `terminal.json`，其中同时封装 result
   与此前全部 file receipts。不存在独立 success result/manifest/failure 文件，因此 success/failure 不能共存；若
   terminal 原子替换前失败，partial 会被改名、hash-bind 进 failure terminal。4 MiB 实体 reserve、4 MiB 逻辑 reserve、
   截断 failure message 与 final wall/RSS/CUDA/evidence guard 共同为 failure terminal 留出空间。RSS 使用 Windows
   `peak_wset`，不以当前 RSS 冒充峰值；VRAM probe/reset 失败必须 fail closed，不能降级记录为 0。

独立 validator 不导入 producer；正式运行后它必须从 inventory 重建 5,215 个逐帧预期路径，重新 hash terminal
内全部 5,218 个 prior-file bindings，解码并重算每个 native candidate、上采样 candidate depth、nested lineage
validators、hash sequences、per-parent counts、read ledger、barriers、result 与 5,219-file exact root set。它还要
独立复验完整 implementation binding map、predecessor/authority/selector、正式 runtime/candidate identity、source
commit/checkpoint，以及 source receipt 对 frozen inventory container/member bytes/decoded content、intrinsics、pose
与水位的逐帧一致性。该 post-execution validator 独立重读/解码 4,172 个允许 source payload，单独报告为 validator
read，不回写或混入 producer success ledger；highres/FARO payload read 仍严格为 0。

per-parent 统计始终同时保留 `visit_id + video_id`，不只以 visit ID 作 key；parent 迭代按冻结 roster 显式重建，
不依赖 `groupby` 相邻输入。R9 scoring 与 top-24 selection 字段在 Phase A completion/result 中必须为 false。

## 验证

- `python -m py_compile ...run_pool_phase_a.py ...validate_pool_phase_a.py ...tests`：PASS；
- Phase-A runner + independent-validator focused tests：`19/19 PASS`；
- 覆盖 exact counts/file formula、formal inventory seals、record mutation、frame capability、highres/跨 phase role
  在 ZIP lookup 前拒绝、role/path mismatch、trajectory single-pass、R7→R11 subset/abstention、递归 repository
  import closure、exact runtime/R9 next-stage bindings、OS peak-RSS/final-wall guard、实体 reserve 与原子 failure
  terminal、R7/R10 import-order 无全局污染，以及无 FARO reader/R10 orchestration import。

## 唯一 successor

`TARO_O1R_R11_FRESH_48_PARENT_SOURCE_ONLY_PHASE_A_ONE_SHOT_EXECUTION_LOCK`。

该 lock 必须在本 implementation commit 推送并确认位于 `origin/master` 后另行提交，绑定 exact implementation
commit、全部文件 SHA、正式 runtime identity、用户既有 exact R11 authority、上述 counts/barriers/read ledger/
budgets，以及 `overwrite=false / rerun=false`。execution lock 提交、验证并再次推送前不得运行正式 Phase A；
更不得执行 parent scoring、top-24 selection 或读取任何 highres/FARO member payload。
