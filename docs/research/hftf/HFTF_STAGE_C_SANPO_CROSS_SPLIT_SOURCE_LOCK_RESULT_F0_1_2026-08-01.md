# HFTF Stage C SANPO cross-split source-lock result F0.1

日期：2026-08-01

终态：`F0_1_SANPO_CROSS_SPLIT_SOURCE_LOCK_VALIDATED`

## 1. 结论

F0.1 已在 media、geometry teacher、corpus 与 student outcome 全部未打开时固定
12 个 parent-session-disjoint sources：

| role | official split | session | source/target FPS |
| --- | --- | --- | ---: |
| train | train | `12b65d2c76d7ad0c17d7ac791089b8cae0bb059c9b02a6f23129044192bc93bb` | `20/10` |
| train | train | `12d0b0a8fc2df1fe8e22c66c9decaeef46fab24f3cda5ca8cae9b97e7f0c49a0` | `20/10` |
| train | train | `1405e451c6cf4a6c5fa04cc9dfc0046990d1717de26907a398e2bd60fcc1ef0c` | `5/5` |
| train | train | `1408ed7f8febaea19f7be0df2de0ea04f5a344b80a1d83cd541e062ecf67ef11` | `20/10` |
| train | train | `140b884287a0dbad0452c003d277fe7b31b4288b0e133eae331a023af25fbdcb` | `20/10` |
| train | train | `1480ad06d49b5c96380df79bbc71208a973143a06b2899e28403ff1b5bb4b583` | `5/5` |
| dev | train | `1486d76dddddbfc2143b40f5fc68cbb47dc9f2b1d777fddb7c4805c5ab916cc3` | `20/10` |
| dev | train | `14d7773cfffb452bc29bd67a6c04c30ca8963da3afdf4c51c6bbfd08ec30e707` | `20/10` |
| dev | train | `15ac9b955da6abc110de01901b903054170146a4ebfc91fd0b8977e43f173e0a` | `20/10` |
| heldout | test | `2d9fa74adbe115a5ad715b05515b33cb55eb69b1b168684c40106861e36c317b` | `20/10` |
| heldout | test | `9bee9c839dfd2855bfe4f45e1b39996095a64437262ea8366d438de898e69b9a` | `20/10` |
| heldout | test | `081eb747d3d529e01b367c5d950b3fa1ccda9903246f5435293caf46badae4c2` | `20/10` |

每条 source 都有 50 个连续 source-index-aligned RGB/mask/depth frames、有限 intrinsics
与 pose object receipt。5 FPS timeline 固定选 `0..24`；20 FPS 降到 10 FPS，
固定选 `0,2,...,48`。pose 后续必须按 `source_frame_index` 绑定，不能按 replay
timeline index 猜测。

## 2. 报告绑定

| report | SHA-256 |
| --- | --- |
| F0 same-split metadata plan | `790e5dae28f3ed37102d0dbde452f8432c6f0ae4d42892fe4342f0bbe781979f` |
| F0.1 cross-split metadata plan | `edaa63a86ff0254b0887d437086be9bda6f3c1b0aa3c3c9cbfc72bc05d5d0f55` |
| F0.1 exact source lock | `f7353779315757b8b4ca5ba13b3544c4348c25f2ac4daa4befe47ad80fc79f62` |

cross-split plan 与 source lock 均第二遍 byte-exact。正式 lock：

`artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-source-lock-20260801/source_lock.json`

## 3. Firewall

source lock 时：

- `geometry_outcome_read=false`；
- `teacher_outcome_read=false`；
- `student_outcome_read=false`；
- 只授权 exact media acquisition；
- teacher label/corpus 与 student training 仍未授权。

下一步必须先提交 split-aware、hash-bound importer，再按 exact lock 获取媒体。
test split 只可用于 heldout，不能进入训练、dev selection 或 normalization statistics。
