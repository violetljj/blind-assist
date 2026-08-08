# BlindAssist Assistive Geometry B0 input/data preflight result

终态：`PARTIAL_PASS / SHAPE_SUPPORTED / SM_S9280_RUNTIME_GEOMETRY_OBSERVED / DATA_ROSTERS_UNRESOLVED / NO_TRAINING_AUTHORITY`

日期：`2026-08-09`

绑定合同：[B0 task contract](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_TASK_CONTRACT_2026-08-09.md)

## 1. 结论

`1×3×608×448` 已获得真实 DepthART-S metric checkpoint 的 PyTorch 与 ONNX graph shape
证据，不再是纸面尺寸假设：

- PyTorch 输入 `1×3×608×448`，输出 `1×608×448`，全部 finite；
- dynamic K 生成四级 camera prompt：`152×112 / 76×56 / 38×28 / 19×14`；
- model direct 与 external camera prompt wrapper 为 bit-exact，max/mean abs 均为 `0`；
- legacy ONNX export 与 checker 通过：`31,830,940` bytes、`2,823` nodes、
  `5×SelectiveScan`、`0×Acos`、`10×Einsum`；
- ONNX image 输入元数据明确为 `1×3×608×448`。输出保持 rank-3，但由于自定义
  `SelectiveScan` 没有 ONNX shape inference，名称仍为 symbolic dims。它不是 PyTorch
  rectangular support FAIL，也还不是 QAIRT/QNN rectangular PASS。

因此原 blocker：

```text
DEPTHART_1X3X608X448_PYTORCH_AND_EXPORT_SHAPE_SMOKE
```

收敛为：

```text
PYTORCH_RECTANGULAR_SHAPE_PASS
ONNX_RECTANGULAR_GRAPH_CHECKER_PASS
ONNX_OUTPUT_METADATA_SYMBOLIC
QAIRT_QNN_RECTANGULAR_NOT_EVALUATED
```

## 2. Shape evidence

- Receipt：`artifacts.local/evidence/hftf/assistive-geometry-b0-rectangular-shape-preflight-20260809-attempt-02/receipt.json`
- Receipt SHA-256：`62339D94EF437384239E30C4F59E88D79368D25ABFC16FD58A19C524850E09C6`
- ONNX SHA-256：`D6704E1871E969E5519849D1AAE7A953A7F954C457A5F8B90D635087D50FE1EB`
- Checkpoint SHA-256：`597631AC7AEAB8346F4DB013C3C65EF3203DF373E21C7265D7A147093C667E65`
- TF32：disabled；Torch：`2.11.0+cu128`；device：CUDA。

第一次 attempt 在 ONNX output metadata 仍为 symbolic 时把它误判为 static shape drift，
因此无成功 receipt。attempt-02 将“真实 PyTorch shape”与“ONNX metadata inference”分开，
保留该限制而不篡改 graph 输出类型。

本证据只授权 synthetic shape/camera externalization。它不证明任务质量、数据准入、QNN 转换、
HTP 执行、latency、默认 App 或 safety。

## 3. CameraX static binding

静态代码确认当前产品请求仍是默认后摄、`640×480`、4:3 优先、RGBA8888、
`KEEP_ONLY_LATEST`、24 FPS：

- `CameraXFrameSource.kt` SHA-256：
  `B6E247BF456C44ED37ACED394A385B904820D436DA2CA73CFF6AFCAD4A999195`
- r832 SHA-256：`8063935E5CA3C2567AAF67944DE3345B088629B9176E4556EC1A35EF573D73A9`
- r834 SHA-256：`DD38852E78DAE070F4956787D28C2CBB2B099E7BE80DEA6CED0F7BC0DFC0A898`

r832/r833 已于本轮在隔离 `hftf-depth-demo-app` 上通过，形成当前 SM-S9280 的
[runtime geometry receipt](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_RUNTIME_GEOMETRY_RECEIPT_2026-08-09.md)：
实际 full `640×480` crop、90 度旋转、`480×640` display，以及派生到 `448×608`
tensor 的动态 K 已明确。该证据仍是 benchmark-only，不证明正式 App runtime、真实重投影或
跨设备相同 mapping；实现不得把本次 K 常数硬编码为通用真值。

## 4. Data capability preflight

本轮只读取现有 metadata ledger 与 gap/role-conflict 摘要，没有打开新 RGB-D payload、truth、
teacher output 或 candidate outcome：

- `DATASET_MASTER_LEDGER.json`：`569,224,796` bytes，SHA-256
  `2227A6ADB23AEBDA5DC4222BBECCDB60FEC8E9ED1FBEB29C048219C1F70EBE7F`；
- `DATASET_GAPS.md` SHA-256：
  `F81741BB033570D4343534F6D6B5C6F44AD5FB24142ABE76F0990EE2A51B836E`；
- `SOURCE_ROLE_CONFLICTS.md` SHA-256：
  `5E83840B237A969F33AAAF0A4A8DD14C326E2C800517A0CD0D98648C879D355D`。

ledger 显示大量媒体和 manifest，但 path-derived role 不能替代路线级 admission；现有 role conflict
也禁止把 `fresh/reserved` 字样直接当成可用 cohort。当前没有选择 TRAIN/DEVELOPMENT/
CONFIRMATION roster，没有复用 consumed 120-frame cohort，也没有触碰部署 R2 的 8-session roster。

因此 data 结论是：

```text
ASSETS_PRESENT
ROUTE_SPECIFIC_LABEL_CAPABILITY_NOT_AUDITED
TRAIN_DEVELOPMENT_CONFIRMATION_ROSTERS_NOT_SELECTED
DATA_OUTCOME_UNOPENED
```

## 5. 未关闭 blocker

1. 正式 App runtime integration 与 real-reprojection receipt（本轮只关闭当前设备 benchmark observation）；
2. 新 TRAIN/DEVELOPMENT/CONFIRMATION roster 的 identity/license/ancestry/near-duplicate lock；
3. ground/clearance truth reader exact implementation/hash；
4. confidence threshold 与 B1 training hyperparameters；
5. ONNX output static metadata 或 QAIRT 对 symbolic output 的明确处理；
6. rectangular graph 的 QAIRT converter/QNN shape preflight。

## 6. 唯一 successor

```text
BLINDASSIST_ASSISTIVE_GEOMETRY_B0_DATA_CAPABILITY_AND_ROSTER_LOCK
```

当前设备 CameraX/K transform receipt 已关闭。下一步只从现有 metadata ledger 生成
label-capability candidate table，并在 payload 打开前冻结
  TRAIN/DEVELOPMENT/CONFIRMATION identity 与 license/near-duplicate 审计协议。

在这两者完成前，不训练 Assistive Geometry student，不运行 DA3/Metric3D teacher matrix，
不转换 rectangular QNN candidate，也不访问独立 task outcome。
