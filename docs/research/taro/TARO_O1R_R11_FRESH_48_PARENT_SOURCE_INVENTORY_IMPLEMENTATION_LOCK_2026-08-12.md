# TARO O1R R11 fresh 48-parent source inventory implementation lock

状态：`TARO_O1R_R11_FRESH_48_PARENT_SOURCE_INVENTORY_IMPLEMENTATION_LOCK_PASS / SCIENTIFIC_NOT_RUN / FORMAL_INVENTORY_NOT_RUN`

## 结论

R11 source inventory runner 已实现并通过聚焦验证。正式路径只读取两个 ZIP 的 central directory metadata 与
source-role trajectory：它不调用 `ZipFile.testzip()`、`ZipFile.open()` 或 `ZipFile.read()`，不解压任何 ZIP
member，也不读取、解码或解释 highres/FARO pixel payload。ZIP 内的 CRC 仅记录为 central directory
`declared_crc32`，不得写成 payload CRC 已验证。

本锁是 implementation evidence，不激活正式 inventory。正式结果、frame count、declared materialized bytes
和 inventory content seal 均必须首次出现在另行提交并消费的 one-shot execution result 中，不能把实现探针
观察反签为 frozen expectation。

## 冻结实现

- runner：`scripts/research/taro_o1r_r11_abstention_runtime/run_pool_inventory.py`
- tests：`scripts/research/taro_o1r_r11_abstention_runtime/test_run_pool_inventory.py`
- module argv：`-m scripts.research.taro_o1r_r11_abstention_runtime.run_pool_inventory`
- source root：`artifacts.local/datasets/taro/o1r-r11-fresh-pool-source-r0`
- future exclusive evidence root：`artifacts.local/evidence/taro/o1r-r11-fresh-pool-inventory-r0`
- parent/asset identity：exact `48 / 144`，绑定既有 R11 pool/request-plan 与正式 download PASS。

每个 ZIP metadata index 必须 fail closed 验证安全相对路径、symlink、encryption、compression、重复 member、
非负声明尺寸、video prefix、exact timestamp 和 recognized member 非空。frame plan 只取 color/highres-depth/
lowres-depth/confidence 的共同 timestamp、exact intrinsics timestamp 与 bounded trajectory pose；同纳秒 alias、
乱序/重复 token、无效 trajectory 或预算越界均拒绝。`UNKNOWN` 仍不作 negative。

## Phase firewall

正式 execution lock 必须精确冻结：

- `zip_index_mode=CENTRAL_DIRECTORY_METADATA_ONLY`；
- highres member metadata 可见，但 member payload read、member CRC validation 与 pixel decode 全为 `0/false`；
- 每 parent 只读取一次 `lowres_wide.traj` payload；
- network、source frame materialization、model、FARO value、truth、training、device、deployment、product、safety
  均为 `0/false`；
- 先完成仅 lock/evidence 的 preflight，再创建 exclusive root 和 sealed start receipt；144 个 source 文件的
  重新 hash/integrity validation 以及 ZIP/trajectory read 必须发生在 root 创建之后，任何后续失败均留下 sealed
  failure 与 manifest。

## 实现期边界事件

独立审计前的一次无输出实现探针复用了旧 `testzip()`，因而解压读取了 ZIP member bytes，包括 highres member。
该探针没有正式 lock/root/result，没有解码或返回 pixel array，没有读取 FARO 数值、运行模型、计算 selector 或
形成 top-24；其 read count 未 instrument，且所得 count/bytes/hash 全部禁止进入正式 execution lock。正式实现已
移除此调用，并由 monkeypatch regression 保证 `testzip/open/read` 任一调用都会使测试失败。

此事件不改写已消费的 download PASS，也不产生 inventory 或科学 evidence；它只冻结了后继正式 runner 必须为
central-directory-only 的更窄边界。

## 验证

- `python -m py_compile ...run_pool_inventory.py ...test_run_pool_inventory.py`：PASS；
- `python -m unittest scripts.research.taro_o1r_r11_abstention_runtime.test_run_pool_inventory -v`：`8/8 PASS`；
- 覆盖 roster/scope/container-byte/frame-order mutation、download evidence record replay，以及禁止 ZIP member
  payload access 的 monkeypatch regression。

## 唯一 successor

`TARO_O1R_R11_FRESH_48_PARENT_SOURCE_INVENTORY_ONE_SHOT_EXECUTION_LOCK`。

该 lock 必须在本 implementation commit 推送后绑定 exact commit、代码/协议/授权/download evidence、
central-directory-only policy、30 GiB declared-materialized ceiling、64 MiB evidence ceiling、2 小时 wall ceiling
与用户既有 exact R11 authority。提交并验证该 lock 前不得运行正式 inventory；更不得提前运行 Phase A、选择
top-24 或读取 FARO member payload。
