# TARO O1R R11 fresh 48-parent source inventory result

状态：`TARO_O1R_R11_FRESH_POOL_INVENTORY_AND_FRAME_PLAN_PASS / PHASE_A_READY / SCIENTIFIC_NOT_RUN`

R11 source inventory one-shot 已按冻结 module argv 消费。exact `48/48` parents 均至少有一个
pose-bounded exact frame，共 `1,043` frames；compressed source 为 `2,960,390,828 bytes`，central-directory
声明展开量为 `3,540,113,101 bytes`。未替换 parent。

正式 evidence root 恰好包含四个文件，共 `95,681 bytes`；inventory content seal 为
`35156C2901A4CBEEDB6D611A56ABE3D711CEB68EF932480C21428BA4FF741600`，result content seal 为
`C4F15A3EA4DC1C51463860B9510658620BA49086116F63EB9514FF89F9A494B1`，manifest content seal 为
`59A1B3180E467266E16330D87C256F5D57B8D3C9BC2111DA9CD060DC043C01B8`。

独立复核未导入 producer module，重算了四个 content seals、manifest 三个 file bindings、48-row roster、
1,043 个 exact-ns token 的排序/唯一性、count/byte sums，并逐一核对 144 个 container binding 与 sealed
download receipt 的 bytes/SHA-256，全部一致。

## Phase firewall

- `zip_index_mode=CENTRAL_DIRECTORY_METADATA_ONLY`；
- ZIP central-directory metadata operations `96`，trajectory payload reads `48`；
- ZIP member payload reads `0`，highres-depth member payload reads `0`；
- member payload CRC validation `false`，记录的 CRC 仅为 central-directory declaration；
- pixel decode、source-frame materialization、model、FARO value、truth、training、network 均为 `0/false`。

本结果只证明 source-container metadata inventory 与 all-48 Phase-A readiness，不是 task metric、算法效果、
路线晋级、部署、产品或安全证据。

唯一 successor：`TARO_O1R_R11_FRESH_48_PARENT_SOURCE_ONLY_PHASE_A_IMPLEMENTATION_LOCK`。只能先实现并验证
all-48 source/DepthART Phase A、全量 seal 与 FARO=0 firewall；不得直接运行 Phase A、top-24 selection 或
selected FARO Phase B。
