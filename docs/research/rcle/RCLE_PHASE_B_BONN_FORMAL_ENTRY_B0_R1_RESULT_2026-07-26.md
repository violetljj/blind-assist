# RCLE Phase B Bonn Formal Entry B0 R1 结果

状态：`PHASE_B_B0_R1_INVENTORY_PASS_B1_METRIC_PROTOCOL_MAY_BE_FROZEN / VALID`

日期：2026-07-26

## 结论

B0 R1 唯一 canonical run 完成。固定六条 official Bonn archive 全部在第一次
GET attempt 完成，总计 `2,262,988,443` bytes；6/6 local archive SHA-256、
central-directory/member inventory、全 file-member CRC stream 与三个允许文本的
timestamp firewall 均通过。独立 validator 在单独 invocation 中完整复算为
`VALID`。

- claim SHA-256：`71f4d6a4786a53a6124fa2c508d78984db5ca6edff1130cb40cedc62e4d2d643`
- implementation lock SHA-256：`2faeb6529377e48ac10008cc233d1c34b32c108a9803402622d7b29e1c53c633`
- receipt SHA-256：`dc0ffe9a890b539478ff4c035b4dfadea6c21347a11b36f164810a18eb811f86`
- attempt ledger SHA-256：`c08f4d222227e8f96ac22b188e5cf3bee0f3030b614aa0158a652422f99fee73`
- transport：6 条 sequence 各 `1 × COMPLETE`，无 retry；
- stderr：`0` bytes。

## 固定 window denominator

| Rank | Sequence | 10s windows |
| ---: | --- | ---: |
| 1 | `rgbd_bonn_crowd2` | 2 |
| 2 | `rgbd_bonn_balloon_tracking` | 1 |
| 3 | `rgbd_bonn_balloon_tracking2` | 1 |
| 4 | `rgbd_bonn_moving_obstructing_box2` | 2 |
| 5 | `rgbd_bonn_balloon2` | 1 |
| 6 | `rgbd_bonn_moving_nonobstructing_box2` | 3 |

六条均有窗口，共 `10` 个固定、连续、不重叠的 10 秒窗口。未来 B1 若获授权，
必须完整保留这个 denominator，不得按画面、pose、support、metric 或结果换窗。

## Firewall 与权限

本轮没有 image decode、pose 后续 token 数值解析、static-map/legacy-result 读取，
也没有 raw/compensated expansion、scale proxy 或其他 Phase B metric。R0
preclaim HEAD 违规已独立公开并关闭，未计入 R1 gate。

B0 PASS 只开放“另立并冻结 B1 metric protocol”；不自动授权 B1 execution、
Kill Gate B、Phase C、Replay、Android、人体、安全或生产。
