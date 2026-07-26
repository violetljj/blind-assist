# RCLE Phase B Bonn Formal Entry B0 设计审查结果

状态：`DESIGN_REVIEW_PASS / EXECUTION_AUTHORIZED / NOT_STARTED`

日期：2026-07-26

## 结论

独立只读复核通过。B0 design lock：

```text
a0b04ac5af2976f921169769179c84922574c000fac31322a0d75caad9b0c757
```

预注册：

```text
9a20e780bf25f554597a993b74ca884b94322679c5b428bb2d6dee9a57da2601
```

上游绑定的 R3 canonical metadata authority receipt：

```text
05a283b84f62bee000447bb567eadd63b424afaa9d81f5f0d83d36a9ed02489b
```

## 复核范围

- 固定 6 条 sequence、rank、official URL 与 `2262.9 MB` display-size 分母；
- 禁止替换 sequence、mirror 或加入第七条；
- network 前 exclusive claim，最多一个 B0 run；
- 同一 run 内每个 URL 最多三次 bounded transport attempt；
- 所有 ZIP member bytes 只可为 CRC 完整性校验流式解压，禁止 decode、persist、
  cache、sample 或 inspection；
- `groundtruth.txt` 只解析和保留每行第一列 timestamp，pose tokens 不得数值化；
- 固定 contiguous non-overlap 10s timestamp windows，零窗口 sequence 保留；
- 失败即 `HOLD...NO_REPLACEMENT_NO_RERUN`。

## 权限边界

B0 已具备执行授权，但仍为 `NOT_STARTED`。本交付没有下载 archive、读取 payload
或运行 Phase B metric。B0 未来即使 PASS，也只允许另行冻结 B1 metric protocol，
不自动授权 decode、pose 数值、RCLE metric、Kill Gate B、Replay、Android、人体、
安全或生产。
