# RGB algorithm canary R0 design tooling

状态：`DESIGN_ONLY / NO_ALGORITHM_IMPLEMENTATION / EXECUTION_NOT_AUTHORIZED`

本目录只包含 F1 设计包的 algorithm-outcome firewall 与合成恶意反例验证。它不包含 producer、正式 scientific validator、runner、cache materializer、implementation lock 或任何 algorithm output。

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/egomotion_compensated_looming/rgb_algorithm_canary_r0/validate_design_package.py `
  --contract docs/research/rcle/RCLE_PHASE_B_RGB_ALGORITHM_CANARY_R0_CONTRACT_2026-07-27.json `
  --manifest docs/research/rcle/RCLE_PHASE_B_RGB_ALGORITHM_CANARY_R0_DATA_ROLE_MANIFEST_2026-07-27.json
```

validator 只读取绑定的设计文件与 upstream geometry evidence，并对 algorithm canonical paths 做存在性检查。若疑似 algorithm claim/output/failure 路径出现，它只报告 firewall breach，不打开其内容。

正式 implementation task 仍需另建 producer、独立 validator、immutable cache materializer、progress sidecar 和 guarded launcher，并在 outcome access 前形成新的 implementation lock 与独立 implementation review。
