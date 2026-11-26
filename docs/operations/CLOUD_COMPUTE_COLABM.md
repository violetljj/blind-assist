# 云端算力与存储运行手册（Colab 多账号池 `colabm` + Google Drive `rclone`）

本手册承接 [AGENTS.md](../../AGENTS.md) 与 [主机研究算力](../HOST_RESEARCH_COMPUTE.md) 的云端训练/评测运维细则。凭据与配置文件均保留在忽略的本机目录，严禁提交进 Git。

## 1. Colab 多账号算力池（`colabm`）与单账号 CLI

- **本机工具位置**：多付费 Colab 算力池管理工具 `colabm` 位于本机 `C:\Users\26442\.gemini\antigravity\scratch\colabm\`（入口 `colabm.cmd` / `colabm.ps1`，桌面图标 `colabm 控制台.lnk`），当前已接入多个 Pro+ 账号（如 `default` 与 `violjjet`，各自独立存放于 `data/accounts/<alias>/` 互不覆盖）。
- **常用命令**：
  - 查全池余额与全部活跃实例：`& "C:\Users\26442\.gemini\antigravity\scratch\colabm\colabm.cmd" dashboard`
  - 指定账号执行任意 `colab` 子命令：`colabm.cmd on <alias> <new|sessions|usage|upload|download|exec|stop ...>`
  - 默认单账号也可直接用全局 `colab`（与 `default` 同步）。
- **选卡与并发隔离**：
  - 启动前务必先查全池余额与各账号活跃会话，**严禁触碰其他并行会话**。
  - 预处理/拉数据用默认 CPU，日常训练优先 `L4`，大显存/重计算评估 `G4`（96 GB，Blackwell `sm_120`）或 `A100`。
- **长任务防断连 / 防闲置回收**：
  - 不要依赖单条 `colab exec` WebSocket（易断连挂起）：须以 `subprocess.Popen(..., start_new_session=True)` 脱离内核后台运行，日志与退出码分别写 `/content/<job>.log` 和 `/content/<job>.exit`（`PYTHONUNBUFFERED=1` / `flush=True`）。
  - 仅靠后台子进程不会重置 Colab 闲置回收计时器（~20 分钟可能被静默回收）：须每 5–7 分钟发一次 1 秒短 `exec` 内核保活（如 `print("PING_OK")`，加 ~25 秒超时护栏防 WebSocket 卡死）。
  - 读取日志优先用走 HTTP 的 `download`。
- **停机核销**：任务结束后务必 `stop -s <name>` 并用 `dashboard` / `sessions` / `usage` 复核已释放。

## 2. Google Drive 免交互跨账号访问（首选 `rclone`）

本项目云盘固定目录为 `BlindAssist/`（下分 `datasets/`、`caches/`、`runs/`）。

- **首选 `rclone`（0 授权弹窗、多线程提速 3–5×、跨 Colab 账号共用同一份云盘）**：
  - 本机已装 `E:\codex-tools\tools\rclone\rclone.exe`（及 `~/.local/bin/rclone.exe`），`C:\Users\26442\.config\rclone\rclone.conf` 已存有持久化授权的 `gdrive:` remote（直连 `gdrive:BlindAssist` 可用）。
  - 无论在 `default`、`violjjet` 还是任意其他 Colab 账号开卡，新开 VM 后直接 `upload` 该配置到远端 `/root/.config/rclone/rclone.conf`（严禁提交进 Git，随 VM 销毁），在 Colab 内用 `rclone copy --transfers 16` 即可免授权高速拉取同一份 `BlindAssist/datasets` 并回传 checkpoint/ONNX，彻底解耦算力账号与存储账号。
- **备选 `colab drivemount` / `gdrive` 插件**：
  - 仅在 `default` 账号确需 `/content/drive` FUSE 挂载时使用 `colab drivemount`；若需把授权 URL 转给用户，必须直接读取 CLI 原始输出原样复制（严禁手打转写），超过 2 分钟超时可 `restart-kernel` 后重挂。
  - Antigravity 本机也可用 `gdrive readonly` / `gdrive mutate` 管理（未授权不删除或公开共享）。

## 3. IO、断点续训与完整性约束

- Colab 本地盘 `/content` 断连清空；**禁止训练时直接从 Drive FUSE 随机读大量小文件**，须以单包（`.zip`/`.tar`）或整块数组（`.npy`/`.npz`）存 Drive，开机后拉入/解压到 `/content` NVMe 计算；同系列调参优先复用已暂存 NVMe 的现有 VM。
- 公开大数据集优先在默认 CPU Colab 上多并发直拉进 Drive 并生成 `manifest_sha256.json`。
- 训练脚本应支持按 Epoch 断点续训并回写 Drive；若经 FUSE 写入大文件，停机前必须执行 `drive.flush_and_unmount()`，经 `rclone` 或 FUSE 写入后均须复核远端文件列表与字节数，重传同名大文件先移走旧文件再直写新文件。
