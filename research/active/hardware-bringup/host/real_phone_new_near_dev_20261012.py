"""Describe native near-return lineage of the 18 new phone recordings.

Observation-only CPU diagnostic: frozen decoded inputs, no model inference,
fitting, physical echo subtraction, or scene/query label assumptions.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

import real_phone_imu_replay as R
from real_contract_v5 import adapter_v5

ROOT = R.ROOT
BATCH = ROOT / "artifacts.local/work/phone-new-capture-20261012"
OUT = BATCH / "near_return"
FOCUS = ("072853", "073020", "073821")


def quantiles(x):
    return np.quantile(x, [.1, .5, .9]).tolist() if len(x) else None


def adjacent_sd(x):
    """Adjacent difference SD/sqrt(2); motion can contribute to this number."""
    return float(np.std(np.diff(x), ddof=0) / np.sqrt(2)) if len(x) > 1 else None


def describe(d, metadata):
    h = d["H"].reshape(-1, 8, 8, 16)[:, ::-1, ::-1]
    dist = d["D"].reshape(-1, 8, 8)[:, ::-1, ::-1]
    status = d["S"].reshape(-1, 8, 8)[:, ::-1, ::-1]
    z = np.asarray(d["z"])
    n = len(h)
    zone_stats = []
    for row, col in np.ndindex(8, 8):
        a, b = h[:, row, col, 0], h[:, row, col, 1]
        normalized = z[:, row, col, 1].astype(np.float64)
        valid = np.isin(status[:, row, col], (5, 6, 9, 10))
        near = valid & (dist[:, row, col] < 600)
        zone_stats.append(dict(
            row=row, col=col, raw_zone=(7-row)*8+7-col,
            bin0_mean=float(a.mean()), bin1_mean=float(b.mean()),
            bin1_frame_sd=float(b.std()), bin1_adjacent_sd=adjacent_sd(b),
            bin1_to_bin0_ratio_p10_median_p90=quantiles(b / np.maximum(a, 1e-12)),
            bin01_correlation=float(np.corrcoef(a, b)[0, 1]) if min(a.std(), b.std()) > 0 else None,
            normalized_bin1_mean=float(normalized.mean()),
            normalized_bin1_frame_sd=float(normalized.std()),
            normalized_bin1_adjacent_sd=adjacent_sd(normalized),
            normalized_bin1_p10_median_p90=quantiles(normalized),
            normalized_bin1_above3_count=int((normalized > 3).sum()),
            firmware_valid=int(valid.sum()), firmware_valid_under600mm=int(near.sum()),
            firmware_valid_at_most10mm=int((valid & (dist[:, row, col] <= 10)).sum()),
            firmware_valid_11to599mm=int((valid & (dist[:, row, col] > 10) & (dist[:, row, col] < 600)).sum()),
            firmware_distance_valid_p10_median_p90_mm=quantiles(dist[valid, row, col]),
            firmware_status_counts={str(int(s)): int((status[:, row, col] == s).sum())
                                    for s in np.unique(status[:, row, col])},
        ))
    mean_map = z[..., 1].mean(0)
    frame_peak = np.argmax(z[..., 1].reshape(n, 64), axis=1)
    peak_counts = np.bincount(frame_peak, minlength=64)
    peak_zone = int(np.argmax(mean_map))
    noise_bins = z[..., 14:16].astype(np.float64)
    valid = np.isin(status, (5, 6, 9, 10))
    near = valid & (dist < 600)
    return dict(
        **metadata, frames=n, zones=zone_stats,
        r7c7=zone_stats[63],
        top_raw_bin1=sorted(zone_stats, key=lambda v: v["bin1_mean"], reverse=True)[:5],
        top_normalized_bin1=sorted(zone_stats, key=lambda v: v["normalized_bin1_mean"], reverse=True)[:5],
        normalized_bin1_mean_map=mean_map.tolist(),
        normalized_bin1_frame_peak_zone_counts=peak_counts.reshape(8, 8).tolist(),
        normalized_bin1_mean_peak_zone=[peak_zone // 8, peak_zone % 8],
        normalized_bin1_peak_p10_median_p90=quantiles(z[..., 1].reshape(n, 64).max(1)),
        normalized_bin1_peak_zone_mode_count=int(peak_counts.max()),
        bin14_15_adjacent_sd_median_over128cells=float(np.median(np.std(np.diff(noise_bins, axis=0), axis=0) / np.sqrt(2))),
        firmware_near_frame_count=int(near.reshape(n, 64).any(1).sum()),
        firmware_near_zone_counts=near.sum(0).tolist(),
        ambient_median=float(np.median(d["A"])),
        normalized_adapter="Existing v5 signed-bias/noise normalization; model order; no per-frame DC baseline subtraction",
    )


def findings(results):
    old = results["old_reference"]
    lines = [
        "# 新手机录制：近 bin1 原始回波诊断", "",
        "**新三段低头输入没有复现旧 100754 右下角约 314 mm 的强局部回波。** 同一 r7c7 的归一化 bin1 均值从旧 136.94 降为新 2.19–2.28，三段均无固件 valid<600 mm。该现象没有稳定复现到新低头输入，支持保留按回波来源诊断，暂不建立跨场景固定扣除模板。", "",
        "18 段新录制均使用原始输入，CPU 描述；不运行模型，不估新阈值。区域为模型侧零起始 row/col，r7c7 对应 raw zone0。新段时间键为 UTC，旧 100754 为本地时间键。", "",
        "固件 valid 为 status 5/6/9/10。valid 距离是固件弱参考，不能单独定位回波来源。normalized bin1 为既有 signed-bias/noise 归一化后的输入；frame SD 与相邻差 SD 均含姿态/场景变化。", "",
        "部分新段固件 valid 距离出现 1–3 mm，另列 <=10 mm 与 11–599 mm 数量，不把 status valid 自动等同于可信几何距离。", "",
        "| 段 UTC | IMU pitch ° | r7c7 bin1/bin0 中位 | r7c7 归一化均值 / 帧 SD | r7c7 valid / 帧 | 距离中位 mm | bin1 均值最大区域 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in results["sessions"]:
        v = r["r7c7"]
        distance = v["firmware_distance_valid_p10_median_p90_mm"]
        lines.append(f"| {r['utc_key']} | {r['pitch_mean']:.2f} | {v['bin1_to_bin0_ratio_p10_median_p90'][1]:.5f} | {v['normalized_bin1_mean']:.2f} / {v['normalized_bin1_frame_sd']:.2f} | {v['firmware_valid']}/{r['frames']} | {distance[1] if distance else '无'} | {r['normalized_bin1_mean_peak_zone']} |")
    v = old["r7c7"]
    lines += ["", f"旧 100754 对照：r7c7 valid {v['firmware_valid']}/{old['frames']}，距离中位 {v['firmware_distance_valid_p10_median_p90_mm'][1]:.1f} mm；bin1/bin0 中位 {v['bin1_to_bin0_ratio_p10_median_p90'][1]:.5f}，归一化均值/帧 SD {v['normalized_bin1_mean']:.2f}/{v['normalized_bin1_frame_sd']:.2f}。", "", "## 三段低头输入", ""]
    for r in results["sessions"]:
        if r["utc_key"] not in FOCUS:
            continue
        v = r["r7c7"]
        near = v["firmware_valid_under600mm"]
        lines.append(f"- {r['utc_key']}：IMU pitch {r['pitch_mean']:.2f}°；r7c7 valid<600 mm {near}/{r['frames']}，归一化 bin1 均值 {v['normalized_bin1_mean']:.2f}，该段最强区域 {r['normalized_bin1_mean_peak_zone']}。")
    lines += ["", "## 证据范围", "",
              "18 新段 r7c7 的 bin1/bin0 中位为 0.01080–0.02240，均远低于旧段 0.75411；但 073138、073245 的 r7c7 归一化均值/帧 SD 为 10.23/29.84、36.79/94.93，仍存在随时间变化的较大回波，不能把低中位比率解释成全帧无近距信号。各段 bin1 均值最大区域也发生变化，三段低头分别落在 r4c6、r4c5、r7c3。", "",
              "可确认每段的局部幅值、空间位置、固件测距和稳定性是否复现旧 r7c7 近回波；这些量不依赖场景标签。RGB 画面未参与本脚本，安装高度和物体处于哪条身体 query 尚无独立真值。不能仅凭 r7c7 重复出现、与 bin0 相关、低头时增强或场景视觉为空，就称固定串扰、按物理回波删除、或排除真实近距表面。跨段比较也同时混有姿态、背景和温度/仪器状态变化。", "",
              "完整 64 区统计、raw zone 对应、status 分母、归一化热图和固件近距区分布均保存在 results.json；不会改写原输入。"]
    return "\n".join(lines) + "\n"


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads((BATCH / "prepared.json").read_text(encoding="utf8"))
    sessions = []
    for row in rows:
        with np.load(BATCH / "inputs" / (row["session"] + ".npz")) as d:
            metadata = {k: row[k] for k in ("session", "pitch_mean", "pitch_sd", "roll_mean", "floor_fit")}
            metadata.update(utc_key=row["session"][17:23], source=str(BATCH / "inputs" / (row["session"] + ".npz")))
            sessions.append(describe(d, metadata))
    old_folder = next(p for p in R.RAW.iterdir() if p.name[17:23] == "020754")
    rotation = np.asarray(R.load_mounting(R.MOUNTING)["tof_from_imu_rotation"])
    old = R.load(old_folder, rotation)
    with np.load(R.P.OUT / "instrument_arrays.npz") as cal:
        old["z"] = adapter_v5(old["H"], old["A"], cal["bias"].reshape(64, 16),
                              R.P.read("instrument_parameters.json")["noise"], None, True, False)
    old_report = describe(old, dict(session=old_folder.name, local_key="100754", source=str(old_folder)))
    results = dict(development=True, fitting=False, model_inference=False, sessions=sessions,
                   old_reference=old_report, command_compute_seconds=time.monotonic() - started)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    (OUT / "FINDINGS.md").write_text(findings(results), encoding="utf8")
    print(json.dumps(dict(sessions=len(sessions), seconds=results["command_compute_seconds"],
                         output=str(OUT)), ensure_ascii=False))


if __name__ == "__main__":
    main()
