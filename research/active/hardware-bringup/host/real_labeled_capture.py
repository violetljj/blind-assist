"""Guided, local-only H3 acquisition; preparation/replay needs no serial device.

Never flashes firmware, controls resets, uploads evidence, or changes a model.
Public test seams: CaptureClock, ReplaySerial, GuidedCapture, segment_plan,
check_preflight, open_serial. Frame parsing reuses the current H3-aware Decoder.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
from pathlib import Path
import statistics
import sys
import time
import urllib.request
from urllib.parse import urlsplit

from capture import Decoder, write_json

FIRMWARE = "xiao-cnh-h3-v1"  # actual boot identity of firmware/tof_cnh_h3
CONDITIONS = (
    ("open", "开阔：前方 >4.8m 无物体；记录远墙、地面、侧墙范围", None),
    ("plane064", "同一大平面：前表面水平前向距离0.64m，板面竖直", .64),
    ("plane150", "同一大平面：前表面水平前向距离1.50m，板面竖直", 1.50),
    ("head_rod", "HEAD固定短竖杆：长0.20m、直径0.02–0.04m，x=0，中心高H−0.10m", 1.50),
    ("body_rod", "BODY固定短竖杆：长0.20m、直径0.02–0.04m，x=0，中心高H−0.65m", 1.50),
    ("outside_rod", "走廊外侧同HEAD短杆：x=+0.50m，中心高H−0.10m", 1.50),
)


class CaptureClock:
    monotonic = staticmethod(time.monotonic)
    monotonic_ns = staticmethod(time.monotonic_ns)
    sleep = staticmethod(time.sleep)
    utcnow = staticmethod(lambda: datetime.now(timezone.utc).isoformat())


class ReplaySerial:
    """Recorded newline stream with an injected clock; not a hardware connection.

    ``clock.sleep`` may advance a fake clock, making 30s virtual captures instant.
    The file is replayed once; EOF is silence. Identity is not fabricated.
    """
    def __init__(self, path, clock=None, frame_interval=.196):
        self.clock = clock or CaptureClock()
        self.handle = Path(path).open("rb")
        self.frame_interval = frame_interval
        self.timeout = .2
        self.closed = False

    def read(self, size=16384):
        line = self.handle.readline()
        if b'"cnh_frame"' in line:
            self.clock.sleep(self.frame_interval)
        elif not line:
            self.clock.sleep(self.timeout)
        return line

    def reset_input_buffer(self):
        pass  # replay has no buffered old frames

    def close(self):
        self.handle.close()
        self.closed = True


def open_serial(port="COM5", baud=115200):
    """Open only the requested port, without DTR/RTS reset or command writes."""
    if not port or port.lower() == "auto":
        raise ValueError("需要显式串口名；不自动扫描串口")
    import serial
    device = serial.Serial(port=None, baudrate=baud, timeout=.2)
    device.port = port
    device.dtr = False
    device.rts = False
    try:
        device.open()
    except BaseException:
        device.close()
        raise
    return device


def segment_plan():
    return [{"segment_id": f"{condition}-{ambient}", "condition": condition,
             "instructions": instructions, "nominal_distance_m": distance,
             "ambient_level": ambient}
            for condition, instructions, distance in CONDITIONS
            for ambient in ("dark", "bright")]


def check_preflight(events, summary, minimum_frames=20):
    """Read-only identity/config/layout/rate gate, with explicit rejection reasons."""
    sensors = [e.get("sensor", e) for e in events]
    identities = [e.get("firmware") for e in sensors if e.get("type") == "boot"]
    configs = [e for e in sensors if e.get("type") == "config"]
    reasons = []
    if not identities or identities[-1] != FIRMWARE:
        reasons.append(f"未收到H3 boot身份 {FIRMWARE}；不能仅由帧尺寸推断固件")
    if not configs or any(configs[-1].get(k) != v for k, v in
                          (("zones", 64), ("bins", 16), ("start_bin", 0), ("sub_sample", 8))):
        reasons.append("未收到正确config：zones64/bins16/start_bin0/sub_sample8")
    if summary.get("frames", 0) < minimum_frames:
        reasons.append(f"有效帧不足{minimum_frames}")
    if summary.get("cnh_layout_counts") != {"8x8x16": summary.get("frames", 0)}:
        reasons.append("并非全部8×8×16 CNH帧")
    hz = summary.get("observed_sensor_hz")
    if hz is None or not 4.5 <= hz <= 5.7:
        reasons.append(f"帧率 {hz} 不在4.5–5.7Hz（名义约5.1Hz）")
    missing = summary.get("missing_sequences", 0)
    loss = missing / max(1, summary.get("frames", 0) + missing)
    if loss > .02:
        reasons.append(f"序号丢帧率{loss:.1%}>2%")
    if summary.get("device_errors", 0) or summary.get("sequence_resets_or_reorders", 0) or summary.get("duplicate_sequences", 0):
        reasons.append("自检期间固件报错、序号重置或重复")
    if summary.get("issues", {}).get("invalid_frame", 0):
        reasons.append("自检期间出现无效帧")
    return {"passed": not reasons, "reasons": reasons, "firmware_identities": identities,
            "config": configs[-1] if configs else None, "drop_fraction": loss, "summary": summary}


def local_camera_url(url):
    p = urlsplit(url)
    address = ipaddress.ip_address(p.hostname or "")  # literal only, no DNS
    allowed = address.is_loopback or (address.version == 4 and any(
        address in ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")))
    if (not allowed or p.scheme != "http" or p.path != "/stream" or p.query or p.fragment
            or p.username or p.password or p.port not in (None, 80, 81)):
        raise ValueError("Atom仅支持显式本机/私有IP http://IP:81/stream，不扫描、不上传")
    return url


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("camera redirect rejected")


def save_optional_jpeg(url, path):
    """One bounded local MJPEG request; any camera failure is nonblocking evidence."""
    receipt = {"path": str(path), "url": url, "status": "unavailable", "synchronized": False}
    try:
        local_camera_url(url)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        deadline = time.monotonic() + 2
        with opener.open(url, timeout=2) as response:
            data = bytearray()
            while time.monotonic() < deadline and len(data) < 4 * 1024 * 1024:
                # read1 avoids waiting to fill 64KB after the first complete JPEG.
                chunk = response.read1(8192)
                if not chunk:
                    break
                data.extend(chunk)
                start = data.find(b"\xff\xd8")
                end = data.find(b"\xff\xd9", max(start, 0))
                if start >= 0 and end > start:
                    jpeg = bytes(data[start:end + 2])
                    Path(path).write_bytes(jpeg)
                    receipt.update(status="saved", bytes=len(jpeg), sha256=hashlib.sha256(jpeg).hexdigest())
                    break
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    return receipt


class GuidedCapture:
    """Inject ``serial_factory()``, clock and console functions for offline testing."""
    def __init__(self, output_root, serial_factory, *, clock=None, input_fn=input,
                 print_fn=print, preflight_seconds=12, segment_seconds=30,
                 countdown_seconds=5, camera_url=None, port="COM5", replay=False):
        self.root = Path(output_root)
        self.serial_factory = serial_factory
        self.clock = clock or CaptureClock()
        self.ask, self.print = input_fn, print_fn
        self.preflight_seconds, self.segment_seconds = preflight_seconds, segment_seconds
        self.countdown_seconds = countdown_seconds
        self.camera_url = local_camera_url(camera_url) if camera_url else None
        self.port, self.replay = port, replay
        self.device = None

    def number(self, prompt, default=None, low=None, high=None, optional=False):
        while True:
            raw = self.ask(prompt + (f" [建议{default}，须输入实测值]" if default is not None else "") + ": ").strip()
            if not raw and optional:
                return None
            if not raw:
                self.print("建议值不是测量；请测量后明确输入数值。")
                continue
            try:
                value = float(raw)
                if not __import__("math").isfinite(value) or (low is not None and value < low) or (high is not None and value > high):
                    raise ValueError()
                return value
            except (ValueError, TypeError):
                self.print("请输入范围内的实测数值（米/度）；不能填估计值冒充测量。")

    def metadata(self, segment, session_metadata):
        condition = segment["condition"]
        self.print(f"\n{segment['segment_id']}：{segment['instructions']}；ambient={segment['ambient_level']}")
        h = self.number("本段重新实测传感器离地高度m（移动设备后必须更新）", session_metadata["sensor_height_m"], .3, 2.5)
        self.print(f"实测安装高H={h:.3f}m；HEAD高{h-.42:.3f}–{h+.20:.3f}m；BODY高{h-.90:.3f}–{h-.42:.3f}m。")
        self.print("暗档关灯/拉帘，亮档开灯/开帘；避免太阳直照。先拍手机场景照，拍完离开视场。")
        distance = self.number("传感器到目标前表面的水平前向z距离m（开阔填已测距离；未知留空并在备注说明下界依据）",
                               segment["nominal_distance_m"], .05, 100, optional=condition == "open")
        target_default = h-.65 if condition == "body_rod" else h-.10 if "rod" in condition else h-.30
        target_height = None if condition == "open" else self.number("目标中心离地高度m（杆中心/板中心）", target_default, 0, 10)
        x = self.number("目标中心横向x偏移m（面向前方，右为正）", .5 if condition == "outside_rod" else 0, -10, 10)
        metadata = {
            **segment, **session_metadata, "sensor_height_m": h, "target_distance_m": distance,
            "target_height_m": target_height, "target_x_m": x,
            "tilt_deg": self.number("手机水平仪实测向下俯角degree（下俯为正）", session_metadata["tilt_deg"], -45, 45),
            "ambient_level": segment["ambient_level"],
            "ambient_method": self.ask("实际ambient操作（开关灯/窗帘状态，不把档位当照度测量）: ").strip(),
            "illuminance_lux": self.number("可选照度lux（无仪器留空）", low=0, high=200000, optional=True),
            "distance_uncertainty_m": self.number("距离/高度测量误差上界m", .02, .001, .5),
            "tilt_uncertainty_deg": self.number("俯角读数误差上界degree", 2, .1, 15),
            "photo_before": self.ask("手机前照路径/文件名（可后补；每段必须拍摄）: ").strip() or None,
            "measurement_source": "operator tape measure + phone inclinometer; firmware distance is weak reference",
            "static_scene": True, "geometry_coordinate": "sensor-origin x right, y down, z horizontal forward",
        }
        if "rod" in condition:
            metadata.update(rod_diameter_m=self.number("实测杆直径m", .03, .005, .2),
                            rod_length_m=self.number("实测杆长度m", .20, .01, 2),
                            rod_material=self.ask("杆材质/颜色: ").strip(),
                            support_description=self.ask("支架位置/外形（须在query外；如实记录）: ").strip())
        if condition.startswith("plane"):
            metadata.update(plane_width_m=self.number("实测板宽m", low=.1, high=10),
                            plane_height_m=self.number("实测板高m", low=.1, high=10),
                            plane_material=self.ask("同一板面的材质/颜色: ").strip())
        metadata["scene_notes"] = self.ask("地面/侧墙/其他物体距离与遮挡、照片标尺等备注: ").strip()
        metadata["operator_confirmed"] = self.ask("确认设备/物体固定，人离开视场，已拍前照；输入 y 开始: ").strip().lower() == "y"
        return metadata

    def collect(self, directory, duration, metadata, *, show=True):
        """Exclusive attempt dir; every received byte and frame survives interruption."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        metadata = dict(metadata, started_utc=self.clock.utcnow(), status="recording",
                        evidence_scope="SIMULATED_SERIAL_REPLAY" if self.replay else "REAL_LABELED_CAPTURE",
                        host_timestamp_rule="chunk-final-newline receipt; host monotonic_ns + UTC; not synchronized")
        write_json(directory / "metadata.json", metadata)
        events, latest = [], None
        preflight_decoder, preflight_events = None, []
        digest, total, monitor = hashlib.sha256(), 0, self.clock.monotonic()
        started = self.clock.monotonic()
        error = None
        completed = False
        camera_receipts, futures = [], []
        pool = ThreadPoolExecutor(max_workers=2) if self.camera_url else None
        if pool:
            futures.append(pool.submit(save_optional_jpeg, self.camera_url, directory / "atom-first.jpg"))
        with (directory / "raw.bin").open("xb") as raw, \
             (directory / "frames.jsonl").open("x", encoding="utf-8") as frames, \
             (directory / "events.jsonl").open("x", encoding="utf-8") as event_file, \
             (directory / "issues.jsonl").open("x", encoding="utf-8") as issues:
            handles = {"frames": frames, "events": event_file, "issues": issues}
            def emit(kind, record):
                nonlocal latest, preflight_decoder, preflight_events
                record = dict(record, host_received_utc=self.clock.utcnow())
                handles[kind].write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                if kind == "events":
                    events.append(record)
                    if metadata.get("phase") == "preflight":
                        if record["sensor"].get("type") == "boot":
                            preflight_decoder = Decoder()
                            preflight_events = [record]
                        elif preflight_decoder is not None:
                            preflight_events.append(record)
                            preflight_decoder.feed((json.dumps(record["sensor"])+"\n").encode(), record["host_received_monotonic_ns"])
                elif kind == "frames":
                    latest = record
                    if preflight_decoder is not None:
                        preflight_decoder.feed((json.dumps(record["sensor"])+"\n").encode(), record["host_received_monotonic_ns"])
                elif kind == "issues" and preflight_decoder is not None and record["reason"] in ("invalid_frame", "malformed_line"):
                    preflight_decoder.issues[record["reason"]] += 1
            decoder = Decoder(emit)
            try:
                while self.clock.monotonic() - started < duration:
                    self.device.timeout = min(.2, max(.001, duration-(self.clock.monotonic()-started)))
                    try:
                        chunk = self.device.read(16384)
                    except OSError:
                        if metadata.get("phase") != "preflight" or self.replay:
                            raise
                        # Manual RESET may disconnect USB. Reopen this exact port only;
                        # never toggle DTR/RTS, reset the device, or scan other ports.
                        self.device.close()
                        self.clock.sleep(.2)
                        try:
                            self.device = self.serial_factory()
                            decoder.issue("manual_reset_serial_reconnect", self.port, self.clock.monotonic_ns())
                        except OSError:
                            pass
                        continue
                    host_ns = self.clock.monotonic_ns()
                    raw.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
                    decoder.feed(chunk, host_ns)
                    if show and self.clock.monotonic() - monitor >= 2:
                        self.monitor(decoder, latest)
                        monitor = self.clock.monotonic()
                completed = True
            except BaseException as exc:
                error = exc
            finally:
                # Preserve trailing incomplete bytes as an issue, without inventing frames.
                if decoder.pending:
                    decoder.issue("incomplete_final_line", f"{len(decoder.pending)} bytes retained in raw.bin", self.clock.monotonic_ns())
                for handle in (raw, frames, event_file, issues):
                    handle.flush()
                if pool:
                    futures.append(pool.submit(save_optional_jpeg, self.camera_url, directory / "atom-last.jpg"))
                    pool.shutdown(wait=True)
                    camera_receipts = [f.result() for f in futures]
                summary = decoder.summary()
                metadata.update(status="complete" if completed else "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                ended_utc=self.clock.utcnow(), host_elapsed_seconds=self.clock.monotonic()-started,
                                raw_bytes=total, raw_sha256=digest.hexdigest(), summary=summary,
                                camera=camera_receipts, error=f"{type(error).__name__}: {error}" if error else None)
                if metadata.get("phase") == "preflight":
                    metadata["startup_window_summary"] = preflight_decoder.summary() if preflight_decoder is not None else Decoder().summary()
                write_json(directory / "metadata.json", metadata)
        if error:
            raise error
        return {"metadata": metadata, "summary": summary, "events": events, "latest": latest,
                "preflight_summary": metadata.get("startup_window_summary"), "preflight_events": preflight_events}

    def monitor(self, decoder, latest):
        summary = decoder.summary()
        missing = summary["missing_sequences"]
        hz = summary["observed_sensor_hz"]
        self.print(f"帧{decoder.frames}，rate={hz if hz is not None else '?'}Hz，丢序号={missing} / {decoder.frames+missing}")
        if latest:
            distances = latest["derived"]["distance_known_mm"]
            ambient = latest["derived"].get("ambient_normalized", [])
            self.print("逐zone固件距离mm（?=UNKNOWN）：")
            for row in range(8):
                self.print(" ".join(f"{v:5d}" if v is not None else "    ?" for v in distances[row*8:(row+1)*8]))
            self.print("逐zone CNH ambient原单位缩略：")
            for row in range(8):
                self.print(" ".join(f"{v:6.1f}" for v in ambient[row*8:(row+1)*8]))
        if missing / max(1, decoder.frames+missing) > .02:
            self.print("警告：丢帧>2%，完成后建议重录；失败证据仍保留。")

    def quality_warnings(self, result, metadata):
        summary = result["summary"]
        warnings = []
        missing = summary["missing_sequences"]
        if missing / max(1, summary["frames"]+missing) > .02:
            warnings.append("序号丢帧>2%")
        hz = summary["observed_sensor_hz"]
        if hz is None or not 4.5 <= hz <= 5.7:
            warnings.append("帧率不在4.5–5.7Hz")
        if summary["frames"] < self.segment_seconds * 4.5 * .95:
            warnings.append("帧数不足，疑似主机停顿/串口缓存")
        if summary["device_errors"] or summary["sequence_resets_or_reorders"] or summary["duplicate_sequences"]:
            warnings.append("固件错误/序号异常")
        if summary["issues"].get("invalid_frame", 0):
            warnings.append("包含无效帧")
        if summary.get("cnh_layout_counts") != {"8x8x16": summary.get("frames", 0)}:
            warnings.append("录制期间出现非8×8×16布局")
        latest = result["latest"]
        if latest and metadata["condition"].startswith("plane"):
            known = latest["derived"]["distance_known_mm"]
            center = [known[r*8+c]/1000 for r in (3, 4) for c in (3, 4) if known[r*8+c] is not None]
            # Firmware distance is slant range; allow geometry/zone angle margin.
            if not center or abs(statistics.median(center)-metadata["target_distance_m"]) > max(.20, metadata["target_distance_m"]*.25):
                warnings.append("板面中心固件距离与实测z明显偏离/UNKNOWN（弱参考，非真值）")
        elif latest and "rod" in metadata["condition"]:
            known = [v/1000 for v in latest["derived"]["distance_known_mm"] if v is not None]
            expected = (metadata["target_distance_m"]**2 + metadata["target_x_m"]**2
                        + (metadata["sensor_height_m"]-metadata["target_height_m"])**2)**.5
            if not any(abs(v-expected) < max(.25, expected*.20) for v in known):
                warnings.append("未见接近杆斜距的有效固件距离；细杆可能漏测，仅提示检查照片与摆放")
        return warnings

    def run_session(self, session_number, *, resume=False, redo=None):
        directory = self.root / f"session-{session_number:02d}"
        manifest_path = directory / "session.json"
        redo_ids = set(redo or [])
        valid_ids = {s["segment_id"] for s in segment_plan()}
        if redo_ids-valid_ids:
            raise ValueError(f"未知segment id: {sorted(redo_ids-valid_ids)}")
        if manifest_path.exists():
            if not (resume or redo_ids):
                raise ValueError("会话已存在；用--resume/--redo，禁止覆盖")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            directory.mkdir(parents=True, exist_ok=False)
            self.print("本工具不刷写。请固定安装；启动身份只能在boot/config出现时读取。")
            self.print("准备：卷尺、手机水平仪/相机、固定支架、同一哑光大平面约1.5m宽×1.4m高、长20cm/直径2–4cm短杆。")
            self.print("名义安装离地H=1.65m、向下俯角10°；优先H=1.60–1.70m，可用1.20–1.70m并逐段记录实际值。")
            self.print("先开阔（前方>4.8m无物体）与两距离平面，后细杆；支架放走廊外，人录制时离开视场。")
            self.print("每条件暗/亮两档，各30s并拍手机前后照；重复两次独立开始，中间手动重启或重新安装。")
            h = self.number("实测传感器离地高度m（名义1.65；可接受1.20–1.70）", 1.65, .3, 2.5)
            tilt = self.number("实测向下俯角degree（名义10；下俯为正）", 10, -45, 45)
            manifest = {"schema": "hardware-bringup.real-labeled-capture.v1", "session_number": session_number,
                        "port": self.port, "created_utc": self.clock.utcnow(), "attempts": [],
                        "sensor_height_m": h, "tilt_deg": tilt,
                        "installation_notes": self.ask("安装/设备朝向与本次独立开始（重启或重新安装）的说明: ").strip(),
                        "status": "preflight"}
            write_json(manifest_path, manifest)
        self.device = self.serial_factory()
        try:
            n = len(list(directory.glob("preflight-*"))) + 1
            if not self.replay:
                self.print("COM监听已打开，DTR/RTS均False；现在手动短按板上RESET，然后立即按Enter。")
                self.print("脚本不发RESET/CONFIG、不刷写；如USB短暂断开，只重连指定串口。启动消息约2s后出现。")
                self.ask("开始读取boot/config与至少20帧自检（按Enter）: ")
            result = self.collect(directory / f"preflight-{n:02d}", self.preflight_seconds, {"phase": "preflight"}, show=False)
            gate = check_preflight(result["preflight_events"], result["preflight_summary"])
            manifest["preflight"] = gate
            write_json(manifest_path, manifest)
            if not gate["passed"]:
                manifest["status"] = "preflight_failed"
                write_json(manifest_path, manifest)
                raise ValueError("自检未通过：" + "；".join(gate["reasons"]) +
                                 "。不刷写；重新运行--resume，先保持监听再手动按RESET读取启动消息。")
            self.print(f"自检通过：{FIRMWARE}、8×8×16、{gate['summary']['observed_sensor_hz']:.2f}Hz；loss={gate['drop_fraction']:.1%}")
            manifest["status"] = "recording"
            write_json(manifest_path, manifest)
            session_meta = {k: manifest[k] for k in ("session_number", "sensor_height_m", "tilt_deg", "installation_notes")}
            session_meta.update(startup_record=f"preflight-{n:02d}", firmware_identity=FIRMWARE,
                                resumed_session=bool(resume or redo_ids))
            for segment in segment_plan():
                sid = segment["segment_id"]
                attempts = [a for a in manifest["attempts"] if a["segment_id"] == sid]
                if redo_ids and sid not in redo_ids:
                    continue
                if sid not in redo_ids and any(a["status"] == "complete" for a in attempts):
                    continue
                while True:
                    metadata = self.metadata(segment, session_meta)
                    if not metadata["operator_confirmed"]:
                        manifest["status"] = "paused"
                        write_json(manifest_path, manifest)
                        return manifest
                    number = len([a for a in manifest["attempts"] if a["segment_id"] == sid]) + 1
                    rel = f"{sid}/attempt-{number:02d}"
                    record = {"segment_id": sid, "attempt": number, "path": rel, "status": "recording"}
                    manifest["attempts"].append(record)
                    write_json(manifest_path, manifest)
                    try:
                        # Drain setup backlog; capture starts with fresh bytes.
                        for remaining in range(self.countdown_seconds, 0, -1):
                            self.print(f"{remaining}s：离开视场，勿移动设备/目标")
                            if hasattr(self.device, "reset_input_buffer"):
                                self.device.reset_input_buffer()
                            self.clock.sleep(1)
                        if hasattr(self.device, "reset_input_buffer"):
                            self.device.reset_input_buffer()
                        result = self.collect(directory / rel, self.segment_seconds, metadata)
                        warnings = self.quality_warnings(result, metadata)
                        after = self.ask("录制完成；请拍手机后照，输入路径/文件名（可后补）: ").strip() or None
                        result["metadata"].update(photo_after=after, quality_warnings=warnings,
                                                  status="needs_review" if warnings else "complete")
                        write_json(directory / rel / "metadata.json", result["metadata"])
                        record.update(status=result["metadata"]["status"], frames=result["summary"]["frames"], warnings=warnings)
                        # A replacement supersedes only previous usable/review attempts; failed evidence keeps its status.
                        for previous in manifest["attempts"]:
                            if previous is not record and previous["segment_id"] == sid and previous["status"] in ("complete", "needs_review"):
                                previous["original_status"] = previous["status"]
                                previous["status"] = "superseded"
                                previous["superseded_by"] = rel
                                old_path = directory / previous["path"] / "metadata.json"
                                old = json.loads(old_path.read_text(encoding="utf-8"))
                                old.update(original_status=old["status"], status="superseded", superseded_by=rel)
                                write_json(old_path, old)
                        write_json(manifest_path, manifest)
                        if warnings:
                            self.print("检查/重录建议：" + "；".join(warnings))
                        answer = self.ask("重录本段? y=保留旧段并重录；Enter=下一段；q=暂停: ").strip().lower()
                        if answer == "q":
                            manifest["status"] = "paused"
                            write_json(manifest_path, manifest)
                            return manifest
                        if answer != "y":
                            break
                    except BaseException:
                        path = directory / rel / "metadata.json"
                        record["status"] = json.loads(path.read_text(encoding="utf-8"))["status"] if path.exists() else "interrupted_before_recording"
                        manifest["status"] = "interrupted"
                        write_json(manifest_path, manifest)
                        raise
            manifest["status"] = "complete" if all(any(a["segment_id"] == sid and a["status"] in ("complete", "needs_review") for a in manifest["attempts"]) for sid in valid_ids) else "partial"
            write_json(manifest_path, manifest)
            return manifest
        finally:
            self.device.close()
            self.device = None
            self.write_inventory(directory)

    def write_inventory(self, directory):
        path = directory / "session.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        lines = ["segment_id\tattempt\tstatus\tframes\tpath"]
        for a in manifest["attempts"]:
            lines.append(f"{a['segment_id']}\t{a['attempt']}\t{a['status']}\t{a.get('frames', '')}\t{a['path']}")
        (directory / "inventory.tsv").write_text("\n".join(lines)+"\n", encoding="utf-8")
        self.print(f"会话{manifest['session_number']}：{manifest['status']}；清单 {directory / 'inventory.tsv'}")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="中文引导H3采集；只读串口、不刷写；默认两次独立会话")
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--session", type=int, default=1, choices=(1, 2))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--redo", action="append", default=[], metavar="SEGMENT_ID")
    parser.add_argument("--one-session", action="store_true", help="只运行指定会话")
    parser.add_argument("--camera-url", help="可选Atom私有/本机IP http://IP:81/stream")
    parser.add_argument("--replay", type=Path, help="模拟串口读取已有raw.bin；标记REPLAY，不连接设备")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[4]
    root = args.output or repo / "artifacts.local" / "work" / f"real-labeled-capture-{datetime.now().strftime('%Y%m%d')}"
    canonical = (repo / "artifacts.local").resolve()
    if not root.resolve().is_relative_to(canonical):
        parser.error("输出必须位于canonical artifacts.local内")
    clock = CaptureClock()
    factory = (lambda: ReplaySerial(args.replay, clock)) if args.replay else (lambda: open_serial(args.port))
    guided = GuidedCapture(root, factory, clock=clock, port=args.port,
                           replay=bool(args.replay), camera_url=args.camera_url)
    try:
        for number in range(args.session, args.session+1 if args.one_session or args.redo else 3):
            if number > args.session:
                answer = input("会话一结束，串口已关闭。请拔插重启（或重新安装），记录实际变化；准备好立即输入 y 开始会话二: ").strip().lower()
                if answer != "y":
                    return 0
            manifest = guided.run_session(number, resume=args.resume, redo=args.redo)
            if manifest["status"] not in ("complete", "partial"):
                return 0
        return 0
    except KeyboardInterrupt:
        print("已中断，证据保留；下一次同路径--resume（必要时--session 2）。")
        return 130
    except (ValueError, OSError) as exc:
        print(f"停止：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
