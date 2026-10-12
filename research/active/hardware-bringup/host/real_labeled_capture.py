"""Simple dorm H3 capture: USB or explicit private-LAN full-CNH TCP, no flashing.

Public offline seams: CaptureClock, ReplaySerial, GuidedCapture, segment_plan,
check_preflight, summarize_distance, default_output_root. Distances are firmware
weak references; installation height is an estimate. Neither supplies query truth.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import statistics
import socket
import sys
import time
import uuid
from capture import Decoder, write_json
from atom_imu import ImuFeed, load_mounting, private_host, unavailable_alignment, receipt_ns, clock_metadata

FIRMWARE = "xiao-cnh-h3-v1"
WIFI_FIRMWARE = "xiao-cnh-h3-wifi-v1"
FIRMWARE_IDENTITIES = (FIRMWARE, WIFI_FIRMWARE)
PROTOCOL_VERSION = "simple-dorm-v2"
CONDITIONS = (
    (1, "near_no_echo", "测距头视线越过栏杆朝外，人和支架留栏杆内；不隔玻璃，避开衣物。", None),
    (2, "wall_near", "正对宿舍左白墙，让ToF读数大约0.6m。", .6),
    (3, "wall_far", "同一左白墙，整套支架移远，让ToF读数大约1.5m。", 1.5),
    (4, "empty_corridor", "明档：门口端沿过道中线朝尽头深色窗帘；清空中线地面杂物，保留左墙/右床梯背景。", None),
    ("4b", "empty_corridor_dark", "支架和场景保持第4段原位，关灯、合上遮光帘，再录同一空过道。", None),
    (5, "rod_near", "先恢复明档（开灯、打开遮光帘）；拖把/扫把杆落地竖在过道中线，大约1m。", 1.),
    (6, "rod_far", "同一杆落地竖在过道中线，大约2m。", 2.),
    (7, "outside_rod", "竖杆大约1m，向右横移一大步约0.5m。", 1.),
    (8, "head_crossbar", "杆横绑床梯，伸进过道，约额头高度、大约1–1.5m。", 1.25),
    (9, "empty_corridor_reboot", "恢复第4段明档，独立拔插重启后，重录第4段的空过道。", None),
)

class CaptureClock:
    monotonic = staticmethod(time.perf_counter)
    monotonic_ns = staticmethod(receipt_ns)
    receipt_clock = staticmethod(clock_metadata)
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


def private_tof_host(host):
    """Explicit RFC1918 IPv4 only; never discover or scan other devices."""
    try:
        address = ipaddress.IPv4Address(str(host).strip())
    except ipaddress.AddressValueError as exc:
        raise ValueError("ToF Wi-Fi需填写明确的局域网IPv4地址") from exc
    if not any(address in ipaddress.IPv4Network(net) for net in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")):
        raise ValueError("ToF Wi-Fi地址必须为10/172.16–31/192.168私有网段")
    return str(address)


class WiFiStream:
    """Full H3 newline JSON over TCP; identical Decoder, no board commands."""
    def __init__(self, host, port=3335, *, socket_factory=socket.create_connection):
        self.host = private_tof_host(host)
        self.port = int(port)
        if not 1 <= self.port <= 65535:
            raise ValueError("ToF TCP端口必须在1–65535")
        self.socket = socket_factory((self.host, self.port), timeout=3)
        self.socket_factory = socket_factory
        self.reconnections = 0
        self.socket.settimeout(.2)
        self.closed = False

    @property
    def timeout(self):
        return self.socket.gettimeout()

    @timeout.setter
    def timeout(self, value):
        self.socket.settimeout(value)

    def read(self, size=16384):
        try:
            chunk = self.socket.recv(size)
        except socket.timeout:
            return b""
        if not chunk:
            self.reconnect()
            return b""
        return chunk

    def reconnect(self):
        self.socket.close()
        self.socket = self.socket_factory((self.host, self.port), timeout=3)
        self.socket.settimeout(.2)
        self.reconnections += 1

    def reset_input_buffer(self):
        # Discard only bytes already pending on the host socket. No command/reset
        # is sent to the board; boot/config are retained by preflight (no drain).
        prior = self.timeout
        self.socket.setblocking(False)
        try:
            for _ in range(64):
                try:
                    if not self.socket.recv(16384):
                        self.reconnect()
                        break
                except BlockingIOError:
                    break
        finally:
            self.timeout = prior

    def close(self):
        if not self.closed:
            self.closed = True
            self.socket.close()


def segment_plan():
    """Single ten-segment session; 4b is a string index, not a numeric pseudo-index."""
    return [{"segment_id": f"{i:02d}-{condition}" if isinstance(i, int) else f"{i}-{condition}", "condition": condition,
             "condition_index": i, "instructions": text,
             "nominal_distance_m": distance, "lighting": "room",
             "ambient_level": "low" if i == "4b" else "high",
             "ambient_pair_id": "empty-corridor-fixed-setup" if i in (4, "4b") else None}
            for i, condition, text, distance in CONDITIONS]


def summarize_distance(directory):
    """Aggregate the complete probe, not its latest frame; center status5 only."""
    medians, values, total = [], [], 0
    zones = (27, 28, 35, 36)
    for line in (Path(directory)/"frames.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        sensor = record["sensor"]
        total += 1
        if (sensor.get("rows"), sensor.get("cols"), sensor.get("bins")) != (8, 8, 16):
            continue
        selected = [sensor["distance_mm"][z]/1000 for z in zones
                    if sensor["distance_mm"][z] > 0 and sensor["target_status"][z] == 5
                    and sensor["nb_target"][z] > 0]
        if selected:
            medians.append(statistics.median(selected))
            values.extend(selected)
    median = statistics.median(medians) if medians else None
    ordered = sorted(medians)
    spread = (ordered[int(.9*(len(ordered)-1))]-ordered[int(.1*(len(ordered)-1))]) if ordered else None
    fraction = len(medians)/max(1, total)
    stable = len(medians) >= 3 and fraction >= .5 and spread <= max(.10, median*.10)
    return {"distance_m": median if stable else None, "observed_median_m": median,
            "stable": stable, "quality": "stable_weak_reference" if stable else "unstable_or_unknown",
            "frames_seen": total, "frames_with_center": len(medians), "valid_center_values": len(values),
            "valid_frame_fraction": fraction, "p90_minus_p10_m": spread, "center_zones": list(zones),
            "sensor_source": "distance_mm where target_status==5 AND nb_target>0 AND distance>0",
            "aggregation": "median of each frame's valid central4 distances, then median over full probe",
            "geometry_limit": "firmware slant range, weak reference only; not measured z or query truth"}


def default_output_root(work, today=None):
    work = Path(work)
    return work / f"real-labeled-capture-{today or datetime.now().strftime('%Y%m%d')}"


def check_preflight(events, summary, minimum_frames=20):
    """Read-only identity/config/layout/rate gate, with explicit rejection reasons."""
    sensors = [e.get("sensor", e) for e in events]
    boots = [e for e in sensors if e.get("type") == "boot"]
    identities = [e.get("firmware") for e in boots]
    configs = [e for e in sensors if e.get("type") == "config"]
    reasons = []
    if not identities or identities[-1] not in FIRMWARE_IDENTITIES:
        reasons.append(f"未收到H3 boot身份 {' / '.join(FIRMWARE_IDENTITIES)}；不能仅由帧尺寸推断固件")
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
            "boot_id": boots[-1].get("boot_id") if boots else None,
            "config": configs[-1] if configs else None, "drop_fraction": loss, "summary": summary}


class GuidedCapture:
    """One estimated height; Enter-led segments; inject IO/clock for offline replay."""
    def __init__(self, output_root, serial_factory, *, clock=None, input_fn=input,
                 print_fn=print, preflight_seconds=12, segment_seconds=30,
                 countdown_seconds=3, probe_seconds=2, port="COM5", replay=False, transport=None,
                 imu_feed=None):
        self.root = Path(output_root)
        self.serial_factory, self.clock = serial_factory, clock or CaptureClock()
        self.ask, self.print = input_fn, print_fn
        self.preflight_seconds, self.segment_seconds = preflight_seconds, segment_seconds
        self.countdown_seconds, self.probe_seconds = countdown_seconds, probe_seconds
        self.port, self.replay, self.device = port, replay, None
        self.transport = dict(transport or {"kind": "recorded_serial_replay" if replay else "usb_serial", "port": port})
        self.startup_path = None
        self.imu_feed = imu_feed

    def drain(self):
        if hasattr(self.device, "reset_input_buffer"):
            self.device.reset_input_buffer()

    def monitor(self, decoder, latest):
        summary = decoder.summary()
        missing = summary["missing_sequences"]
        distances = latest["derived"]["distance_known_mm"] if latest else []
        center = [distances[z] for z in (27, 28, 35, 36) if len(distances) == 64 and distances[z] is not None]
        ambient = latest["derived"].get("ambient_normalized", []) if latest else []
        self.print(f"帧{decoder.frames}，{summary['observed_sensor_hz'] or 0:.2f}Hz，丢帧{missing}；"
                   f"中央距离{statistics.median(center) if center else '?'}mm，ambient中位{statistics.median(ambient) if ambient else '?'}")

    def preflight(self, reason="initial"):
        self.device = self.serial_factory()
        n = len(list(self.root.glob("startup-*")))+1
        directory = self.root/f"startup-{n:02d}"
        if self.transport["kind"] == "wifi_tcp":
            self.print("ToF Wi-Fi已连接；等待本次TCP连接真实发送的H3 boot/config和帧，不需要按USB RESET。")
        elif not self.replay:
            self.print("COM监听已打开；现在手动短按RESET后马上按Enter（不刷写，脚本不发控制命令）。")
            self.ask("读取新启动boot/config与H3自检，按Enter: ")
        result = self.collect(directory, self.preflight_seconds,
                              {"phase": "preflight", "reason": reason, "protocol_version": PROTOCOL_VERSION}, show=False)
        gate = check_preflight(result["preflight_events"], result["preflight_summary"])
        gate["path"] = directory.name
        gate["transport"] = dict(self.transport)
        gate["boot_record_source"] = "tcp_cached_actual_hardware_boot" if self.transport["kind"] == "wifi_tcp" else "recorded_serial_replay" if self.replay else "serial_observed_boot"
        gate["transport_connection_events"] = [e["sensor"] for e in result["events"] if e["sensor"].get("type") == "transport_connection"]
        if self.transport["kind"] == "wifi_tcp" and not (isinstance(gate["boot_id"], str) and gate["boot_id"].strip()):
            gate["reasons"].append("Wi-Fi H3缺少真实hardware boot_id，无法区分连接与物理重启")
            gate["passed"] = False
        write_json(directory/"gate.json", gate)
        if not gate["passed"]:
            retry = "检查Wi-Fi固件/地址，--resume重新连接重试" if self.transport["kind"] == "wifi_tcp" else "保持监听手动RESET，--resume重试"
            raise ValueError("自检未通过："+"；".join(gate["reasons"])+"；"+retry)
        self.startup_path = directory.name
        self.print(f"自检通过：H3 8×8×16 {gate['summary']['observed_sensor_hz']:.2f}Hz，丢帧{gate['drop_fraction']:.1%}")
        return gate

    def probe(self, attempt, name, metadata, source_kind):
        self.drain()
        directory = attempt/name
        self.collect(directory, self.probe_seconds, dict(metadata, phase="distance_probe", source_kind=source_kind), show=False)
        reference = summarize_distance(directory)
        reference.update(probe_path=f"{attempt.relative_to(self.root)}/{name}", source_kind=source_kind)
        write_json(directory/"distance-reference.json", reference)
        return reference

    @staticmethod
    def off_nominal(reference, segment):
        distance = reference.get("distance_m")
        if distance is None:
            return True
        nominal = segment["nominal_distance_m"]
        if segment["condition_index"] == 8:
            return not .75 <= distance <= 1.75
        return nominal is not None and abs(distance-nominal) > max(.30, nominal*.25)

    def quality_warnings(self, result, metadata, attempt):
        summary = result["summary"]
        warnings = []
        loss = summary["missing_sequences"]/max(1, summary["frames"]+summary["missing_sequences"])
        if loss > .02:
            warnings.append("丢帧>2%")
        hz = summary["observed_sensor_hz"]
        if hz is None or not 4.5 <= hz <= 5.7:
            warnings.append("帧率异常")
        if summary["frames"] < self.segment_seconds*4.5*.95:
            warnings.append("帧数不足")
        if summary.get("cnh_layout_counts") != {"8x8x16": summary["frames"]}:
            warnings.append("不是全部8×8×16")
        if summary["device_errors"] or summary["sequence_resets_or_reorders"] or summary["duplicate_sequences"] or summary["issues"].get("invalid_frame", 0):
            warnings.append("固件/序号/帧校验异常")
        captured = summarize_distance(attempt)
        reference = metadata["distance_reference"]
        # Thin rod/book proxy can vanish from central zones; this is a prompt, never truth.
        if reference["stable"]:
            if not captured["stable"]:
                warnings.append("中央距离无法与准备读数核对（仅弱参考，细杆可能未被固件看到）")
            elif abs(captured["distance_m"]-reference["distance_m"]) > max(.30, reference["distance_m"]*.25):
                warnings.append("中央固件距离与准备读数变化明显（仅弱参考，可能是杆未被固件看到）")
        return warnings, captured

    def inventory(self, manifest):
        lines = ["segment_id\tattempt\tstatus\tframes\tpath"]
        for a in manifest["attempts"]:
            lines.append(f"{a['segment_id']}\t{a['attempt']}\t{a['status']}\t{a.get('frames', '')}\t{a['path']}")
        (self.root/"inventory.tsv").write_text("\n".join(lines)+"\n", encoding="utf-8")

    def run_session(self, session_number=1, *, resume=False, redo=None):
        manifest_path = self.root/"capture.json"
        if any(self.root.glob("session-*/session.json")):
            raise ValueError("目录含旧复杂版会话，不能混写simple-dorm-v2；用--output另选artifacts.local目录")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("protocol_version") != PROTOCOL_VERSION:
                raise ValueError("目录协议不匹配；禁止混写")
            if not (resume or redo):
                raise ValueError("录制目录已存在，用--resume续录或--output新目录")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            self.print("准备：椅子+纸箱+胶带固定ToF，拖把/扫把杆、一本书、手机；不用卷尺/三脚架。")
            self.print("尽量水平；换条件尽量只挪整套支架。人离开前方与两侧，带轮物体别动。照片可拍可不拍。")
            self.print("一次完成10段：第4段明档空过道后，不挪支架，关灯拉帘录4b；第5段恢复明档，第9段明档独立重启。")
            while True:
                raw = self.ask("大约离地多少厘米？Enter默认120（这是估计，不是实测）: ").strip()
                try:
                    height = float(raw) if raw else 120.
                    if not math.isfinite(height) or not 30 <= height <= 250:
                        raise ValueError()
                    break
                except ValueError:
                    self.print("填大约厘米数，或Enter用估计120cm。")
            manifest = {"schema": "hardware-bringup.real-labeled-capture.v2", "protocol_version": PROTOCOL_VERSION,
                        "created_utc": self.clock.utcnow(), "port": self.port, "transport": dict(self.transport), "attempts": [], "startups": [],
                        "sensor_height_m": height/100, "sensor_height_source": "operator_estimate" if raw else "default_estimate",
                        "sensor_height_uncertainty_m": .15, "tilt_deg": 0, "tilt_source": "nominal_horizontal_assumption",
                        "status": "prepared", "session_complete": False}
            write_json(manifest_path, manifest)
        plan = segment_plan()
        mapping = {str(s['condition_index']): s['segment_id'] for s in segment_plan()}
        mapping.update({s['condition']: s['segment_id'] for s in segment_plan()})
        selected = {mapping.get(str(s), str(s)) for s in (redo or [])}
        if selected - {s["segment_id"] for s in plan}:
            raise ValueError("--redo编号/名称不属于本次10段清单")
        manifest["status"] = "recording"
        manifest["last_requested_transport"] = dict(self.transport)
        write_json(manifest_path, manifest)
        try:
            gate = self.preflight("session_start_or_resume")
            manifest["startups"].append(gate)
            write_json(manifest_path, manifest)
            for segment in plan:
                sid = segment["segment_id"]
                existing = [a for a in manifest["attempts"] if a["segment_id"] == sid]
                if selected and sid not in selected:
                    continue
                if sid not in selected and any(a["status"] in ("complete", "needs_review", "skipped") for a in existing):
                    continue
                if segment["condition_index"] == 9:
                    previous_boot_id = manifest["startups"][-1].get("boot_id")
                    self.device.close()
                    self.device = None
                    restart = "第9段独立重启：断开充电宝/USB供电再重新供电，等ToF重新连上同一Wi-Fi后按Enter: " if self.transport["kind"] == "wifi_tcp" else "第9段独立重启：现在拔下USB再插回，完成后按Enter（随后监听时手动RESET）: "
                    self.ask(restart)
                    gate = self.preflight("independent_unplug_restart_before_segment9")
                    if self.transport["kind"] == "wifi_tcp":
                        changed = bool(previous_boot_id and gate.get("boot_id") and gate["boot_id"] != previous_boot_id)
                        gate["independent_restart"] = {"previous_boot_id": previous_boot_id, "new_boot_id": gate.get("boot_id"), "verified": changed,
                                                       "rule": "Hardware boot_id must change; a TCP reconnect alone is not independent restart"}
                        if not changed:
                            gate["passed"] = False
                            gate["reasons"].append("第9段boot_id未变化：网络重连不是独立断电重启")
                        write_json(self.root/gate["path"]/"gate.json", gate)
                    manifest["startups"].append(gate)
                    write_json(manifest_path, manifest)
                    if not gate["passed"]:
                        raise ValueError("第9段独立重启未通过："+"；".join(gate["reasons"]))
                while True:
                    number = 1+len([a for a in manifest["attempts"] if a["segment_id"] == sid])
                    rel = f"{sid}/attempt-{number:02d}"
                    attempt = self.root/rel
                    self.print(f"\n第{segment['condition_index']}段：{segment['instructions']}")
                    ready = self.ask("摆好按Enter；s跳过；也可直接填可选照片名后Enter: ").strip()
                    record = {"segment_id": sid, "condition": segment["condition"], "condition_index": segment["condition_index"],
                              "lighting": segment["lighting"], "ambient_level": segment["ambient_level"],
                              "ambient_pair_id": segment["ambient_pair_id"], "attempt": number, "path": rel,
                              "status": "skipped" if ready.lower() in ("s", "skip") else "preparing"}
                    manifest["attempts"].append(record)
                    write_json(manifest_path, manifest)
                    if record["status"] == "skipped":
                        break
                    metadata = {**segment, "protocol_version": PROTOCOL_VERSION,
                                "session_number": 1,
                                "sensor_height_m": manifest["sensor_height_m"], "sensor_height_source": manifest["sensor_height_source"],
                                "sensor_height_uncertainty_m": .15, "tilt_deg": 0,
                                "tilt_source": "nominal_horizontal_assumption", "target_height_m": None,
                                "target_x_m": None, "rod_length_m": None, "rod_diameter_m": None,
                                "nominal_target_x_m": .5 if segment["condition_index"] == 7 else 0,
                                "geometry_quality": "operator_estimate_and_firmware_weak_reference",
                                "photo_optional": ready or None, "photo_before": ready or None, "photo_after": None,
                                "startup_record": self.startup_path, "static_scene": True,
                                "distance_source_limit": "central firmware range may be background; book proxy is not rod contact truth"}
                    attempt.mkdir(parents=True, exist_ok=False)
                    write_json(attempt/"metadata.json", dict(metadata, status="preparing"))
                    try:
                        reference = self.probe(attempt, "distance-probe-01", metadata, "scene_central4")
                        if isinstance(segment["condition_index"], int) and 5 <= segment["condition_index"] <= 8 and self.off_nominal(reference, segment):
                            self.ask("杆读数不稳定或偏离建议；请在目标位置临时放一本书，放好按Enter测距: ")
                            reference = self.probe(attempt, "book-probe-02", metadata, "book_proxy")
                            self.ask("书测距完成；现在拿走书，恢复杆摆放，按Enter: ")
                        metadata.update(distance_reference=reference, target_distance_m=reference["distance_m"],
                                        target_distance_source="book_proxy_firmware_weak_reference" if reference["source_kind"] == "book_proxy" else "firmware_central4_status5_weak_reference")
                        self.print(f"自动距离：{reference['distance_m'] if reference['stable'] else '未知'}m（固件弱参考；未知继续）。")
                        write_json(attempt/"metadata.json", dict(metadata, status="ready"))
                        self.ask("按Enter开始；录制时人离开前方与两侧、设备物体不动: ")
                        if segment["condition_index"] in (2, 3):
                            reference = self.probe(attempt, "distance-probe-final", metadata, "scene_central4")
                            metadata.update(distance_reference=reference, target_distance_m=reference["distance_m"])
                        for remaining in range(self.countdown_seconds, 0, -1):
                            self.print(f"{remaining}s")
                            self.drain()
                            self.clock.sleep(1)
                        self.drain()
                        result = self.collect(attempt, self.segment_seconds, metadata, allow_existing=True)
                        warnings, captured = self.quality_warnings(result, metadata, attempt)
                        result["metadata"].update(quality_warnings=warnings, capture_distance_summary=captured,
                                                  status="needs_review" if warnings else "complete")
                        write_json(attempt/"metadata.json", result["metadata"])
                        record.update(status=result["metadata"]["status"], frames=result["summary"]["frames"], warnings=warnings)
                        for previous in manifest["attempts"]:
                            if previous is not record and previous["segment_id"] == sid and previous["status"] in ("complete", "needs_review", "skipped"):
                                previous.update(original_status=previous["status"], status="superseded", superseded_by=rel)
                                old_path = self.root/previous["path"]/"metadata.json"
                                if old_path.exists():
                                    old = json.loads(old_path.read_text(encoding="utf-8"))
                                    old.update(original_status=old["status"], status="superseded", superseded_by=rel)
                                    write_json(old_path, old)
                        write_json(manifest_path, manifest)
                        self.print("好" if not warnings else "检查提示："+"；".join(warnings))
                        if self.ask("Enter下一段；r重录（旧段保留）: ").strip().lower() != "r":
                            break
                    except BaseException as exc:
                        path = attempt/"metadata.json"
                        failed = json.loads(path.read_text(encoding="utf-8"))
                        prior = failed.get("status")
                        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                        if prior in ("preparing", "ready") and isinstance(exc, KeyboardInterrupt):
                            status = "interrupted_before_recording"
                        failed.update(status=status, error=f"{type(exc).__name__}: {exc}")
                        write_json(path, failed)
                        record["status"] = status
                        raise
            manifest["session_complete"] = all(any(a["segment_id"] == s["segment_id"] and a["status"] in ("complete", "needs_review", "skipped") for a in manifest["attempts"]) for s in plan)
            manifest["status"] = "complete" if manifest["session_complete"] else "partial"
            write_json(manifest_path, manifest)
            self.print(f"本次10段结束；清单：{self.root / 'inventory.tsv'}")
            return manifest
        except BaseException as exc:
            manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            manifest["last_error"] = f"{type(exc).__name__}: {exc}"
            write_json(manifest_path, manifest)
            raise
        finally:
            if self.device is not None:
                self.device.close()
                self.device = None
            self.inventory(manifest)

    def collect(self, directory, duration, metadata, *, show=True, allow_existing=False):
        """Exclusive attempt dir; every received byte and frame survives interruption."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=allow_existing)
        metadata = dict(metadata, started_utc=self.clock.utcnow(), status="recording",
                        transport=dict(self.transport),
                        evidence_scope="SIMULATED_SERIAL_REPLAY" if self.replay else "REAL_LABELED_CAPTURE",
                        host_timestamp_rule="chunk-final-newline receipt; host monotonic_ns + UTC; not synchronized")
        metadata['imu'] = self.imu_feed.snapshot() if self.imu_feed else {'status': 'IMU_NOT_CONFIGURED'}
        metadata['host_receipt_clock'] = self.clock.receipt_clock() if hasattr(self.clock, 'receipt_clock') else {'domain':'INJECTED_CLOCK_UNSPECIFIED'}
        context_path = self.root / "operator-context.json"
        if context_path.exists() and not metadata.get("phase"):
            context = json.loads(context_path.read_text(encoding="utf-8-sig"))
            metadata.update(operator_context=context, static_scene=context.get("static_scene", False),
                            mounting=context.get("mounting"), tilt_deg=None,
                            tilt_source=context.get("tilt_source", "unmeasured"))
        write_json(directory / "metadata.json", metadata)
        events, latest = [], None
        ambient_frame_medians = []
        imu_residuals = []
        imu_status_counts = {}
        preflight_decoder, preflight_events = None, []
        digest, total, monitor = hashlib.sha256(), 0, self.clock.monotonic()
        started = self.clock.monotonic()
        error = None
        completed = False
        with (directory / "raw.bin").open("xb") as raw, \
             (directory / "frames.jsonl").open("x", encoding="utf-8") as frames, \
             (directory / "events.jsonl").open("x", encoding="utf-8") as event_file, \
             (directory / "issues.jsonl").open("x", encoding="utf-8") as issues:
            handles = {"frames": frames, "events": event_file, "issues": issues}
            def emit(kind, record):
                nonlocal latest, preflight_decoder, preflight_events
                record = dict(record, host_received_utc=self.clock.utcnow(), host_receipt_clock_domain=metadata['host_receipt_clock']['domain'])
                if kind == 'frames':
                    record['imu'] = self.imu_feed.align(record['host_received_monotonic_ns']) if self.imu_feed else unavailable_alignment()
                    status = record['imu']['status']
                    imu_status_counts[status] = imu_status_counts.get(status, 0) + 1
                    if record['imu']['available']:
                        imu_residuals.append(record['imu']['residual_ms'])
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
                    ambient_values = record["derived"].get("ambient_normalized", [])
                    if ambient_values:
                        ambient_frame_medians.append(statistics.median(ambient_values))
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
                        # Reopen only this exact endpoint during preflight: USB
                        # manual RESET or Wi-Fi drop. Never command/reset/scan.
                        self.device.close()
                        self.clock.sleep(.2)
                        try:
                            self.device = self.serial_factory()
                            decoder.issue("wifi_tcp_reconnect" if self.transport["kind"] == "wifi_tcp" else "manual_reset_serial_reconnect", self.port, self.clock.monotonic_ns())
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
                summary = decoder.summary()
                metadata.update(status="complete" if completed else "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                ended_utc=self.clock.utcnow(), host_elapsed_seconds=self.clock.monotonic()-started,
                                raw_bytes=total, raw_sha256=digest.hexdigest(), summary=summary,
                                error=f"{type(error).__name__}: {error}" if error else None)
                metadata["ambient_summary"] = {
                    "source": "derived.ambient_normalized; CNH raw/scaler-normalized units, not lux",
                    "frames_with_ambient": len(ambient_frame_medians),
                    "median_of_zone_medians": statistics.median(ambient_frame_medians) if ambient_frame_medians else None}
                metadata['imu_end'] = self.imu_feed.snapshot() if self.imu_feed else {'status': 'IMU_NOT_CONFIGURED'}
                ordered_residuals = sorted(abs(value) for value in imu_residuals)
                metadata['imu_alignment_summary'] = dict(frames=summary['frames'], aligned_frames=len(imu_residuals),
                    status_counts=imu_status_counts, signed_mean_residual_ms=statistics.mean(imu_residuals) if imu_residuals else None,
                    absolute_p95_residual_ms=ordered_residuals[int(.95*(len(ordered_residuals)-1))] if ordered_residuals else None,
                    absolute_max_residual_ms=max(ordered_residuals) if ordered_residuals else None,
                    rule='nearest final-newline host receipt; signed residual = IMU - CNH; max absolute 100 ms; up to 30 ms future receipt wait',
                    exposure_synchronized=False)
                if metadata.get("phase") == "preflight":
                    metadata["startup_window_summary"] = preflight_decoder.summary() if preflight_decoder is not None else Decoder().summary()
                write_json(directory / "metadata.json", metadata)
        if error:
            raise error
        return {"metadata": metadata, "summary": summary, "events": events, "latest": latest,
                "preflight_summary": metadata.get("startup_window_summary"), "preflight_events": preflight_events}



def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="宿舍最简H3 v2：一次10段，USB或明确局域网Wi-Fi H3；不刷写")
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--tof-host", help="明确私有IPv4；使用完整H3 Wi-Fi TCP流，未填写则USB串口")
    parser.add_argument("--tof-tcp-port", type=int, default=3335)
    parser.add_argument('--imu-host', help='明确Atom私有IPv4；逐帧IMU TCP，留空则明确标记未配置')
    parser.add_argument('--imu-tcp-port', type=int, default=3336)
    parser.add_argument('--imu-mounting', type=Path, help='已核对IMU轴/水平零点/墙面读数的atom.imu.mount.v1记录')
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--redo", action="append", default=[], metavar="编号或名称")
    parser.add_argument("--replay", type=Path, help="已有raw.bin模拟串口，标记REPLAY，不连接设备")
    parser.add_argument("--output", type=Path, help="明确原目录，尤其跨日/恢复；必须在artifacts.local内")
    args = parser.parse_args(argv)
    imu = None
    repo = Path(__file__).resolve().parents[4]
    try:
        root = args.output or default_output_root(repo/"artifacts.local/work")
        if not root.resolve().is_relative_to((repo/"artifacts.local").resolve()):
            parser.error("输出必须位于canonical artifacts.local内")
        clock = CaptureClock()
        if args.replay and args.tof_host:
            parser.error("--replay与--tof-host不能同时使用")
        if args.replay and args.imu_host:
            parser.error('--replay不连接实际IMU')
        if args.imu_mounting and not args.imu_host:
            parser.error('--imu-mounting需要--imu-host')
        host = private_tof_host(args.tof_host) if args.tof_host else None
        if not 1 <= args.tof_tcp_port <= 65535:
            parser.error("ToF TCP端口必须在1–65535")
        transport = {"kind": "wifi_tcp", "host": host, "tcp_port": args.tof_tcp_port} if host else None
        factory = (lambda: ReplaySerial(args.replay, clock)) if args.replay else (lambda: WiFiStream(host, args.tof_tcp_port)) if host else (lambda: open_serial(args.port))
        source = f"tcp://{host}:{args.tof_tcp_port}" if host else args.port
        if args.imu_host:
            root.mkdir(parents=True, exist_ok=True)
            mounting = load_mounting(args.imu_mounting) if args.imu_mounting else None
            imu = ImuFeed(private_host(args.imu_host), root/f'imu-{uuid.uuid4().hex}', args.imu_tcp_port, mounting=mounting).start()
        app = GuidedCapture(root, factory, clock=clock, port=source, replay=bool(args.replay), transport=transport, imu_feed=imu)
        app.run_session(resume=args.resume, redo=args.redo)
        return 0
    except KeyboardInterrupt:
        print(f'已中断，证据保留：--resume --output "{root}"')
        return 130
    except (ValueError, OSError) as exc:
        print(f"停止：{exc}")
        return 2
    finally:
        if imu is not None:
            imu.close()


if __name__ == "__main__":
    raise SystemExit(main())
