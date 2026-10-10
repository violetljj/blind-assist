"""Local live CNH/M3 bench viewer; current-scene residuals are engineering inputs.

No training, scene truth, firmware flashing, or changes to frozen experiments.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlsplit, parse_qs

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SERIAL_SITE = ROOT / 'artifacts.local/hardware-bringup/venv/Lib/site-packages'
sys.path.append(str(SERIAL_SITE))
from capture import validate_frame
from m3_live_engine import Engine
from m3_live_camera import CameraFeed
from m3_wifi_setup import WifiSetup, profiles

PAGE_PATH = Path(__file__).with_suffix('.html')


def reference_quality(histograms):
    """Display-only scene drift heuristic; does not accept/reject model inputs."""
    h = np.asarray(histograms, dtype=np.float64)
    middle = len(h)//2
    sd = h.std(0, ddof=1)
    floor = max(float(np.median(sd[sd > 0]))*1e-3 if (sd > 0).any() else 0., 1e-9)
    shift = np.abs(h[:middle].mean(0)-h[middle:].mean(0))/np.maximum(sd, floor)
    p95 = float(np.percentile(shift, 95))
    return dict(half_mean_shift_p95=p95, scene_change_warning=p95 > .8,
                rule='display-only: p95 half-mean shift / sample SD > 0.8',
                scope='Scene-change hint, not physical noise or obstacle-free calibration')

class Demo:
    def __init__(self, args):
        self.args = args
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.reconnects = 0
        self.acquisition_epoch = 0
        self.reference_request = not bool(getattr(args, 'reference', None))
        self.reference_version = 0
        self.reference_frames = []
        self.latest = None
        self.reference_mean = None
        self.reference_sd = None
        self.trend = deque(maxlen=110)
        self.compute_times = deque(maxlen=100)
        self.state = dict(status='加载模型', reference_count=0, reference_ready=False,
                          models_ready=False, error=None, input_errors=0,
                          reference_total=100, reference_collecting=self.reference_request,
                          reference_version=0, reference_resets=0)
        self.receipts = deque(maxlen=50)
        self.threads = []
        self.camera = CameraFeed(args.out / 'camera', self.stop, url=getattr(args, 'camera_url', None),
                                 device_id=getattr(args, 'camera_id', None), max_seconds=getattr(args, 'max_seconds', 3600)) if getattr(args, 'camera_id', None) or getattr(args, 'camera_url', None) else None
        self.wifi = WifiSetup(self.stop, self.camera, getattr(args, 'camera_port', 'COM11'),
                              getattr(args, 'camera_id', None) or 'b4:3a:45:bd:12:d8')

    def snapshot(self):
        with self.lock:
            result = dict(self.state)
            age = (time.monotonic_ns() - result.get('received_ns', 0)) / 1e6
            prediction_age = (time.monotonic_ns() - result.get('prediction_received_ns', 0)) / 1e6
            result['stale'] = age > 1500 or (result['reference_ready'] and prediction_age > 1500)
            result.update(receipt_age_ms=max(0, age), prediction_age_ms=max(0, prediction_age),
                          trend=list(self.trend), reference_version=self.reference_version,
                          compute_median_ms=float(np.median(self.compute_times)) if self.compute_times else None)
            if result['stale'] and result.get('models_ready'):
                result['status'] = '输入中断 · UNKNOWN' if age > 1500 else '等待新预测 · UNKNOWN'
            elif self.reference_request:
                result['status'] = '采集场景参照'
        result['camera'] = self.camera.snapshot() if self.camera else dict(status='未开启相机', available=False)
        result['wifi'] = self.wifi.snapshot()
        return result

    def zone(self, index):
        if not 0 <= index < 64:
            raise ValueError('Zone index must be 0..63')
        with self.lock:
            current = self.latest[2].reshape(64,16)[index].tolist() if self.latest else None
            mean = self.reference_mean.reshape(64,16)[index].tolist() if self.reference_mean is not None else None
            return dict(index=index, row=index//8, col=index%8, current=current, reference=mean,
                        bin_width_m=8*.0375348, seq=self.state.get('seq'),
                        received_ns=self.state.get('received_ns'), reference_version=self.reference_version)

    def reference(self):
        with self.condition:
            self.reference_request = True
            self.reference_version += 1
            self.reference_frames = []
            self.reference_mean = self.reference_sd = None
            self.trend.clear()
            self.compute_times.clear()
            self.state.update(reference_count=0, reference_ready=False, prediction=None,
                              reference_collecting=True, status='采集场景参照', residual_max=None,
                              reference_quality=None, error=None)
            self.condition.notify_all()

    def acquire(self):
        import serial
        raw_path = self.args.out / 'live-frames.jsonl'
        try:
            with raw_path.open('x', encoding='utf-8') as log, (self.args.out / 'serial.bin').open('xb') as raw, (self.args.out / 'rejected.jsonl').open('x', encoding='utf-8') as rejected:
                while not self.stop.is_set():
                    try:
                        with serial.Serial(port=None, baudrate=115200, timeout=.2) as link:
                            link.dtr = False
                            link.rts = False
                            link.port = self.args.port
                            link.open()
                            with self.lock:
                                self.state.update(serial_released=False, reconnecting=False)
                            self.read_serial(link, log, raw, rejected)
                    except (OSError, serial.SerialException) as exc:
                        with self.condition:
                            self.reconnects += 1
                            self.acquisition_epoch += 1
                            self.latest = None
                            self.receipts.clear()
                            self.trend.clear()
                            self.state.update(error='串口: '+str(exc), prediction=None, received_ns=0,
                                              reconnecting=True, reconnects=self.reconnects)
                            if self.reference_request:
                                self.reference_frames.clear()
                                self.state.update(reference_count=0, reference_resets=self.state['reference_resets']+1)
                        self.stop.wait(1.5)
        except Exception as exc:
            with self.lock:
                self.state['error'] = '串口: '+str(exc)
        finally:
            with self.lock:
                self.state['serial_released'] = True

    def read_serial(self, link, log, raw, rejected):
        pending = bytearray()
        previous = None
        while not self.stop.is_set():
            chunk = link.read(max(1, link.in_waiting))
            raw.write(chunk)
            pending.extend(chunk)
            if len(pending) > 2_000_000:
                rejected.write(json.dumps(dict(reason='serial buffer overflow', bytes=len(pending)))+'\n')
                rejected.flush()
                pending.clear()
            while b'\n' in pending:
                line, _, pending = pending.partition(b'\n')
                raw.flush()
                try:
                    sensor = json.loads(line)
                    if sensor.get('type') != 'cnh_frame':
                        continue
                    derived = validate_frame(sensor)
                    if (derived['rows'], derived.get('bins')) != (8, 16):
                        raise ValueError('Expected 8x8x16 CNH')
                    h = np.asarray(derived['hist_normalized'], dtype=np.float64).reshape(8, 8, 16)
                    if not np.isfinite(h).all():
                        raise ValueError('Nonfinite CNH input')
                    stamp = time.monotonic_ns()
                    continuous = previous is None or (sensor['seq'] == previous[0]+1 and 0 < sensor['ms']-previous[1] <= 500)
                    previous = (sensor['seq'], sensor['ms'])
                    log.write(json.dumps(dict(sensor=sensor, host_received_monotonic_ns=stamp))+'\n')
                    log.flush()
                    with self.condition:
                        self.receipts.append(stamp)
                        hz = (len(self.receipts)-1)*1e9/(stamp-self.receipts[0]) if len(self.receipts)>1 else None
                        if self.reference_request:
                            if not continuous:
                                self.reference_frames.clear()
                                self.state['reference_resets'] += 1
                            if len(self.reference_frames) < 100:
                                self.reference_frames.append(h)
                            self.state['reference_count'] = len(self.reference_frames)
                        self.latest = (sensor['seq'], sensor['ms'], h, self.reference_version, stamp, self.acquisition_epoch)
                        distance = derived['distance_known_mm']
                        self.state.update(seq=sensor['seq'], received_ns=stamp, sensor_hz=hz,
                                          distance=distance, valid_zones=sum(d is not None for d in distance),
                                          hist_max=h.max(-1).flatten().tolist())
                        self.condition.notify_all()
                except (ValueError, UnicodeError, KeyError) as exc:
                    rejected.write(json.dumps(dict(received_ns=time.monotonic_ns(), reason=str(exc),
                                                   serial_line=line.decode('utf-8', errors='replace')))+'\n')
                    rejected.flush()
                    with self.lock:
                        self.state['input_errors'] += 1
                        if not isinstance(exc, json.JSONDecodeError):
                            self.state['error'] = '输入检查: '+str(exc)

    def infer(self):
        engine = None
        processed = None
        previous_epoch = None
        previous_receipt_ns = None
        try:
            engine = Engine(feature_precision='float32-engineering')
            (self.args.out / 'model-identity.json').write_text(json.dumps(engine.metadata(), indent=2), encoding='utf-8')
            if self.args.reference:
                with np.load(self.args.reference, allow_pickle=False) as saved:
                    reference = saved['histograms']
                receipt = engine.set_reference(reference)
                quality = reference_quality(reference)
                receipt.update(quality=quality, restored_from=str(self.args.reference.resolve()))
                np.savez_compressed(self.args.out / 'reference-0.npz', histograms=reference)
                (self.args.out / 'reference-0.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
                with self.lock:
                    if self.reference_version == 0 and not self.reference_request:
                        self.reference_mean, self.reference_sd = engine.mean.copy(), engine.std.copy()
                        self.state.update(reference_ready=True, reference_count=len(reference),
                                          reference_collecting=False, reference_quality=quality, reference_restored=True)
            with self.lock:
                self.state.update(models_ready=True, status='采集场景参照')
            with (self.args.out / 'inference.jsonl').open('x', encoding='utf-8') as log:
                while not self.stop.is_set():
                    with self.condition:
                        self.condition.wait_for(lambda: self.stop.is_set() or (self.latest is not None and (self.latest[0], self.latest[3], self.latest[5]) != processed), timeout=.5)
                        if self.stop.is_set():
                            break
                        current = self.latest
                        if current is None or (current[0], current[3], current[5]) == processed:
                            continue
                        reference = np.asarray(self.reference_frames) if self.reference_request and len(self.reference_frames) >= 100 else None
                        need_reference = self.reference_request
                        version = self.reference_version
                    if reference is not None:
                        reference_receipt = engine.set_reference(reference[:100])
                        quality = reference_quality(reference[:100])
                        reference_receipt.update(quality=quality)
                        np.savez_compressed(self.args.out / f'reference-{version}.npz', histograms=reference[:100])
                        (self.args.out / f'reference-{version}.json').write_text(json.dumps(reference_receipt, indent=2), encoding='utf-8')
                        with self.lock:
                            if version == self.reference_version:
                                self.reference_request = False
                                self.reference_mean, self.reference_sd = engine.mean.copy(), engine.std.copy()
                                self.state.update(reference_ready=True, reference_count=100,
                                                  reference_collecting=False, reference_quality=quality,
                                                  reference_restored=False)
                        need_reference = False
                    processed = (current[0], current[3], current[5])
                    if need_reference:
                        continue
                    input_received_ns = current[4]
                    if ((previous_epoch is not None and current[5] != previous_epoch) or
                            (previous_receipt_ns is not None and input_received_ns-previous_receipt_ns > 1_500_000_000)):
                        engine.reset()
                        with self.lock:
                            self.trend.clear()
                    previous_epoch, previous_receipt_ns = current[5], input_received_ns
                    try:
                        result = engine.step(current[2], current[0], current[1])
                    except ValueError as exc:
                        engine.reset()
                        log.write(json.dumps(dict(seq=current[0], ms=current[1], status='UNKNOWN',
                                                  input_error=str(exc), reference_version=current[3]))+'\n')
                        log.flush()
                        with self.lock:
                            self.trend.clear()
                            self.state.update(prediction=None, error='本帧输入: '+str(exc), status='输入不可评价 · UNKNOWN')
                        continue
                    result.update(smooth=result['smoothed'], crossing=result['reference_crossing'], processing_ms=result['compute_ms'])
                    result.update(reference_version=current[3], acquisition_epoch=current[5])
                    log.write(json.dumps(result, allow_nan=False)+'\n')
                    log.flush()
                    residual = engine.normalized(current[2])
                    with self.lock:
                        if current[3] == self.reference_version and current[5] == self.acquisition_epoch and not self.reference_request:
                            self.compute_times.append(result['compute_ms'])
                            if result['reset']:
                                self.trend.clear()
                            if result['ready']:
                                self.trend.append(dict(t_ms=input_received_ns/1e6, seq=current[0],
                                                       head=result['smooth'][0], body=result['smooth'][1]))
                            self.state.update(status='实时 M3 · 工程输入', prediction=result,
                                              residual_max=residual.max(-1).flatten().tolist(), error=None,
                                              prediction_received_ns=input_received_ns)
        except Exception as exc:
            with self.lock:
                self.state.update(error='M3: '+str(exc), prediction=None, status='M3 错误 · UNKNOWN')
            import traceback
            traceback.print_exc()
        finally:
            if engine is not None:
                engine.close()
            with self.lock:
                self.state['model_released'] = True

    def start(self):
        for func in (self.acquire, self.infer) + ((self.camera.run,) if self.camera else ()):
            worker = threading.Thread(target=func, daemon=True)
            worker.start()
            self.threads.append(worker)

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        for worker in self.threads:
            worker.join(timeout=15)
        self.wifi.close()
        (self.args.out / 'final-status.json').write_text(json.dumps(self.snapshot(), indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='COM5')
    parser.add_argument('--http-port', type=int, default=8766)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=3600)
    parser.add_argument('--reference', type=Path, help='Reuse an actual recorded scene reference NPZ')
    parser.add_argument('--camera-url', help='Explicit local Atom http://IP:81/stream')
    parser.add_argument('--camera-id', default='b4:3a:45:bd:12:d8', help='Atom station MAC for discovery; empty disables discovery')
    parser.add_argument('--camera-port', default='COM11', help='Selected Atom USB port for Wi-Fi setup')
    args = parser.parse_args()
    if args.camera_port.lower() == args.port.lower():
        parser.error('Camera Wi-Fi setup and ToF must use separate USB ports')
    args.out.mkdir(parents=True, exist_ok=False)
    import shutil
    source_dir = args.out / 'source'
    source_dir.mkdir()
    for source in (Path(__file__), Path(__file__).with_name('m3_live_engine.py'), Path(__file__).with_name('m3_live_camera.py'), Path(__file__).with_name('m3_wifi_setup.py'), Path(__file__).with_name('m3_camera_view.js'), PAGE_PATH):
        shutil.copyfile(source, source_dir / source.name)
    demo = Demo(args)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, data, content_type='application/json; charset=utf-8', headers=None):
            payload = data.encode('utf-8') if isinstance(data, str) else data
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            for name, value in (headers or {}).items():
                self.send_header(name, str(value))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == '/':
                self.respond(PAGE_PATH.read_text(encoding='utf-8'), 'text/html; charset=utf-8')
            elif url.path == '/m3_camera_view.js':
                self.respond(PAGE_PATH.with_name('m3_camera_view.js').read_text(encoding='utf-8'), 'text/javascript; charset=utf-8')
            elif url.path == '/api/camera.mjpeg':
                if not demo.camera or not demo.camera.snapshot()['available']:
                    self.send_error(503, 'No fresh camera frame')
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=BAFRAME')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.close_connection = True
                self.connection.settimeout(2)
                try:
                    for header, jpeg in demo.camera.stream_frames():
                        part = (f'--BAFRAME\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n'
                                f'X-Frame-Sequence: {header["seq"]}\r\nX-Sequence-Id: {header["sequence_id"]}\r\n'
                                f'X-Host-Age-Ms: {header["host_age_ms"]:.3f}\r\n'
                                f'X-Host-Sent-Unix-Ms: {time.time()*1000:.3f}\r\n\r\n').encode('ascii')
                        self.wfile.write(part + jpeg + b'\r\n')
                        self.wfile.flush()
                except OSError:
                    pass  # Closed/slow browser; camera acquisition remains independent.
            elif url.path == '/api/status':
                self.respond(json.dumps(demo.snapshot(), allow_nan=False))
            elif url.path == '/api/wifi/profiles':
                try:
                    self.respond(json.dumps(profiles()))
                except (ValueError, OSError, __import__('subprocess').TimeoutExpired):
                    self.respond(json.dumps(dict(profiles=[], current=None)))
            elif url.path == '/api/camera.jpg':
                jpeg, header = demo.camera.image() if demo.camera else (None, None)
                if jpeg is None:
                    self.send_error(503, 'No fresh camera frame')
                else:
                    self.respond(jpeg, 'image/jpeg', {
                        'X-Frame-Sequence': header['seq'], 'X-Sequence-Id': header['sequence_id'],
                        'X-Host-Age-Ms': round(header['host_age_ms'], 3)})
            elif url.path == '/api/zone':
                try:
                    index = int(parse_qs(url.query).get('index', ['27'])[0])
                    self.respond(json.dumps(demo.zone(index), allow_nan=False))
                except ValueError:
                    self.send_error(400, 'Zone index must be 0..63')
            else:
                self.send_error(404)

        def do_POST(self):
            origin = self.headers.get('Origin')
            if origin and origin != f'http://127.0.0.1:{args.http_port}':
                self.send_error(403)
                return
            if self.path == '/api/camera/wifi':
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 2048:
                        raise ValueError('配置输入长度无效')
                    data = json.loads(self.rfile.read(length))
                    if not isinstance(data, dict):
                        raise ValueError('配置格式无效')
                    demo.wifi.request(data)
                except (ValueError, UnicodeError) as exc:
                    self.respond(json.dumps(dict(ok=False, error=str(exc))))
                    return
            elif self.path == '/api/reference':
                demo.reference()
            elif self.path == '/api/stop':
                demo.stop.set()
            else:
                self.send_error(404)
                return
            self.respond('{"ok":true}')

    server = ThreadingHTTPServer(('127.0.0.1', args.http_port), Handler)
    server.timeout = .25
    (args.out / 'service.json').write_text(json.dumps(dict(pid=__import__('os').getpid(), port=args.port,
        url=f'http://127.0.0.1:{args.http_port}', started_utc=datetime.now(timezone.utc).isoformat(),
        max_seconds=args.max_seconds, reference='current scene, not obstacle-free truth',
        input_scope='empirical per-bin background residual; fixed sensor; no physical accuracy claim'), indent=2), encoding='utf-8')
    try:
        demo.start()
        deadline = time.monotonic() + args.max_seconds
        while not demo.stop.is_set() and time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()
        demo.close()


if __name__ == '__main__':
    main()
