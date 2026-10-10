"""Local live CNH/M3 bench viewer; current-scene residuals are engineering inputs.

No training, scene truth, firmware writes, or changes to frozen experiments.
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

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SERIAL_SITE = ROOT / 'artifacts.local/hardware-bringup/venv/Lib/site-packages'
sys.path.append(str(SERIAL_SITE))
from capture import validate_frame
from m3_live_engine import Engine

PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>M3 实时台架</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#10191f;color:#edf3f4;font:16px system-ui,"Microsoft YaHei",sans-serif}
main{max-width:1150px;margin:auto;padding:30px}h1{font-size:30px;margin:0 0 8px}p{line-height:1.7;color:#a7b9c2}
.live{color:#62dfb7}.bar{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.pill{background:#21333d;border-radius:30px;padding:8px 14px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:20px}.card{background:#17252e;border:1px solid #2a3f4b;border-radius:18px;padding:22px}
h2{font-size:20px;margin:0 0 15px}.grid{display:grid;grid-template-columns:repeat(8,1fr);gap:4px}.cell{text-align:center;border-radius:5px;padding:11px 0;font:13px ui-monospace,monospace;min-width:0}
.score{font:54px ui-monospace,monospace;margin:12px 0}.muted{font-size:13px;color:#9bb1bf}.badge{color:#ffc88d;min-height:25px}.warn{background:#382e24;border:1px solid #5a4631;border-radius:12px;padding:14px;line-height:1.7}
button,select{font:inherit;background:#244956;color:white;border:1px solid #477386;border-radius:9px;padding:10px 16px;cursor:pointer}button:hover{background:#356473}button.stop{background:#543332}
progress{width:100%;height:16px;accent-color:#62dfb7}footer{margin-top:20px}#error{color:#ffa99b;white-space:pre-wrap}
@media(max-width:760px){main{padding:18px}.cols{grid-template-columns:1fr}.cell{padding:10px 0;font-size:11px}}
</style><main>
<h1>M3 · 实时 CNH 台架</h1><p>真实传感器 → 最近 8 帧空间投影 → 冻结 M3 五模型 → 身体 / 头部分数</p>
<div class="bar"><span class="pill live" id="state">连接中</span><span class="pill" id="seq">—</span><span class="pill" id="hz">—</span><span class="pill" id="latency">—</span></div>
<div class="card" style="margin-top:20px"><h2 id="refTitle">采集当前场景参照</h2><progress id="progress" max="100" value="0"></progress><p id="refText">首次采集约 20 秒。保持 ToF 和场景静止；之后移动物体观察响应。</p>
<div class="bar"><button onclick="act('reference')">重新采集场景参照</button><button id="sound" onclick="toggleSound()">开启越线提示音</button><button class="stop" onclick="act('stop')">结束演示并释放设备</button></div></div>
<div class="cols"><div class="card"><h2>头部 HEAD</h2><div class="score" id="head">—</div><div class="badge" id="headFlag">等待输入</div><div class="muted">平滑 logit，数值不是概率</div></div>
<div class="card"><h2>身体 BODY</h2><div class="score" id="body">—</div><div class="badge" id="bodyFlag">等待输入</div><div class="muted">固定台架、名义俯角 −10°</div></div></div>
<div class="cols"><div class="card"><h2>传感器原始区域距离 · mm</h2><div class="grid" id="range"></div><p class="muted">8×8 原始区域顺序；无有效距离显示 ?。不是相机上的物体位置。</p></div>
<div class="card"><div class="bar"><h2>CNH 回波</h2><select id="view"><option value="residual">相对场景参照的最大变化</option><option value="raw">原始每区最大 bin 幅度</option></select></div><div class="grid" id="hist"></div><p class="muted" id="scale">每区 16 个距离 bin；色标随当前帧缩放。</p></div></div>
<footer><div class="warn">本演示运行原 M3 权重，输入采用当前场景均值与波动归一化后的残差，并以 FP32 保留真实强回波，避免 FP16 上溢。参照中已有物体也会被减去，因此这里展示背景变化响应。参考线 0.855764 来自模拟，尚未标定实物报警效果；越线不能解释成已确认障碍。设备必须固定，移动设备后需重新采集参照。</div><p id="error"></p><p class="muted" id="identity"></p></footer>
</main><script>
let sound=false,context=null,previous=false,lastBeep=0,ended=false;
const el=id=>document.getElementById(id);
async function act(action){try{const r=await fetch('/api/'+action,{method:'POST'});if(!r.ok)throw Error('操作未完成');if(action==='stop'){ended=true;el('state').textContent='已结束，正在释放设备';el('head').textContent=el('body').textContent='—';previous=false;}}catch(e){el('error').textContent='操作失败，请重试：'+e.message;}}
function toggleSound(){sound=!sound;el('sound').textContent=sound?'关闭越线提示音':'开启越线提示音';if(sound){context=context||new AudioContext();context.resume()}}
function beep(){if(!sound||!context)return;const o=context.createOscillator(),g=context.createGain();o.connect(g);g.connect(context.destination);o.frequency.value=650;g.gain.value=.08;o.start();o.stop(context.currentTime+.16)}
function grid(id,vals,type){const max=Math.max(1,...vals.filter(v=>v!==null).map(Math.abs));el(id).innerHTML=vals.map(v=>{const t=v===null?'?':type==='range'?Math.round(v):Number(v).toPrecision(2);const ratio=v===null?0:Math.min(1,Math.abs(v)/max);const col=type==='range'?`hsl(${170-Math.min(1,(v||0)/3000)*110} 35% 25%)`:`hsl(${v<0?220:165} 45% ${17+ratio*25}%)`;return `<div class="cell" style="background:${col}">${t}</div>`}).join('')}
async function update(){if(ended)return;try{const s=await(await fetch('/api/status',{cache:'no-store'})).json();el('state').textContent=s.status;el('seq').textContent=s.seq===undefined?'COM5':`COM5 · 帧 ${s.seq}`;el('hz').textContent=s.sensor_hz?`${s.sensor_hz.toFixed(2)} Hz`:'8×8×16';el('latency').textContent=s.prediction?`推理 ${s.prediction.processing_ms.toFixed(1)} ms`:'初始化';el('progress').value=s.reference_count;el('refTitle').textContent=s.reference_ready?'当前场景参照已就绪':'采集当前场景参照';el('refText').textContent=s.reference_ready?'现在可移动物体观察模型分数。设备移动或场景重置后，重新采集参照。':`已采集 ${s.reference_count} / 100 帧，请保持设备和场景静止。`;if(s.distance)grid('range',s.distance,'range');const vals=el('view').value==='raw'?s.hist_max:s.residual_max;if(vals)grid('hist',vals,'hist');const p=s.prediction;let crossing=false;for(const [id,index]of [['head',0],['body',1]]){el(id).textContent=p&&!s.stale?p.smooth[index].toFixed(3):'—';el(id+'Flag').textContent=s.stale?'输入中断 · UNKNOWN':!p?'等待参照与推理':!p.ready?`历史预热 ${p.history}/8`:p.crossing[index]?'超过模拟参考线':'低于模拟参考线';if(p&&p.ready&&!s.stale&&p.crossing[index])crossing=true}if(crossing&&!previous&&Date.now()-lastBeep>3000){beep();lastBeep=Date.now()}previous=crossing;el('error').textContent=s.error||'';el('identity').textContent=s.models_ready?'原始冻结 M3 · seed 0–4 · 当前/历史投影 · 本机运行':'加载冻结模型中';}catch(e){el('state').textContent='连接中断 · UNKNOWN';el('head').textContent=el('body').textContent='—';previous=false;}setTimeout(update,250)}update();
</script></html>'''


class Demo:
    def __init__(self, args):
        self.args = args
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.reference_request = True
        self.reference_version = 0
        self.reference_frames = []
        self.latest = None
        self.state = dict(status='加载模型', reference_count=0, reference_ready=False,
                          models_ready=False, error=None, input_errors=0)
        self.receipts = deque(maxlen=50)
        self.threads = []

    def snapshot(self):
        with self.lock:
            result = dict(self.state)
            age = (time.monotonic_ns() - result.get('received_ns', 0)) / 1e6
            prediction_age = (time.monotonic_ns() - result.get('prediction_received_ns', 0)) / 1e6
            result['stale'] = age > 1500 or (result['reference_ready'] and prediction_age > 1500)
            if result['stale'] and result.get('models_ready'):
                result['status'] = '输入中断 · UNKNOWN'
            return result

    def reference(self):
        with self.condition:
            self.reference_request = True
            self.reference_version += 1
            self.reference_frames = []
            self.state.update(reference_count=0, reference_ready=False, prediction=None)
            self.condition.notify_all()

    def acquire(self):
        import serial
        raw_path = self.args.out / 'live-frames.jsonl'
        try:
            with serial.Serial(port=None, baudrate=115200, timeout=.2) as link, raw_path.open('x', encoding='utf-8') as log, (self.args.out / 'serial.bin').open('xb') as raw, (self.args.out / 'rejected.jsonl').open('x', encoding='utf-8') as rejected:
                link.dtr = False
                link.rts = False
                link.port = self.args.port
                link.open()
                pending = bytearray()
                previous = None
                while not self.stop.is_set():
                    chunk = link.read(max(1, link.in_waiting))
                    raw.write(chunk)
                    raw.flush()
                    pending.extend(chunk)
                    if len(pending) > 2_000_000:
                        pending.clear()
                    while b'\n' in pending:
                        line, _, pending = pending.partition(b'\n')
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
                                        self.reference_frames = []
                                    self.reference_frames.append(h)
                                    self.state['reference_count'] = len(self.reference_frames)
                                self.latest = (sensor['seq'], sensor['ms'], h, self.reference_version, stamp)
                                self.state.update(seq=sensor['seq'], received_ns=stamp, sensor_hz=hz,
                                                  distance=derived['distance_known_mm'], hist_max=h.max(-1).flatten().tolist())
                                self.condition.notify_all()
                        except (ValueError, UnicodeError, KeyError) as exc:
                            rejected.write(json.dumps(dict(received_ns=time.monotonic_ns(), reason=str(exc),
                                                           serial_line=line.decode('utf-8', errors='replace')))+'\n')
                            rejected.flush()
                            if isinstance(exc, json.JSONDecodeError):
                                continue  # Opening in the middle of one serial frame is expected.
                            with self.lock:
                                self.state['error'] = '输入检查: '+str(exc)
                                self.state['input_errors'] += 1
        except Exception as exc:
            with self.lock:
                self.state['error'] = '串口: '+str(exc)
        finally:
            with self.lock:
                self.state['serial_released'] = True

    def infer(self):
        engine = None
        processed = None
        try:
            engine = Engine(feature_precision='float32-engineering')
            (self.args.out / 'model-identity.json').write_text(json.dumps(engine.metadata(), indent=2), encoding='utf-8')
            with self.lock:
                self.state.update(models_ready=True, status='采集场景参照')
            with (self.args.out / 'inference.jsonl').open('x', encoding='utf-8') as log:
                while not self.stop.is_set():
                    with self.condition:
                        self.condition.wait_for(lambda: self.stop.is_set() or (self.latest is not None and (self.latest[0], self.latest[3]) != processed), timeout=.5)
                        if self.stop.is_set():
                            break
                        current = self.latest
                        if current is None or (current[0], current[3]) == processed:
                            continue
                        reference = np.asarray(self.reference_frames) if self.reference_request and len(self.reference_frames) >= 100 else None
                        need_reference = self.reference_request
                        version = self.reference_version
                    if reference is not None:
                        reference_receipt = engine.set_reference(reference[:100])
                        np.savez_compressed(self.args.out / f'reference-{version}.npz', histograms=reference[:100])
                        (self.args.out / f'reference-{version}.json').write_text(json.dumps(reference_receipt, indent=2), encoding='utf-8')
                        with self.lock:
                            if version == self.reference_version:
                                self.reference_request = False
                                self.state.update(reference_ready=True, reference_count=100)
                        need_reference = False
                    processed = (current[0], current[3])
                    if need_reference:
                        continue
                    input_received_ns = current[4]
                    try:
                        result = engine.step(current[2], current[0], current[1])
                    except ValueError as exc:
                        engine.reset()
                        log.write(json.dumps(dict(seq=current[0], ms=current[1], status='UNKNOWN',
                                                  input_error=str(exc), reference_version=current[3]))+'\n')
                        log.flush()
                        with self.lock:
                            self.state.update(prediction=None, error='本帧输入: '+str(exc), status='输入不可评价 · UNKNOWN')
                        continue
                    result.update(smooth=result['smoothed'], crossing=result['reference_crossing'], processing_ms=result['compute_ms'])
                    log.write(json.dumps(result, allow_nan=False)+'\n')
                    log.flush()
                    residual = engine.normalized(current[2])
                    with self.lock:
                        if current[3] == self.reference_version and not self.reference_request:
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
        for func in (self.acquire, self.infer):
            worker = threading.Thread(target=func, daemon=True)
            worker.start()
            self.threads.append(worker)

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        for worker in self.threads:
            worker.join(timeout=15)
        (self.args.out / 'final-status.json').write_text(json.dumps(self.snapshot(), indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='COM5')
    parser.add_argument('--http-port', type=int, default=8766)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=3600)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import shutil
    source_dir = args.out / 'source'
    source_dir.mkdir()
    for source in (Path(__file__), Path(__file__).with_name('m3_live_engine.py')):
        shutil.copyfile(source, source_dir / source.name)
    demo = Demo(args)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, data, content_type='application/json; charset=utf-8'):
            payload = data.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == '/':
                self.respond(PAGE, 'text/html; charset=utf-8')
            elif self.path == '/api/status':
                self.respond(json.dumps(demo.snapshot(), allow_nan=False))
            else:
                self.send_error(404)

        def do_POST(self):
            origin = self.headers.get('Origin')
            if origin and origin != f'http://127.0.0.1:{args.http_port}':
                self.send_error(403)
                return
            if self.path == '/api/reference':
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
