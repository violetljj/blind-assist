"""Loopback web UI for the existing GuidedCapture; loading never opens serial."""
from __future__ import annotations
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import threading
import time
import sys
import uuid
from m3_live_camera import CameraFeed
from real_labeled_capture import GuidedCapture, CaptureClock, open_serial, default_output_root, segment_plan

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

class InterruptClock(CaptureClock):
    def __init__(self, stop):self.stop=stop
    def sleep(self, seconds):
        if self.stop.wait(seconds):raise KeyboardInterrupt('网页停止')

class InterruptSerial:
    def __init__(self, device, stop):self.device,self.stop=device,stop
    @property
    def timeout(self):return self.device.timeout
    @timeout.setter
    def timeout(self,value):self.device.timeout=value
    def read(self,size):
        if self.stop.is_set():raise KeyboardInterrupt('网页停止')
        return self.device.read(size)
    def reset_input_buffer(self):return self.device.reset_input_buffer()
    def close(self):return self.device.close()

class Bridge:
    def __init__(self, output, *, port='COM5', resume=False, redo=None, serial_factory=None, capture_factory=GuidedCapture):
        self.output=Path(output);self.port=port;self.resume=resume;self.redo=redo or []
        self.serial_factory=serial_factory;self.capture_factory=capture_factory
        self.lock=threading.RLock();self.stop_event=threading.Event();self.answers=queue.Queue()
        self.logs=deque(maxlen=180);self.prompt='';self.prompt_id=0;self.waiting=False
        self.phase='idle';self.error=None;self.worker=None;self.capture=None;self.closing=False
        self.latest=None;self.frame_path=None;self.frame_offset=0;self.frame_pending=b''
        self.camera=None;self.camera_stop=threading.Event();self.camera_worker=None
    def log(self,*values,**kwargs):
        with self.lock:self.logs.append(' '.join(str(v) for v in values))
    def ask(self,prompt):
        with self.lock:self.prompt=prompt;self.prompt_id+=1;self.waiting=True
        try:
            while not self.stop_event.is_set():
                try:return self.answers.get(timeout=.2)
                except queue.Empty:pass
            raise KeyboardInterrupt('网页停止')
        finally:
            with self.lock:self.waiting=False
    def reply(self,text,prompt_id):
        with self.lock:
            if not self.waiting or prompt_id!=self.prompt_id:raise ValueError('提示已变化，请等待页面更新')
            self.waiting=False;self.answers.put(str(text))
    def start(self,payload):
        with self.lock:
            if self.closing or self.worker and self.worker.is_alive():raise ValueError('已有任务正在运行')
            port=str(payload.get('port') or self.port).strip()
            if not port or port.lower()=='auto':raise ValueError('请填写明确ToF串口')
            redo=str(payload.get('redo','')).strip()
            selected=[p.strip() for p in redo.split(',') if p.strip()] or self.redo
            valid={str(s['condition_index']) for s in segment_plan()}|{s['condition'] for s in segment_plan()}|{s['segment_id'] for s in segment_plan()}
            if any(s not in valid for s in selected):raise ValueError('重录可填1–9、4b或逗号分隔编号')
            self.port=port;self.stop_event.clear();self.answers=queue.Queue();self.prompt='';self.waiting=False
            self.phase='running';self.error=None;self.logs.clear()
            resume=bool(payload.get('resume',self.resume))
            def run():
                try:
                    def factory():
                        if self.stop_event.is_set():raise KeyboardInterrupt('网页停止')
                        device=self.serial_factory() if self.serial_factory else open_serial(self.port)
                        return InterruptSerial(device,self.stop_event)
                    self.capture=self.capture_factory(self.output,factory,clock=InterruptClock(self.stop_event),input_fn=self.ask,print_fn=self.log,port=self.port)
                    manifest=self.capture.run_session(resume=resume,redo=selected)
                    with self.lock:self.phase='complete' if manifest.get('session_complete') else 'partial'
                except KeyboardInterrupt:
                    with self.lock:self.phase='stopped';self.log('已停止，已有数据保留；可以勾选续录后继续。')
                except BaseException as exc:
                    with self.lock:self.phase='error';self.error=f'{type(exc).__name__}: {exc}';self.log(self.error)
                finally:
                    if self.capture and self.capture.device is not None:
                        self.capture.device.close();self.capture.device=None
                    with self.lock:self.waiting=False
            self.worker=threading.Thread(target=run,name='guided-capture-web',daemon=True);self.worker.start()
    def stop(self):self.stop_event.set()
    def close(self):
        self.closing=True;self.stop()
        self.camera_stop.set()
        if self.camera_worker:self.camera_worker.join(5)
        if self.worker and self.worker.is_alive():self.worker.join(5)
        if self.worker and self.worker.is_alive():
            if self.capture and self.capture.device is not None:self.capture.device.close()
            self.worker.join(2)
            if self.worker.is_alive():raise RuntimeError('采集线程尚未退出，请检查终端')
    def state(self):
        # Read existing collector outputs; never duplicate its serial/decoder pipeline.
        metadata={};manifest={};current=None
        if self.output.exists():
            paths=list(self.output.rglob('metadata.json'))
            if paths:
                current=max(paths,key=lambda p:p.stat().st_mtime_ns)
                try:metadata=json.loads(current.read_text(encoding='utf8'))
                except (OSError,ValueError):pass
            try:manifest=json.loads((self.output/'capture.json').read_text(encoding='utf8'))
            except (OSError,ValueError):pass
        if current:
            frames=current.parent/'frames.jsonl'
            if frames!=self.frame_path:self.frame_path=frames;self.frame_offset=0;self.frame_pending=b'';self.latest=None
            if frames.exists():
                with frames.open('rb') as stream:
                    stream.seek(self.frame_offset);chunk=stream.read();self.frame_offset=stream.tell()
                lines=(self.frame_pending+chunk).split(b'\n');self.frame_pending=lines.pop()
                for line in lines:
                    try:self.latest=json.loads(line)
                    except ValueError:pass
        with self.lock:
            stamp=self.latest.get('host_received_monotonic_ns') if self.latest else None
            return dict(phase=self.phase,prompt=self.prompt,prompt_id=self.prompt_id,waiting=self.waiting,logs=list(self.logs),error=self.error,
                output=str(self.output),port=self.port,resume=self.resume,redo=self.redo,segments=segment_plan(),metadata=metadata,manifest=manifest,
                latest=self.latest,age_ms=max(0,(time.monotonic_ns()-stamp)/1e6) if stamp else None,worker_active=bool(self.worker and self.worker.is_alive()),
                camera=self.camera.snapshot() if self.camera else dict(status='未开启相机',available=False))

def handler_for(bridge):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,status,data,content_type='application/json; charset=utf-8'):
            body=data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf8')
            self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if self.path in ('/','/index.html'):self.send(200,(HERE/'real_labeled_capture.html').read_bytes(),'text/html; charset=utf-8')
            elif self.path=='/m3_camera_view.js':self.send(200,(HERE/'m3_camera_view.js').read_bytes(),'text/javascript; charset=utf-8')
            elif self.path=='/api/camera.mjpeg':
                if not bridge.camera or not bridge.camera.snapshot()['available']:
                    self.send(503,dict(error='等待实时相机'));return
                self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=BAFRAME')
                self.send_header('Cache-Control','no-store');self.send_header('Connection','close');self.end_headers()
                self.close_connection=True;self.connection.settimeout(2)
                try:
                    for header,jpeg in bridge.camera.stream_frames():
                        part=(f'--BAFRAME\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n'
                              f'X-Frame-Sequence: {header["seq"]}\r\nX-Sequence-Id: {header["sequence_id"]}\r\n'
                              f'X-Host-Age-Ms: {header["host_age_ms"]:.3f}\r\nX-Host-Sent-Unix-Ms: {time.time()*1000:.3f}\r\n\r\n').encode('ascii')
                        self.wfile.write(part+jpeg+b'\r\n');self.wfile.flush()
                except OSError:pass
            elif self.path=='/api/state':self.send(200,bridge.state())
            else:self.send(404,dict(error='路径不存在'))
        def do_POST(self):
            expected=f'http://127.0.0.1:{self.server.server_port}'
            if self.headers.get('Origin')!=expected or self.headers.get('Host')!=f'127.0.0.1:{self.server.server_port}':
                self.send(403,dict(error='只接受当前本地页面的操作'));return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16384:raise ValueError('请求大小不正确')
                data=json.loads(self.rfile.read(length))
                if not isinstance(data,dict):raise ValueError('请求必须为对象')
                if self.path=='/api/start':bridge.start(data)
                elif self.path=='/api/input':bridge.reply(data.get('text',''),data.get('prompt_id'))
                elif self.path=='/api/stop':bridge.stop()
                elif self.path=='/api/shutdown':
                    bridge.close();self.send(200,dict(ok=True));threading.Thread(target=self.server.shutdown,daemon=True).start();return
                else:self.send(404,dict(error='路径不存在'));return
                self.send(200,dict(ok=True))
            except (ValueError,OSError,RuntimeError) as exc:self.send(400,dict(error=str(exc)))
    return Handler

def main(argv=None):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description='只打开本地ToF十段采集台，点击开始才连接串口')
    parser.add_argument('--output',type=Path);parser.add_argument('--port',default='COM5');parser.add_argument('--web-port',type=int,default=8791)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--redo',action='append',default=[])
    parser.add_argument('--camera-url',default='http://192.168.5.9:81/stream',help='已连接Atom的局域网MJPEG地址；空字符串禁用相机')
    args=parser.parse_args(argv);output=args.output or default_output_root(ROOT/'artifacts.local/work')
    if not output.resolve().is_relative_to((ROOT/'artifacts.local').resolve()):parser.error('输出必须位于canonical artifacts.local内')
    bridge=Bridge(output,port=args.port,resume=args.resume,redo=args.redo)
    server=ThreadingHTTPServer(('127.0.0.1',args.web_port),handler_for(bridge))
    if args.camera_url:
        preview_root=ROOT/'artifacts.local/work/real-labeled-camera-preview'
        preview_root.mkdir(parents=True,exist_ok=True)
        bridge.camera=CameraFeed(preview_root/uuid.uuid4().hex,bridge.camera_stop,url=args.camera_url,max_seconds=12*3600)
        bridge.camera_worker=threading.Thread(target=bridge.camera.run,name='atom-preview',daemon=True)
        bridge.camera_worker.start()
    print(f'本地采集台：http://127.0.0.1:{server.server_port}/；尚未连接设备',flush=True)
    try:server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:pass
    finally:
        try:bridge.close()
        finally:server.server_close()
    return 0

if __name__=='__main__':raise SystemExit(main())
