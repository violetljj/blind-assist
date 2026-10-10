"""Latest-only Atom MJPEG preview; independent from M3 inputs and decisions."""
from __future__ import annotations

from collections import deque
import io
import ipaddress
import json
from pathlib import Path
import re
import socket
import threading
import time
import urllib.request

from atom_capture import ProtocolError, validate_jpeg
from wifi_camera_capture import Body, NoRedirect, StopCapture, TransportError, receive, validate_url


def discover(device_id, seconds=2):
    """Discover the explicitly selected camera on local IPv4 interfaces."""
    addresses = {row[4][0] for row in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
    addresses = [ip for ip in addresses if any(ipaddress.ip_address(ip) in ipaddress.ip_network(net)
                 for net in ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'))]
    links = []
    try:
        for ip in addresses:
            link = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            links.append(link)
            link.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            link.bind((ip, 0))
            link.settimeout(.1)
            link.sendto(b'BADEMO_DISCOVER_V1', ('255.255.255.255', 3334))
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            for link in links:
                try:
                    data, source = link.recvfrom(4096)
                    identity = json.loads(data)
                    if identity.get('role') == 'camera' and identity.get('device_id', '').lower() == device_id.lower():
                        url = validate_url(f'http://{source[0]}:81/stream')
                        return url, identity
                except (socket.timeout, ValueError, AttributeError):
                    continue
            if not links:
                time.sleep(.1)
        return None, None
    finally:
        for link in links:
            link.close()


class RecentBytes:
    """Bounded recent body evidence, retained on a protocol failure only."""
    def __init__(self):
        self.pending = bytearray()
        self.count = 0

    def write(self, data):
        self.count += len(data)
        self.pending.extend(data)
        if len(self.pending) > 4*1024*1024:
            del self.pending[:-4*1024*1024]

    def flush(self):
        pass


class PreviewBody(Body):
    def __init__(self, *args, stop, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop = stop

    def check(self):
        if self.stop.is_set():
            raise StopCapture()
        super().check()


class CameraFeed:
    def __init__(self, output, stop, *, url=None, device_id=None, max_seconds=3600):
        self.output, self.stop = Path(output), stop
        self.url = validate_url(url) if url else None
        self.device_id = device_id
        self.deadline = time.monotonic()+max_seconds
        self.lock = threading.Lock()
        self.latest = None
        self.receipts = deque(maxlen=50)
        self.state = dict(status='寻找相机', frames=0, reconnects=0, error=None, released=False)

    def snapshot(self):
        with self.lock:
            result = dict(self.state)
            age = (time.monotonic_ns()-self.latest[2])/1e6 if self.latest else None
            result.update(age_ms=age, available=age is not None and age < 1500)
            if age is not None and age >= 1500 and result['status'] == '实时画面':
                result['status'] = '画面中断'
            return result

    def image(self):
        with self.lock:
            if self.latest and time.monotonic_ns()-self.latest[2] < 1_500_000_000:
                return self.latest[1], dict(self.latest[0])
            return None, None

    def run(self):
        self.output.mkdir()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect()).open
        failures = 0
        first = True
        previous = None
        try:
            with (self.output/'frames.jsonl').open('x', encoding='utf-8') as records, (self.output/'events.jsonl').open('x', encoding='utf-8') as events:
                while not self.stop.is_set() and time.monotonic() < self.deadline:
                    if not self.url:
                        url, identity = discover(self.device_id)
                        if not url:
                            with self.lock:
                                self.state.update(status='等待相机接入同一网络', error=None)
                            self.stop.wait(3)
                            continue
                        self.url = url
                        events.write(json.dumps(dict(kind='discovered', identity=identity, url=url))+'\n')
                        events.flush()
                    recent = RecentBytes()
                    try:
                        with opener(self.url, timeout=3) as response:
                            content_type = response.headers.get('Content-Type', '')
                            match = re.search(r'boundary="?([A-Za-z0-9_-]{1,70})"?(?:;|$)', content_type)
                            if not content_type.lower().startswith('multipart/x-mixed-replace') or not match:
                                raise ProtocolError('Invalid MJPEG Content-Type/boundary')
                            body = PreviewBody(response, recent, self.deadline, stop=self.stop)
                            with self.lock:
                                self.state.update(status='连接相机', url=self.url)
                                self.receipts.clear()
                            while not self.stop.is_set():
                                header, jpeg = receive(body, match.group(1).encode('ascii'))
                                stamp = time.monotonic_ns()
                                from PIL import Image
                                try:
                                    with Image.open(io.BytesIO(jpeg)) as image:
                                        header['width'], header['height'] = image.size
                                    validate_jpeg(jpeg, header)
                                except Exception as exc:
                                    raise ProtocolError(f'Invalid JPEG: {exc}') from exc
                                if previous and previous[0] == header['sequence_id']:
                                    step = (header['seq']-previous[1]) % 2**32
                                    if not 0 < step < 2**31:
                                        raise ProtocolError('Frozen or regressing camera sequence')
                                if previous and previous[0] != header['sequence_id']:
                                    self.receipts.clear()
                                previous = (header['sequence_id'], header['seq'])
                                records.write(json.dumps(dict(header=header, host_received_monotonic_ns=stamp,
                                    scope='preview only; independent clocks, no exposure synchronization'))+'\n')
                                records.flush()
                                if first:
                                    (self.output/'first-frame.jpg').write_bytes(jpeg)
                                    first = False
                                with self.lock:
                                    self.latest = (header, jpeg, stamp)
                                    self.receipts.append(stamp)
                                    fps = (len(self.receipts)-1)*1e9/(stamp-self.receipts[0]) if len(self.receipts)>1 else None
                                    self.state.update(status='实时画面', frames=self.state['frames']+1,
                                                      seq=header['seq'], sequence_id=header['sequence_id'], fps=fps, error=None)
                    except StopCapture:
                        break
                    except (OSError, TransportError) as exc:
                        failures += 1
                        events.write(json.dumps(dict(kind='disconnect', error=str(exc), url=self.url,
                                                     received_ns=time.monotonic_ns()))+'\n')
                        events.flush()
                        with self.lock:
                            self.latest = None
                            self.state.update(status='重连相机', reconnects=failures, error=str(exc))
                        if self.device_id:
                            self.url = None  # DHCP may have changed.
                        self.stop.wait(min(5, failures))
                    except Exception as exc:
                        (self.output/'rejected-recent-body.bin').write_bytes(recent.pending)
                        with self.lock:
                            self.latest = None
                            self.state.update(status='相机数据错误', error=str(exc))
                        break
        finally:
            with self.lock:
                if self.latest:
                    (self.output/'last-frame.jpg').write_bytes(self.latest[1])
                self.latest = None
                self.state['released'] = True
                if self.stop.is_set():
                    self.state['status'] = '画面已结束'
            (self.output/'final-status.json').write_text(json.dumps(self.snapshot(), indent=2, ensure_ascii=False), encoding='utf-8')
