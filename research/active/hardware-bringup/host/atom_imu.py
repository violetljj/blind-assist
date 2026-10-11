"""Atom IMU receipts, raw evidence and CNH alignment; device clocks stay unmapped.

Alignment uses final-newline host receipt times, not exposure synchronization.
The optical-axis transform requires an explicit mounting record with its
verification scope; an operator pilot zero does not establish absolute accuracy.
"""
from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import socket
import threading
import time

SCHEMA = 'atom.imu.mount.v1'
ALIGNMENT_RULE = 'nearest host final-newline receipt; residual = IMU - CNH'
RECEIPT_CLOCK_DOMAIN = 'HOST_PERF_COUNTER_NS'
receipt_ns = time.perf_counter_ns


def clock_metadata():
    info = time.get_clock_info('perf_counter')
    return dict(domain=RECEIPT_CLOCK_DOMAIN, api='time.perf_counter_ns',
                implementation=info.implementation, resolution_ns=info.resolution*1e9,
                monotonic=info.monotonic, mapping='host-only; device clocks unmapped')


def private_host(host):
    address = ipaddress.IPv4Address(str(host).strip())
    if not any(address in ipaddress.IPv4Network(net) for net in
               ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16')):
        raise ValueError('IMU requires an explicit private IPv4 address')
    return str(address)


def finite(value, key):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{key}: finite number required')
    return float(value)


def validate_frame(sensor):
    if sensor.get('type') != 'imu_frame':
        raise ValueError('imu_frame required')
    if sensor.get('schema_version') != 1:
        raise ValueError('schema_version: expected 1')
    for key,maximum in (('seq',2**32-1),('device_us',2**64-1)):
        if type(sensor.get(key)) is not int or not 0 <= sensor[key] <= maximum:
            raise ValueError(f'{key}: unsigned integer in range required')
    for key in ('device_id', 'boot_id'):
        if not isinstance(sensor.get(key), str) or not sensor[key].strip():
            raise ValueError(f'{key}: device identity required')
    for key in ('pitch_deg', 'roll_deg'):
        finite(sensor.get(key), key)
    for key in ('accel_g', 'gyro_dps', 'gravity_unit'):
        vector = sensor.get(key)
        if not isinstance(vector, list) or len(vector) != 3:
            raise ValueError(f'{key}: three values required')
        for value in vector:
            finite(value, key)
    gravity_norm = math.sqrt(sum(v*v for v in sensor['gravity_unit']))
    if not .95 <= gravity_norm <= 1.05:
        raise ValueError('gravity_unit: approximately unit gravity required')
    for key in ('accel_raw', 'gyro_raw'):
        vector = sensor.get(key)
        if not isinstance(vector, list) or len(vector) != 3 or any(type(v) is not int or not -32768 <= v <= 32767 for v in vector):
            raise ValueError(f'{key}: three signed int16 values required')
    if sensor.get('valid') is not True:
        raise ValueError('IMU filter reports invalid attitude')
    return sensor


def load_mounting(path):
    """Require actual level/down/up/return evidence and an explicit proper basis.

    ``verified`` records completion of the declared procedure, not metrological
    accuracy. OPERATOR_PILOT_ZERO retains the manually established reference and
    its measured return residual without treating it as precise optical truth.
    """
    record = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if record.get('schema') != SCHEMA or record.get('verified') is not True:
        raise ValueError('Verified atom.imu.mount.v1 mounting required')
    for key in ('mounting_id', 'atom_device_id', 'axis_relation', 'evidence'):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f'Mounting missing {key}')
    rotation = record.get('tof_from_imu_rotation')
    if not isinstance(rotation, list) or len(rotation) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in rotation):
        raise ValueError('tof_from_imu_rotation: explicit 3x3 calibrated rotation required')
    for row in rotation:
        for value in row:
            finite(value, 'tof_from_imu_rotation')
    for i in range(3):
        for j in range(3):
            if abs(sum(rotation[i][k]*rotation[j][k] for k in range(3)) - (1 if i == j else 0)) > 1e-3:
                raise ValueError('tof_from_imu_rotation must be orthonormal')
    a, b, c = rotation
    determinant = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
    if abs(determinant-1) > 1e-3:
        raise ValueError('tof_from_imu_rotation must be right-handed (determinant +1)')
    for key in ('level_reference', 'wall_check', 'up_check'):
        if not isinstance(record.get(key), dict) or record[key].get('verified') is not True:
            raise ValueError(f'{key}: actual verified observation required')
    if record.get('verification_level') == 'OPERATOR_PILOT_ZERO':
        if record.get('absolute_optical_pitch_accuracy_verified') is not False:
            raise ValueError('Operator pilot zero must explicitly retain unknown absolute optical accuracy')
        if not isinstance(record.get('calibration_scope'), str) or not record['calibration_scope'].strip():
            raise ValueError('Operator pilot zero requires its calibration scope')
    record['model_pitch_convention'] = 'negative = ToF optical axis downward; positive = upward'
    record['model_roll_convention'] = 'positive right-hand rotation about ToF forward optical axis'
    record['source_path'] = str(Path(path).resolve())
    record['source_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return record


def calibrated_attitude(alignment, mounting):
    unavailable = dict(available=False, source='IMU_UNAVAILABLE', pitch_deg=None, roll_deg=None)
    if not alignment or not alignment.get('available'):
        return dict(unavailable, reason=(alignment or {}).get('status', 'IMU_NOT_CONFIGURED'))
    if mounting is None or mounting.get('verified') is not True:
        return dict(unavailable, reason='MOUNTING_CALIBRATION_REQUIRED')
    sensor = alignment['sensor']
    if sensor['device_id'].lower() != mounting['atom_device_id'].lower():
        return dict(unavailable, reason='MOUNTING_DEVICE_ID_MISMATCH')
    gravity = sensor['gravity_unit']
    transformed = [sum(row[k]*gravity[k] for k in range(3)) for row in mounting['tof_from_imu_rotation']]
    norm = math.sqrt(sum(value*value for value in transformed))
    transformed = [value/norm for value in transformed]
    if math.hypot(transformed[0], transformed[1]) < 1e-3:
        return dict(unavailable, reason='ROLL_UNDEFINED_NEAR_VERTICAL_OPTICAL_AXIS')
    verification = mounting.get('verification_level', 'UNSPECIFIED_VERIFIED_MOUNTING')
    source = ('MEASURED_IMU_OPERATOR_PILOT_ZERO' if verification == 'OPERATOR_PILOT_ZERO'
              else 'MEASURED_IMU_VERIFIED_MOUNTING')
    return dict(available=True, measured=True, source=source,
                pitch_deg=math.degrees(math.asin(max(-1, min(1, transformed[2])))),
                roll_deg=math.degrees(math.atan2(-transformed[0], -transformed[1])),
                gravity_tof=transformed,
                mounting_id=mounting['mounting_id'], mounting_sha256=mounting.get('source_sha256'),
                verification_level=verification,
                calibration_scope=mounting.get('calibration_scope'),
                absolute_optical_pitch_accuracy_verified=mounting.get('absolute_optical_pitch_accuracy_verified'),
                tof_atom_extrinsic_independently_measured=mounting.get('tof_atom_extrinsic_independently_measured'),
                zero_repeat_residual=mounting.get('zero_repeat_residual'),
                installation_reference=mounting.get('installation_reference'),
                imu_seq=sensor['seq'], imu_boot_id=sensor['boot_id'],
                sample_gap=sensor.get('sample_gap', False),
                alignment_residual_ms=alignment['residual_ms'],
                scope='gravity tilt only; no yaw/translation or exposure-clock mapping')


def unit(vector):
    length = math.sqrt(sum(v*v for v in vector))
    if length < 1e-6:
        raise ValueError('Distinct gravity vectors required to infer tilt plane')
    return [v/length for v in vector]


def derive_mounting(level_gravity, down_gravity, *, atom_device_id, mounting_id, evidence):
    """Infer a proper basis from operator-confirmed optical level and down tilt.

    This candidate remains unverified until a separate up pose and frontal wall
    check are recorded. Exact down tilt angle is not needed to infer the plane.
    """
    up, down = unit(level_gravity), unit(down_gravity)
    dot = sum(a*b for a, b in zip(up, down))
    forward = unit([u*dot-d for u, d in zip(up, down)])
    right = unit([forward[1]*up[2]-forward[2]*up[1],
                  forward[2]*up[0]-forward[0]*up[2],
                  forward[0]*up[1]-forward[1]*up[0]])
    return dict(schema=SCHEMA, verified=False, mounting_id=mounting_id,
                atom_device_id=atom_device_id, axis_relation='Optical level gravity defines up; distinct down tilt defines ToF forward tilt plane',
                tof_from_imu_rotation=[right, [-v for v in up], forward],
                level_reference=dict(verified=False, gravity_unit=up, optical_axis_horizontal_operator_confirmed=False),
                down_reference=dict(gravity_unit=down, inferred_down_pitch_deg=math.degrees(math.asin(max(-1,min(1,-math.sqrt(max(0,1-dot*dot))))))),
                wall_check=dict(verified=False), up_check=dict(verified=False), evidence=evidence,
                calibration_scope='Gravity pitch/roll only; operator optical horizontal and tilt plane, no camera/ToF image registration')


class ImuBuffer:
    def __init__(self, maxlen=2000):
        self.records = deque(maxlen=maxlen)
        self.condition = threading.Condition()
        self.epoch = None
        self.previous = None

    def add(self, record):
        sensor = validate_frame(record['sensor'])
        epoch = (sensor['device_id'], sensor['boot_id'])
        with self.condition:
            discontinuity = self.previous is not None and (sensor['seq'] <= self.previous['seq'] or sensor['device_us'] <= self.previous['device_us'])
            if epoch != self.epoch or discontinuity:
                self.records.clear()
                self.epoch = epoch
            self.previous = sensor
            self.records.append(record)
            self.condition.notify_all()

    def clear(self):
        with self.condition:
            self.records.clear()
            self.previous = None
            self.condition.notify_all()

    def nearest(self, cnh_host_ns, *, max_age_ms=100, future_wait_ms=30):
        deadline = time.perf_counter() + max(0, future_wait_ms) / 1000
        with self.condition:
            while future_wait_ms and (not self.records or self.records[-1]['host_received_monotonic_ns'] < cnh_host_ns):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self.condition.wait(remaining)
            if not self.records:
                return dict(available=False, status='IMU_UNAVAILABLE', rule=ALIGNMENT_RULE,
                            residual_ms=None, max_age_ms=max_age_ms, sensor=None)
            # In a receipt tie choose the earlier receipt, then lower sequence.
            record = min(self.records, key=lambda r: (abs(r['host_received_monotonic_ns'] - cnh_host_ns),
                                                     r['host_received_monotonic_ns'], r['sensor']['seq']))
            residual = (record['host_received_monotonic_ns'] - cnh_host_ns) / 1e6
            available = abs(residual) <= max_age_ms
            return dict(available=available, status='ALIGNED' if available else 'IMU_TOO_OLD',
                        rule=ALIGNMENT_RULE, residual_ms=residual, max_age_ms=max_age_ms,
                        host_received_monotonic_ns=record['host_received_monotonic_ns'],
                        host_received_utc=record.get('host_received_utc'),
                        raw_offset=record.get('raw_offset'), raw_length=record.get('raw_length'),
                        sensor=record['sensor'] if available else None,
                        clock_mapping='Atom and XIAO monotonic clocks remain independent/unmapped',
                        synchronization_scope='receipt proximity; network and sensor latency remain unmeasured')


class ImuDecoder:
    def __init__(self, emit):
        self.emit = emit
        self.pending = bytearray()
        self.offset = 0
        self.counts = Counter()

    def feed(self, chunk, host_ns, host_utc=None):
        self.pending.extend(chunk)
        while b'\n' in self.pending:
            line, _, rest = self.pending.partition(b'\n')
            self.pending = bytearray(rest)
            length = len(line) + 1
            record = dict(host_received_monotonic_ns=host_ns, host_received_utc=host_utc,
                          host_receipt_clock_domain=RECEIPT_CLOCK_DOMAIN,
                          raw_offset=self.offset, raw_length=length)
            self.offset += length
            try:
                sensor = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                if not isinstance(sensor, dict):
                    raise ValueError('JSON object required')
                record['sensor'] = sensor
                if sensor.get('type') == 'imu_frame':
                    validate_frame(sensor)
                    record['valid_frame'] = True
                    self.counts['frames'] += 1
                else:
                    record['valid_frame'] = False
                    self.counts['events'] += 1
            except (ValueError, UnicodeError) as exc:
                record.update(valid_frame=False, issue=str(exc))
                self.counts['invalid_lines'] += 1
            self.emit(record)


class ImuFeed:
    """Explicit endpoint only; one reader, continuous exact-byte evidence, retry."""
    def __init__(self, host, output, port=3336, *, mounting=None, socket_factory=socket.create_connection):
        self.host = private_host(host)
        self.port = int(port)
        if not 1 <= self.port <= 65535:
            raise ValueError('IMU TCP port must be 1..65535')
        self.output = Path(output)
        self.mounting = mounting
        self.socket_factory = socket_factory
        self.buffer = ImuBuffer()
        self.stop = threading.Event()
        self.worker = None
        self.socket = None
        self.lock = threading.Lock()
        self.status = 'NOT_STARTED'
        self.error = None
        self.frames = 0
        self.raw_bytes = 0
        self.last_record = None
        self.reconnects = 0

    def start(self):
        if self.worker is not None:
            raise ValueError('IMU feed is already started')
        self.worker = threading.Thread(target=self.run, name='atom-imu-reader', daemon=True)
        self.worker.start()
        return self

    def run(self):
        self.output.mkdir(parents=True, exist_ok=False)
        digest = hashlib.sha256()
        try:
            with (self.output / 'raw.bin').open('xb') as raw, (self.output / 'receipts.jsonl').open('x', encoding='utf-8') as log:
                def emit(record):
                    log.write(json.dumps(record, allow_nan=False) + '\n')
                    log.flush()
                    if record.get('valid_frame'):
                        self.buffer.add(record)
                        with self.lock:
                            self.frames += 1
                            self.last_record = record
                            self.status, self.error = 'STREAMING', None
                    elif record.get('issue') or (record.get('sensor', {}).get('type') == 'imu_status' and record['sensor'].get('ready') is not True):
                        self.buffer.clear()
                        with self.lock:
                            self.status = 'IMU_INVALID'
                            self.error = record.get('issue') or record['sensor'].get('init_status')
                while not self.stop.is_set():
                    decoder = ImuDecoder(emit)
                    decoder.offset = self.raw_bytes
                    try:
                        self.socket = self.socket_factory((self.host, self.port), timeout=3)
                        self.socket.settimeout(.2)
                        last_received = time.perf_counter()
                        with self.lock:
                            self.status = 'CONNECTED'
                        while not self.stop.is_set():
                            try:
                                chunk = self.socket.recv(16384)
                            except socket.timeout:
                                if time.perf_counter() - last_received > 1.5:
                                    raise OSError('IMU stream silent for 1.5 seconds')
                                continue
                            host_ns = receipt_ns()
                            host_utc = datetime.now(timezone.utc).isoformat()
                            if not chunk:
                                raise OSError('IMU TCP peer disconnected')
                            last_received = time.perf_counter()
                            raw.write(chunk)
                            raw.flush()
                            digest.update(chunk)
                            self.raw_bytes += len(chunk)
                            decoder.feed(chunk, host_ns, host_utc)
                    except (OSError, ValueError) as exc:
                        if not self.stop.is_set():
                            with self.lock:
                                self.status, self.error = 'RECONNECTING', str(exc)
                                self.reconnects += 1
                            self.buffer.clear()
                    finally:
                        if self.socket is not None:
                            self.socket.close()
                            self.socket = None
                        if decoder.pending:
                            log.write(json.dumps(dict(issue='incomplete_connection_tail', raw_offset=self.raw_bytes-len(decoder.pending),
                                                      raw_length=len(decoder.pending), host_received_monotonic_ns=receipt_ns(),
                                                      host_receipt_clock_domain=RECEIPT_CLOCK_DOMAIN))+'\n')
                            log.flush()
                    if self.stop.wait(1):
                        break
        except BaseException as exc:
            with self.lock:
                self.status, self.error = 'FAILED', str(exc)
        finally:
            receipt = self.snapshot()
            receipt.update(raw_sha256=digest.hexdigest(), raw_bytes=self.raw_bytes,
                           host_receipt_clock=clock_metadata(),
                           format='exact TCP bytes; receipts index final-newline host timestamps',
                           clock_mapping='EXTERNAL_DEVICE_MONOTONIC_UNMAPPED')
            (self.output / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    def align(self, cnh_host_ns, *, wait=True):
        alignment = self.buffer.nearest(cnh_host_ns, future_wait_ms=30 if wait else 0)
        alignment['raw_path'] = str(self.output / 'raw.bin')
        alignment['host_receipt_clock_domain'] = RECEIPT_CLOCK_DOMAIN
        alignment['attitude'] = calibrated_attitude(alignment, self.mounting)
        return alignment

    def snapshot(self):
        with self.lock:
            record = self.last_record
            age = (receipt_ns()-record['host_received_monotonic_ns'])/1e6 if record else None
            available = self.status == 'STREAMING' and record is not None and age <= 100
            return dict(status=self.status, error=self.error, host=self.host, port=self.port,
                        frames=self.frames, reconnects=self.reconnects, age_ms=age,
                        available=available,
                        sensor=record['sensor'] if available else None,
                        output=str(self.output), mounting=self.mounting, host_receipt_clock=clock_metadata())

    def close(self):
        self.stop.set()
        link = self.socket
        if link is not None:
            link.close()
        if self.worker is not None:
            self.worker.join(5)
            if self.worker.is_alive():
                raise RuntimeError('IMU reader has not released its socket')
        self.buffer.clear()
        with self.lock:
            self.status = 'CLOSED'


def unavailable_alignment():
    return dict(available=False, status='IMU_NOT_CONFIGURED', rule=ALIGNMENT_RULE,
                residual_ms=None, sensor=None, attitude=calibrated_attitude(None, None))
