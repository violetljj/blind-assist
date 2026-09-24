"""Unified queue-4 exporter for RGB/depth pairs and float32 attribute batches.

One frame is two pair jobs plus one four-attribute job. Caller must capture all
targets, enqueue them in render order, then may prepare the next frame. Always
finish/drain before destroying targets or accepting a shard. native_probe writes
same-target synchronous references and compares exact NPY bytes after drain.
"""
from array import array
import ast
import hashlib
from pathlib import Path
import struct
import sys
import time
from ue_pair_export import PairExporter


def write_rgb_reference(u, world, target, path):
    values = u.RenderingLibrary.read_render_target_raw(world, target, normalize=False)
    width, height = target.size_x, target.size_y
    if len(values) != width*height or sys.byteorder != 'little':
        raise ValueError('Attribute reference dimensions/endianness differ')
    header = str(dict(descr='<f4', fortran_order=False, shape=(height, width, 3)))
    header += ' '*((64-(10+len(header)+1)%64)%64)+'\n'
    raw = header.encode('ascii')
    with Path(path).open('xb') as stream:
        stream.write(b'\x93NUMPY\x01\x00'+struct.pack('<H', len(raw))+raw)
        array('f', (v for p in values for v in (p.r, p.g, p.b))).tofile(stream)


def validate_npy_payload(path, shape):
    path = Path(path)
    with path.open('rb') as stream:
        if stream.read(8) != b'\x93NUMPY\x01\x00':
            raise ValueError('Expected NPY v1 float32: '+str(path))
        length = struct.unpack('<H', stream.read(2))[0]
        header = ast.literal_eval(stream.read(length).decode('ascii').strip())
    if header != dict(descr='<f4', fortran_order=False, shape=tuple(shape)):
        raise ValueError('Attribute NPY schema differs: '+str(path))
    elements = 1
    for extent in shape:
        elements *= extent
    if path.stat().st_size != 10+length+4*elements:
        raise ValueError('Attribute NPY payload byte count differs: '+str(path))
    return dict(bytes=path.stat().st_size, shape=list(shape), dtype='<f4')


class AttributePairExporter(PairExporter):
    def __init__(self, u, mode='native_async', limit=4, probe_indices=()):
        if limit != 4:
            raise ValueError('Only the existing four-job queue is authorized')
        if not hasattr(u.BlindAssistCaptureLibrary, 'submit_capture_npy_batch'):
            raise RuntimeError('Native v6 attribute batch API required')
        super().__init__(u, mode=mode, limit=limit)
        self.attribute_batches = []
        self.probe_indices = frozenset(probe_indices)

    def ready(self, jobs_needed=3):
        """Reserve capacity for a complete frame BEFORE any capture or submission."""
        if not 1 <= jobs_needed <= self.limit:
            raise ValueError('Invalid per-tick job requirement')
        profile = self.profile()
        if profile['failed']:
            raise RuntimeError('Unified GPU export failed: '+profile['last_error'])
        return profile['pending']+jobs_needed <= self.limit

    def export_attributes(self, world, targets, paths, depth_metres, index):
        if not (1 <= len(targets) <= 4 and len(targets) == len(paths) == len(depth_metres)):
            raise ValueError('Expected matching 1..4 attribute targets/paths/modes')
        paths = [Path(p) for p in paths]
        if len({str(p.resolve()).lower() for p in paths}) != len(paths):
            raise ValueError('Duplicate attribute destinations')
        references = []
        probe = self.mode == 'native_probe' or index in self.probe_indices
        for target, path, depth in zip(targets, paths, depth_metres):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() or Path(str(path)+'.partial').exists():
                raise FileExistsError(path)
            shape = (target.size_y, target.size_x) if depth else (target.size_y, target.size_x, 3)
            reference = path.with_name(path.stem+'.reference.npy')
            if probe:
                if reference.exists():
                    raise FileExistsError(reference)
                if depth:
                    if not self.u.BlindAssistCaptureLibrary.export_depth_npy(world, target, str(reference)):
                        raise RuntimeError('Synchronous depth reference failed')
                else:
                    write_rgb_reference(self.u, world, target, reference)
            references.append(dict(path=str(path), reference=str(reference) if probe else None,
                shape=shape, units='metres_axial_invalid_zero' if depth else 'unchanged_render_target_RGB'))
        started = time.perf_counter()
        if not self.u.BlindAssistCaptureLibrary.submit_capture_npy_batch(targets, [str(p) for p in paths], depth_metres, self.limit):
            raise RuntimeError('Attribute batch submission failed; drain before release')
        row = dict(sample_index=index, files=references, submission_s=time.perf_counter()-started)
        self.attribute_batches.append(row)
        return row

    def finish(self):
        profile = self.profile(drain=True)
        expected = len(self.rows)+len(self.attribute_batches)
        if profile['pending'] or profile['failed'] or profile['submitted'] != expected or profile['completed'] != expected:
            raise RuntimeError('Incomplete unified GPU queue: '+str(profile))
        for row in self.attribute_batches:
            for entry in row['files']:
                path = Path(entry['path'])
                entry['validation'] = validate_npy_payload(path, entry['shape'])
                if entry['reference']:
                    actual = path.read_bytes(); reference = Path(entry['reference']).read_bytes()
                    if actual != reference:
                        raise ValueError('Asynchronous attributes differ from same-target reference: '+str(path))
                    entry['same_target_bytes_equal'] = True
                    entry['sha256'] = hashlib.sha256(actual).hexdigest()
        return dict(mode=self.mode, queue_limit=self.limit, shared_queue=True, profile=profile,
            pair_rows=self.rows, attribute_batches=self.attribute_batches,
            probe_indices=sorted(self.probe_indices),
            acceptance='ALL_ATTRIBUTE_BYTE_PARITY' if self.mode=='native_probe' else 'SAMPLED_ATTRIBUTE_BYTE_PARITY' if self.probe_indices else 'SCHEMA_AND_DRAIN_ONLY_NOT_PARITY')
