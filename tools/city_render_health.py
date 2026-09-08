"""Conservative admission check for observed missing Nanite/VT render resources.

No matching messages is not proof of complete visibility. Any match makes the
whole capture unsuitable for ready data: logs do not identify affected pixels.
Source-only reconnaissance may retain the images with this explicit review flag.
"""
import hashlib
from pathlib import Path


def inspect_log(text):
    counts = {'nanite_invalid_resource': 0, 'virtual_texture_ddc_fetch_failed': 0}
    examples = []
    for number, line in enumerate(text.splitlines(), 1):
        kind = None
        if 'LogNaniteStreaming:' in line and 'marking resource invalid' in line:
            kind = 'nanite_invalid_resource'
        elif 'LogVTDiskCache:' in line and 'Failed to fetch data from DDC' in line:
            kind = 'virtual_texture_ddc_fetch_failed'
        if kind:
            counts[kind] += 1
            if len(examples) < 64:
                examples.append(dict(line=number, kind=kind, message=line))
    return dict(schema='city-render-resource-health-v1',
                status='REVIEW' if any(counts.values()) else 'NO_MATCHING_RESOURCE_ERRORS',
                ready_data_eligible=not any(counts.values()), counts=counts, examples=examples,
                scope='Observed missing-resource signatures only; no pixel/instance visibility certification')


def inspect_capture(capture):
    path = Path(capture) / 'editor.log'
    data = path.read_bytes()
    result = inspect_log(data.decode('utf-8', errors='replace'))
    result['log_sha256'] = hashlib.sha256(data).hexdigest()
    return result
