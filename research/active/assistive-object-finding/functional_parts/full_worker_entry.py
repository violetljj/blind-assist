"""Run the sealed experiment on a worker without rewriting its input manifests.

Only declared absolute path prefixes and Python executable locations are mapped.
The original trainer, manifest bytes, tensor checkpoint and scientific settings
stay intact. This is host portability, not a claim of bitwise cross-GPU replay.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


def relocate(value, mappings):
    if isinstance(value, str):
        normal = value.replace('\\', '/')
        for old, new in mappings:
            prefix = old.replace('\\', '/').rstrip('/')
            if normal.lower() == prefix.lower() or normal.lower().startswith(prefix.lower() + '/'):
                return new.rstrip('/\\') + normal[len(prefix):]
        return value
    if isinstance(value, list):
        return [relocate(item, mappings) for item in value]
    if isinstance(value, tuple):
        return tuple(relocate(item, mappings) for item in value)
    if isinstance(value, dict):
        return {relocate(key, mappings): relocate(item, mappings) for key, item in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--script', type=Path, required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    original_loads = json.loads
    spec = original_loads(args.mapping.read_text(encoding='utf-8-sig'))
    mappings = sorted(spec['prefixes'], key=lambda pair: len(pair[0]), reverse=True)
    script = Path(relocate(str(args.script), mappings)).resolve()
    source_dir = script.parent
    sys.path.insert(0, str(source_dir))
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    sys.argv = [str(script), *relocate(arguments, mappings)]
    # Hashes are computed by the unchanged scripts from raw files. Mapping only
    # happens after JSON decoding, so rows and frozen SHA checks are preserved.
    def portable_loads(*positional, **keyword):
        return relocate(original_loads(*positional, **keyword), mappings)
    json.loads = portable_loads

    # Every child in the existing queue must inherit the same adapter. Restrict
    # this to this experiment's full_*.py commands; leave all other commands alone.
    original_popen = subprocess.Popen
    entry = str(Path(__file__).resolve())
    class PortablePopen(original_popen):
        def __init__(self, command, *positional, **keyword):
            if isinstance(command, (list, tuple)):
                command = list(command)
                target = next((i for i, part in enumerate(command[1:], 1)
                               if str(part).endswith('.py')), None)
                if target is not None and Path(str(command[target])).name.startswith('full_'):
                    command = [*command[:target], entry, '--mapping', str(args.mapping.resolve()),
                               '--script', str(command[target]), '--', *command[target + 1:]]
            super().__init__(command, *positional, **keyword)
    # asyncio.windows_utils subclasses Popen during Torch import. Keep a real
    # class, and scope this hook to the controller that actually launches jobs.
    if script.stem == 'full_run_queue':
        subprocess.Popen = PortablePopen

    if script.stem in ('full_train_baselines', 'full_transfer_export'):
        import torch
        original_torch_load = torch.load
        def portable_torch_load(*positional, **keyword):
            state = original_torch_load(*positional, **keyword)
            if isinstance(state, dict) and isinstance(state.get('config'), dict):
                # Only host path strings in config; do not touch tensor/RNG state.
                state['config'] = relocate(state['config'], mappings)
            return state
        torch.load = portable_torch_load

    receipt = {'pid': os.getpid(), 'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'mapping_sha256': hashlib.sha256(args.mapping.read_bytes()).hexdigest(),
               'script': str(script), 'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
               'arguments': sys.argv[1:], 'scope': 'path-only runtime adapter; frozen raw inputs unchanged'}
    print(json.dumps({'worker_adapter': receipt}), flush=True)
    if script.stem == 'full_train_baselines':
        # Import by name so Windows spawn can unpickle the IIT dataset class.
        importlib.import_module(script.stem).main()
    else:
        runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
