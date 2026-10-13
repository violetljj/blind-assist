"""Bounded synthetic test for native JPEG framing; no capture artifacts needed."""
import argparse, ctypes, json, subprocess
from pathlib import Path
cli=argparse.ArgumentParser(description=__doc__)
cli.add_argument('--compiler',required=True,type=Path)
cli.add_argument('--output-dir',required=True,type=Path)
cli.add_argument('--repo-root',type=Path,default=Path.cwd())
args=cli.parse_args()
artifacts=(args.repo_root/'artifacts.local').resolve(); output=args.output_dir.resolve()
assert output!=artifacts and output.is_relative_to(artifacts), 'Output must be below artifacts.local'
output.mkdir(parents=True,exist_ok=True)
here=Path(__file__).resolve().parent
libpath=output/'jpeg_parser_test.dll'
subprocess.run([str(args.compiler),'-std=c99','-Wall','-Wextra','-Werror','-shared','-O2','-nostdlib','-Wl,/noentry','-I'+str(here),str(here/'parser_api.c'),'-o',str(libpath)],check=True)
lib=ctypes.CDLL(str(libpath))
for name in ('parse_chunks','parse_split'):
 f=getattr(lib,name);f.argtypes=[ctypes.c_char_p,ctypes.c_size_t,ctypes.c_size_t];f.restype=ctypes.c_size_t
def parse(data, chunk=1024):
    return lib.parse_chunks(data, len(data), chunk)

def segment(marker, payload):
    return bytes([255, marker]) + (len(payload)+2).to_bytes(2, 'big') + payload

sof = segment(0xC0, bytes([8, 0, 1, 0, 1, 1, 1, 0x11, 0]))
sos = segment(0xDA, bytes([1, 1, 0, 0, 63, 0]))
prefix = b'\xff\xd8' + sof + sos
basic = prefix + b'abc\xff\x00\xd9def\xff\xd0ghi\xff\xff\xd9'
app = b'\xff\xd8' + segment(0xE1, b'fake\xff\xd9\xff\xd8'+bytes(1020)) + sof + sos + b'xyz\xff\xd9'
multiscan = prefix + b'xyz' + segment(0xC4, b'123') + sos + b'\xff\x00\xff\xd7xyz\xff\xd9'
dnl = prefix + b'xyz' + segment(0xDC, b'\x00\x01') + b'xyz\xff\xd9'
cases = [basic, app, multiscan, dnl]
assertions = 0
for data in cases:
    for chunk in (1, 2, 3, 17, 1023, 1024, 1025, len(data)+1):
        assert parse(data, chunk) == len(data)
        assert parse(data + basic, chunk) == len(data)
        assertions += 2
    for split in range(len(data)+1):
        assert lib.parse_split(data, len(data), split) == len(data)
        assertions += 1
    for end in range(len(data)):
        assert parse(data[:end]) == 0
        assertions += 1
for data in [b'', b'\xff\xd8\xff\xd9', b'garbage'+basic,
             b'\xff\xd8\xff\xe1\x00\x01'+basic,
             b'\xff\xd8\xff\xd8'+sof+sos+b'\xff\xd9',
             b'\xff\xd8'+sos+b'\xff\xd9',
             b'\xff\xd8'+sof+b'\xff\x00'+sos+b'\xff\xd9']:
    assert parse(data, 1) == 0
    assertions += 1

result=dict(status='PASS',synthetic_assertions=assertions,limit='Marker framing only; not entropy decoding')
(output/'test-result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
