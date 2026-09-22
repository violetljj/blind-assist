"""Verify full flash identity against retained current app and produce flash plan."""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def verify(backup, old_app, new_app):
    flash = backup.read_bytes()
    assert len(flash) == 8 * 1024 * 1024, "Expected full 8 MB backup"
    partitions = []
    for offset in range(0x8000, 0x8C00, 32):
        magic, kind, subtype, start, size, label, flags = struct.unpack("<HBBII16sI", flash[offset:offset + 32])
        if magic != 0x50AA:
            break
        partitions.append(dict(type=kind, subtype=subtype, offset=start, size=size,
                               label=label.split(b"\0")[0].decode("ascii")))
    app = next(p for p in partitions if p["label"] == "app0")
    before, after = old_app.read_bytes(), new_app.read_bytes()
    assert app["offset"] == 0x10000
    assert flash[app["offset"]:app["offset"] + len(before)] == before, "Current app differs from retained USB build"
    assert len(after) <= app["size"], "App exceeds existing partition"
    return dict(backup_bytes=len(flash), backup_sha256=hashlib.sha256(flash).hexdigest(),
                original_app_matches=True, partitions=partitions,
                new_app_bytes=len(after), new_app_sha256=hashlib.sha256(after).hexdigest(),
                write_scope="app0 only at 0x10000; NVS and partitions preserved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--old-app", type=Path, required=True)
    parser.add_argument("--new-app", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.backup, args.old_app, args.new_app)
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
